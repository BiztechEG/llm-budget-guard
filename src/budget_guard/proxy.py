"""A transparent stand-in for an SDK client that routes billable calls through BudgetGuard."""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Optional, Tuple

from .context import CallContext, current_context
from .streams import (
    ACCUMULATORS,
    AsyncTrackedStream,
    TrackedStream,
    TrackedStreamManager,
    is_injected_usage_chunk,
)

if TYPE_CHECKING:
    from .core import BudgetGuard

Path = Tuple[str, ...]

# Billable methods, by attribute path from the client, and how to read their usage.
# A "_helper" suffix marks `.stream()` context-manager helpers.
TRACKED: Dict[str, Dict[Path, str]] = {
    "openai": {
        ("chat", "completions", "create"): "openai_chat",
        ("chat", "completions", "parse"): "openai_chat",
        ("chat", "completions", "stream"): "openai_chat_helper",
        ("completions", "create"): "openai_chat",
        ("responses", "create"): "openai_responses",
        ("responses", "parse"): "openai_responses",
        ("responses", "stream"): "openai_responses_helper",
    },
    "anthropic": {
        ("messages", "create"): "anthropic",
        ("messages", "parse"): "anthropic",
        ("messages", "stream"): "anthropic_helper",
        ("beta", "messages", "create"): "anthropic",
        ("beta", "messages", "parse"): "anthropic",
        ("beta", "messages", "stream"): "anthropic_helper",
    },
}

_PREFIXES: Dict[str, FrozenSet[Path]] = {
    provider: frozenset(path[:i] for path in paths for i in range(1, len(path)))
    for provider, paths in TRACKED.items()
}

# Client methods that return a new client; their result gets guarded too.
_CLIENT_FACTORIES = ("with_options", "copy")


def detect_provider(client: Any) -> str:
    module = type(client).__module__.split(".")[0]
    if module in TRACKED:
        return module
    raise TypeError(
        f"Can't tell which provider {type(client).__name__} belongs to; pass provider='openai' or 'anthropic'."
    )


class GuardedClient:
    """Behaves like the wrapped client. Only the methods in TRACKED are intercepted."""

    def __init__(
        self,
        target: Any,
        guard: "Optional[BudgetGuard]",
        provider: str,
        context: CallContext,
        path: Path = (),
    ):
        if provider not in TRACKED:
            raise ValueError(f"Unsupported provider {provider!r}; expected one of {sorted(TRACKED)}")
        # Stored via object.__setattr__ so they never collide with SDK attribute names.
        object.__setattr__(self, "_bg_target", target)
        object.__setattr__(self, "_bg_guard_obj", guard)
        object.__setattr__(self, "_bg_provider", provider)
        object.__setattr__(self, "_bg_context", context)
        object.__setattr__(self, "_bg_path", path)

    @property
    def _bg_guard(self) -> "BudgetGuard":
        from .core import get_default_guard

        # None means "the shared guard", looked up per call so configure() applies to existing clients.
        return self._bg_guard_obj or get_default_guard()

    @property
    def unwrapped(self) -> Any:
        """The original SDK object."""
        return self._bg_target

    def with_context(self, *, user: Optional[str] = None, feature: Optional[str] = None) -> "GuardedClient":
        """A copy of this client attributed to a different user and/or feature."""
        return GuardedClient(
            self._bg_target,
            self._bg_guard_obj,
            self._bg_provider,
            self._bg_context.merge(CallContext(user, feature)),
            self._bg_path,
        )

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._bg_target, name)
        path = self._bg_path + (name,)
        kind = TRACKED[self._bg_provider].get(path)
        if kind is not None:
            return self._bg_wrap_method(attr, kind)
        if path in _PREFIXES[self._bg_provider]:
            return GuardedClient(attr, self._bg_guard_obj, self._bg_provider, self._bg_context, path)
        if not self._bg_path and name in _CLIENT_FACTORIES and callable(attr):
            return self._bg_wrap_factory(attr)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._bg_target, name, value)

    def __repr__(self) -> str:
        return f"<guarded {self._bg_target!r}>"

    def _bg_wrap_factory(self, factory: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(factory)
        def call(*args: Any, **kwargs: Any) -> Any:
            return GuardedClient(factory(*args, **kwargs), self._bg_guard_obj, self._bg_provider, self._bg_context)

        return call

    def _bg_wrap_method(self, method: Callable[..., Any], kind: str) -> Callable[..., Any]:
        provider = self._bg_provider

        @functools.wraps(method)
        def call(*args: Any, **kwargs: Any) -> Any:
            guard = self._bg_guard
            ctx = self._bg_context.merge(current_context())
            hide_usage_chunk = _prepare_kwargs(kind, kwargs)
            guard.before_call(provider, kind, kwargs, ctx)

            fallback_model = kwargs.get("model") if isinstance(kwargs.get("model"), str) else None

            def on_done(response_like: Any) -> None:
                guard.record_response(response_like, provider, ctx, fallback_model)

            def handle(result: Any) -> Any:
                if kind.endswith("_helper"):
                    return TrackedStreamManager(result, on_done)
                if kwargs.get("stream") is True:
                    acc = ACCUMULATORS[kind]()
                    skip = is_injected_usage_chunk if hide_usage_chunk else None
                    wrapper = AsyncTrackedStream if hasattr(result, "__aiter__") else TrackedStream
                    return wrapper(result, acc, on_done, skip)
                guard.record_response(result, provider, ctx, fallback_model)
                return result

            result = method(*args, **kwargs)
            if inspect.isawaitable(result):

                async def finish() -> Any:
                    return handle(await result)

                return finish()
            return handle(result)

        return call


def _prepare_kwargs(kind: str, kwargs: Dict[str, Any]) -> bool:
    """Ask OpenAI chat streams to report usage. Returns True if the caller hadn't asked,
    meaning the extra usage-only chunk should be hidden from them."""
    streaming = kwargs.get("stream") is True or kind == "openai_chat_helper"
    if not (kind.startswith("openai_chat") and streaming):
        return False
    options = kwargs.get("stream_options")
    options = dict(options) if isinstance(options, dict) else {}
    already_asked = options.get("include_usage") is True
    options["include_usage"] = True
    kwargs["stream_options"] = options
    return not already_asked and kind == "openai_chat"
