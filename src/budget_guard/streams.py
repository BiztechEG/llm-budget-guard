"""Watch streamed responses so their cost is recorded once they finish."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .usage import _get

# Called with a response-like {"model": ..., "usage": ...}, or None if no usage was seen.
OnDone = Callable[[Optional[Dict[str, Any]]], None]


class _OpenAIChatAccumulator:
    """Usage arrives on the last chunk when `stream_options.include_usage` is set."""

    def __init__(self) -> None:
        self.result: Optional[Dict[str, Any]] = None

    def feed(self, chunk: Any) -> None:
        usage = _get(chunk, "usage")
        if usage is not None:
            self.result = {"model": _get(chunk, "model"), "usage": usage}


class _OpenAIResponsesAccumulator:
    _FINAL = {"response.completed", "response.incomplete", "response.failed"}

    def __init__(self) -> None:
        self.result: Optional[Dict[str, Any]] = None

    def feed(self, event: Any) -> None:
        if _get(event, "type") in self._FINAL:
            response = _get(event, "response")
            if _get(response, "usage") is not None:
                self.result = {"model": _get(response, "model"), "usage": _get(response, "usage")}


class _AnthropicAccumulator:
    """`message_start` carries input usage; each `message_delta` carries cumulative totals."""

    _FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")

    def __init__(self) -> None:
        self.model: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None

    def feed(self, event: Any) -> None:
        kind = _get(event, "type")
        if kind == "message_start":
            message = _get(event, "message")
            self.model = _get(message, "model")
            usage = _get(message, "usage")
            self.usage = {f: _get(usage, f) for f in self._FIELDS}
        elif kind == "message_delta":
            usage = _get(event, "usage")
            if self.usage is None:
                self.usage = {}
            for f in self._FIELDS:
                value = _get(usage, f)
                if value is not None:
                    self.usage[f] = value

    @property
    def result(self) -> Optional[Dict[str, Any]]:
        if self.usage is None:
            return None
        return {"model": self.model, "usage": self.usage}


ACCUMULATORS = {
    "openai_chat": _OpenAIChatAccumulator,
    "openai_responses": _OpenAIResponsesAccumulator,
    "anthropic": _AnthropicAccumulator,
}


def is_injected_usage_chunk(chunk: Any) -> bool:
    """The extra usage-only chunk we asked OpenAI for; callers didn't, so hide it."""
    return not _get(chunk, "choices") and _get(chunk, "usage") is not None


class _TrackedBase:
    def __init__(self, stream: Any, accumulator: Any, on_done: OnDone, skip: Optional[Callable[[Any], bool]] = None):
        self._bg_stream = stream
        self._bg_acc = accumulator
        self._bg_on_done = on_done
        self._bg_skip = skip
        self._bg_done = False

    def _bg_finish(self) -> None:
        if not self._bg_done:
            self._bg_done = True
            self._bg_on_done(self._bg_acc.result)

    def _bg_keep(self, item: Any) -> bool:
        self._bg_acc.feed(item)
        return not (self._bg_skip and self._bg_skip(item))

    def __del__(self) -> None:
        # A stream dropped part-way (say, the browser disconnected) was still billed: count what it reported.
        try:
            self._bg_finish()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_bg_"):  # not set yet (copy, unpickle, failed __init__); don't recurse
            raise AttributeError(name)
        return getattr(self._bg_stream, name)


class TrackedStream(_TrackedBase):
    """Iterates exactly like the SDK stream it wraps; records cost at the end or on close."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._bg_iter = iter(self._bg_stream)

    def __iter__(self) -> "TrackedStream":
        return self

    def __next__(self) -> Any:
        while True:
            try:
                item = next(self._bg_iter)
            except BaseException:
                self._bg_finish()
                raise
            if self._bg_keep(item):
                return item

    def __enter__(self) -> "TrackedStream":
        return self

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._bg_stream.__exit__(*exc)
        finally:
            self._bg_finish()

    def close(self) -> None:
        try:
            self._bg_stream.close()
        finally:
            self._bg_finish()


class AsyncTrackedStream(_TrackedBase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._bg_iter = self._bg_stream.__aiter__()

    def __aiter__(self) -> "AsyncTrackedStream":
        return self

    async def __anext__(self) -> Any:
        while True:
            try:
                item = await self._bg_iter.__anext__()
            except BaseException:
                self._bg_finish()
                raise
            if self._bg_keep(item):
                return item

    async def __aenter__(self) -> "AsyncTrackedStream":
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._bg_stream.__aexit__(*exc)
        finally:
            self._bg_finish()

    async def close(self) -> None:
        try:
            await self._bg_stream.close()
        finally:
            self._bg_finish()


def _snapshot(stream: Any) -> Optional[Dict[str, Any]]:
    """Final usage from an SDK stream helper after it has been consumed."""
    for attr in ("current_message_snapshot", "current_completion_snapshot"):  # Anthropic, OpenAI chat
        try:
            snap = getattr(stream, attr)
        except Exception:  # raises before the first event arrives
            continue
        if snap is not None and _get(snap, "usage") is not None:
            return {"model": _get(snap, "model"), "usage": _get(snap, "usage")}
    # OpenAI Responses helper keeps the final response on its private state.
    final = getattr(getattr(stream, "_state", None), "_completed_response", None)
    if final is not None and _get(final, "usage") is not None:
        return {"model": _get(final, "model"), "usage": _get(final, "usage")}
    return None


class TrackedStreamManager:
    """Wraps `client.messages.stream(...)`-style helpers; records cost when the `with` block exits."""

    def __init__(self, manager: Any, on_done: OnDone, on_error: Optional[Callable[[], None]] = None):
        self._bg_manager = manager
        self._bg_on_done = on_done
        self._bg_on_error = on_error
        self._bg_stream: Any = None
        self._bg_settled = False

    def _bg_finish(self) -> None:
        if not self._bg_settled:
            self._bg_settled = True
            self._bg_on_done(_snapshot(self._bg_stream))

    def _bg_failed(self) -> None:
        if not self._bg_settled:
            self._bg_settled = True
            if self._bg_on_error is not None:
                self._bg_on_error()

    def __enter__(self) -> Any:
        try:
            self._bg_stream = self._bg_manager.__enter__()  # the request is sent here
        except BaseException:
            self._bg_failed()
            raise
        return self._bg_stream

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._bg_manager.__exit__(*exc)
        finally:
            self._bg_finish()

    async def __aenter__(self) -> Any:
        try:
            self._bg_stream = await self._bg_manager.__aenter__()
        except BaseException:
            self._bg_failed()
            raise
        return self._bg_stream

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._bg_manager.__aexit__(*exc)
        finally:
            self._bg_finish()

    def __del__(self) -> None:
        # Never entered means nothing was sent; entered but never exited still cost money.
        try:
            if self._bg_stream is None:
                self._bg_failed()
            else:
                self._bg_finish()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_bg_"):
            raise AttributeError(name)
        return getattr(self._bg_manager, name)
