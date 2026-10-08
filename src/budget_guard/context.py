"""Who a call is being made for: the user and the feature."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Iterator, Optional


@dataclass(frozen=True)
class CallContext:
    user: Optional[str] = None
    feature: Optional[str] = None

    def merge(self, other: "CallContext") -> "CallContext":
        """Fields set on `other` win; unset ones keep this context's value."""
        return replace(
            self,
            user=other.user if other.user is not None else self.user,
            feature=other.feature if other.feature is not None else self.feature,
        )


_current: ContextVar[CallContext] = ContextVar("budget_guard_context", default=CallContext())


def current_context() -> CallContext:
    return _current.get()


@contextmanager
def budget_context(user: Optional[str] = None, feature: Optional[str] = None) -> Iterator[CallContext]:
    """Attribute every guarded call inside the block to this user and/or feature.

    Works across threads' own contexts and asyncio tasks, so it suits
    per-request middleware. Overrides the defaults given to `guard()`.
    """
    ctx = _current.get().merge(CallContext(user=user, feature=feature))
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)
