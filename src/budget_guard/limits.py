"""Spend limits, loop detection, and what happens when either trips."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Union

from .context import CallContext

Amount = Union[int, float, str, Decimal]

ACTIONS = ("raise", "warn")


def to_decimal(value: Amount) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class BudgetError(Exception):
    """Base class for everything budget_guard raises to stop a call."""


class BudgetExceeded(BudgetError):
    def __init__(self, scope: str, name: Optional[str], limit: Decimal, spent: Decimal):
        self.scope = scope  # "user", "feature" or "daily"
        self.name = name
        self.limit = limit
        self.spent = spent
        who = "Daily total" if scope == "daily" else f"Daily budget for {scope} {name!r}"
        super().__init__(f"{who} exceeded: spent ${spent:.4f} of ${limit:.4f}.")


class LoopDetected(BudgetError):
    def __init__(self, fingerprint: str, count: int, window_seconds: float, context: CallContext):
        self.fingerprint = fingerprint
        self.count = count
        self.window_seconds = window_seconds
        self.context = context
        super().__init__(
            f"The same request was sent {count} times in {window_seconds:g}s "
            f"(user={context.user!r}, feature={context.feature!r}); this looks like a loop."
        )


def _decimal_map(values: Mapping[str, Amount]) -> Dict[str, Decimal]:
    return {k: to_decimal(v) for k, v in values.items()}


@dataclass
class Limits:
    """Daily (UTC) spend caps in USD. Anything left as None is unlimited.

        Limits(per_user=1, users={"vip": 20}, features={"chat": 5}, daily_total=50)

    A call is refused once a total has reached its cap. The call that crosses
    the cap is allowed, because its cost is only known after it returns.
    """

    per_user: Optional[Amount] = None  # default cap for every user
    users: Mapping[str, Amount] = field(default_factory=dict)  # per-user overrides
    per_feature: Optional[Amount] = None
    features: Mapping[str, Amount] = field(default_factory=dict)
    daily_total: Optional[Amount] = None
    action: Optional[str] = None  # "raise" or "warn"; None follows the guard's on_exceeded

    def __post_init__(self) -> None:
        self.per_user = to_decimal(self.per_user) if self.per_user is not None else None
        self.per_feature = to_decimal(self.per_feature) if self.per_feature is not None else None
        self.daily_total = to_decimal(self.daily_total) if self.daily_total is not None else None
        self.users = _decimal_map(self.users)
        self.features = _decimal_map(self.features)
        if self.action is not None and self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")

    def for_user(self, user: Optional[str]) -> Optional[Decimal]:
        if user is None:
            return None
        return self.users.get(user, self.per_user)  # type: ignore[return-value]

    def for_feature(self, feature: Optional[str]) -> Optional[Decimal]:
        if feature is None:
            return None
        return self.features.get(feature, self.per_feature)  # type: ignore[return-value]


@dataclass
class LoopDetection:
    """Flag the same request repeated more than `max_repeats` times within `window_seconds`.

    "Same" means same provider, method, model and request body, for the same
    user and feature, so two users asking the same question never collide.
    """

    max_repeats: int = 10
    window_seconds: float = 60.0
    action: Optional[str] = None

    def __post_init__(self) -> None:
        if self.max_repeats < 1 or self.window_seconds <= 0:
            raise ValueError("max_repeats must be >= 1 and window_seconds > 0")
        if self.action is not None and self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")


# Request options that don't change what is being asked.
_IGNORED_KEYS = frozenset({"stream", "stream_options", "timeout", "extra_headers", "extra_query", "metadata"})


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)  # pydantic objects from the SDKs
    return dump() if callable(dump) else repr(value)


def fingerprint(provider: str, kind: str, kwargs: Mapping[str, Any], ctx: CallContext) -> str:
    body = {k: v for k, v in kwargs.items() if k not in _IGNORED_KEYS}
    payload = json.dumps(
        [provider, kind.replace("_helper", ""), ctx.user, ctx.feature, body],
        sort_keys=True,
        default=_jsonable,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
