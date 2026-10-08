"""BudgetGuard: prices each call and keeps running totals per user, feature and day."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, List, Mapping, Optional

from .context import CallContext
from .cost import calculate_cost, default_pricing
from .pricing import PricingTable, UnknownModelError
from .storage import InMemoryStorage, Storage
from .usage import Usage, extract_usage

logger = logging.getLogger("budget_guard")

_UNKNOWN_MODEL_MODES = ("warn", "ignore", "raise")


@dataclass(frozen=True)
class CallRecord:
    provider: str
    model: str
    context: CallContext
    usage: Usage
    cost: Decimal
    day: str


class BudgetGuard:
    """Holds pricing, storage and settings shared by every client it wraps.

    on_unknown_model:
        "warn"   (default) let the call through, record nothing, warn once per model.
        "ignore" same, without the warning.
        "raise"  refuse calls to unpriced models before they are sent.
    """

    def __init__(
        self,
        *,
        pricing: Optional[PricingTable] = None,
        storage: Optional[Storage] = None,
        on_unknown_model: str = "warn",
        on_record: Optional[Callable[[CallRecord], None]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ):
        if on_unknown_model not in _UNKNOWN_MODEL_MODES:
            raise ValueError(f"on_unknown_model must be one of {_UNKNOWN_MODEL_MODES}")
        self.pricing = pricing or default_pricing()
        self.storage = storage or InMemoryStorage()
        self.on_unknown_model = on_unknown_model
        self._listeners: List[Callable[[CallRecord], None]] = [on_record] if on_record else []
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._warned_models: set = set()

    # -- keys ---------------------------------------------------------------

    def today(self) -> str:
        return self._clock().astimezone(timezone.utc).date().isoformat()

    @staticmethod
    def _key(day: str, scope: str, name: Optional[str] = None) -> str:
        return f"spend:{day}:{scope}" if name is None else f"spend:{day}:{scope}:{name}"

    def spent(self, *, user: Optional[str] = None, feature: Optional[str] = None, day: Optional[str] = None) -> Decimal:
        """USD spent today (or on `day`, YYYY-MM-DD, UTC). No arguments means the daily total."""
        if user is not None and feature is not None:
            raise ValueError("Pass user or feature, not both.")
        day = day or self.today()
        if user is not None:
            return self.storage.get(self._key(day, "user", user))
        if feature is not None:
            return self.storage.get(self._key(day, "feature", feature))
        return self.storage.get(self._key(day, "total"))

    def add_listener(self, listener: Callable[[CallRecord], None]) -> None:
        self._listeners.append(listener)

    # -- call lifecycle -------------------------------------------------------

    def before_call(self, provider: str, kind: str, kwargs: Mapping[str, Any], ctx: CallContext) -> None:
        """Runs before a request is sent. Raising here stops the request."""
        model = kwargs.get("model")
        if self.on_unknown_model == "raise" and isinstance(model, str):
            self.pricing.get(model, provider)

    def record_response(
        self,
        response: Any,
        provider: str,
        ctx: CallContext,
        fallback_model: Optional[str] = None,
    ) -> Optional[CallRecord]:
        """Price a finished response and add it to the totals. Never raises."""
        try:
            if response is None:
                logger.warning("No usage reported for a %s call; it was not counted.", provider)
                return None
            model, usage = extract_usage(response, provider)
            return self.record(provider, model or fallback_model, usage, ctx)
        except Exception:
            logger.exception("budget_guard failed to record a %s call", provider)
            return None

    def record(self, provider: str, model: Optional[str], usage: Usage, ctx: CallContext) -> Optional[CallRecord]:
        if not model:
            logger.warning("A %s response had no model name; it was not counted.", provider)
            return None
        try:
            price = self.pricing.get(model, provider)
        except UnknownModelError:
            self._unknown_model(model)
            return None
        cost = calculate_cost(usage, price)
        day = self.today()
        ttl = 2 * 24 * 3600  # keep yesterday around for reporting, then let it expire
        self.storage.add(self._key(day, "total"), cost, ttl)
        if ctx.user is not None:
            self.storage.add(self._key(day, "user", ctx.user), cost, ttl)
        if ctx.feature is not None:
            self.storage.add(self._key(day, "feature", ctx.feature), cost, ttl)
        record = CallRecord(provider=provider, model=model, context=ctx, usage=usage, cost=cost, day=day)
        for listener in self._listeners:
            try:
                listener(record)
            except Exception:
                logger.exception("budget_guard on_record listener failed")
        return record

    def _unknown_model(self, model: str) -> None:
        if self.on_unknown_model == "ignore" or model in self._warned_models:
            return
        self._warned_models.add(model)
        message = (
            f"budget_guard has no price for model {model!r}; its calls are not being counted. "
            "Add it with default_pricing().set(provider, model, {...})."
        )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    # -- wrapping -----------------------------------------------------------

    def wrap(self, client: Any, *, user: Optional[str] = None, feature: Optional[str] = None, provider: Optional[str] = None) -> Any:
        return _wrap(client, self, user, feature, provider)


def _wrap(client: Any, guard_obj: Optional[BudgetGuard], user: Optional[str], feature: Optional[str], provider: Optional[str]) -> Any:
    from .proxy import GuardedClient, detect_provider

    return GuardedClient(client, guard_obj, provider or detect_provider(client), CallContext(user, feature))


_default_guard: Optional[BudgetGuard] = None


def get_default_guard() -> BudgetGuard:
    global _default_guard
    if _default_guard is None:
        _default_guard = BudgetGuard()
    return _default_guard


def configure(**settings: Any) -> BudgetGuard:
    """Replace the shared guard used by `guard()` with one built from these settings."""
    global _default_guard
    _default_guard = BudgetGuard(**settings)
    return _default_guard


def guard(
    client: Any,
    *,
    user: Optional[str] = None,
    feature: Optional[str] = None,
    budget: Optional[BudgetGuard] = None,
    provider: Optional[str] = None,
) -> Any:
    """Wrap an OpenAI or Anthropic client (sync or async) so every call is priced and counted.

        client = guard(OpenAI(), user="u1", feature="chat")
    """
    # Without `budget`, the client follows the shared guard, including later configure() calls.
    return _wrap(client, budget, user, feature, provider)


def spent(*, user: Optional[str] = None, feature: Optional[str] = None, day: Optional[str] = None) -> Decimal:
    """Spend so far on the shared guard."""
    return get_default_guard().spent(user=user, feature=feature, day=day)
