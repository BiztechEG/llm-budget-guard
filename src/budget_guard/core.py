"""BudgetGuard: prices each call and keeps running totals per user, feature and day."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union

from .context import CallContext
from .cost import calculate_cost, default_pricing
from .estimate import estimate_usage
from .limits import ACTIONS, BudgetError, BudgetExceeded, Limits, LoopDetection, LoopDetected, fingerprint
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


@dataclass
class Reservation:
    """Budget held for a call in flight, from an estimate of its cost, until it finishes."""

    model: str
    estimate: Usage
    amount: Decimal
    keys: List[str] = field(default_factory=list)


_DAY_TTL = 2 * 24 * 3600  # keep yesterday around for reporting, then let it expire


class BudgetGuard:
    """Holds pricing, storage, limits and settings shared by every client it wraps.

    limits:           a `Limits` (or a dict of its fields). None means no caps.
    loop_detection:   a `LoopDetection`; on by default (10 identical requests in 60s). None turns it off.
    on_exceeded:      "raise" (default) stops the call with BudgetExceeded / LoopDetected;
                      "warn" logs a warning and lets it through. Limits and LoopDetection
                      can override this with their own `action`.
    on_violation:     called with the BudgetError each time a limit or loop check trips,
                      in either mode, before raising or warning.
    on_unknown_model: "warn" (default) lets the call through uncounted and warns once per model;
                      "ignore" does the same silently; "raise" refuses unpriced models before sending.
    reserve_output_tokens: output tokens assumed for a call that sets no max_tokens, when estimating
                      the cost of calls still running (see `before_call`).
    """

    def __init__(
        self,
        *,
        limits: Union[Limits, Mapping[str, Any], None] = None,
        loop_detection: Optional[LoopDetection] = LoopDetection(),
        on_exceeded: str = "raise",
        on_violation: Optional[Callable[[BudgetError], None]] = None,
        pricing: Optional[PricingTable] = None,
        storage: Optional[Storage] = None,
        on_unknown_model: str = "warn",
        on_record: Optional[Callable[[CallRecord], None]] = None,
        clock: Optional[Callable[[], datetime]] = None,
        reserve_output_tokens: int = 1024,
    ):
        if on_unknown_model not in _UNKNOWN_MODEL_MODES:
            raise ValueError(f"on_unknown_model must be one of {_UNKNOWN_MODEL_MODES}")
        if on_exceeded not in ACTIONS:
            raise ValueError(f"on_exceeded must be one of {ACTIONS}")
        self.limits = Limits(**limits) if isinstance(limits, Mapping) else limits
        self.loop_detection = loop_detection
        self.on_exceeded = on_exceeded
        self._violation_listeners: List[Callable[[BudgetError], None]] = [on_violation] if on_violation else []
        self.pricing = pricing or default_pricing()
        self.storage = storage or InMemoryStorage()
        self.on_unknown_model = on_unknown_model
        self._listeners: List[Callable[[CallRecord], None]] = [on_record] if on_record else []
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._warned_models: set = set()
        self.reserve_output_tokens = reserve_output_tokens

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

    def remaining(self, *, user: Optional[str] = None, feature: Optional[str] = None) -> Optional[Decimal]:
        """USD left today under the matching limit, or None if that scope has no limit."""
        if user is not None and feature is not None:
            raise ValueError("Pass user or feature, not both.")
        if self.limits is None:
            return None
        if user is not None:
            limit = self.limits.for_user(user)
        elif feature is not None:
            limit = self.limits.for_feature(feature)
        else:
            limit = self.limits.daily_total  # type: ignore[assignment]
        if limit is None:
            return None
        return max(Decimal(0), limit - self.spent(user=user, feature=feature))

    def add_listener(self, listener: Callable[[CallRecord], None]) -> None:
        self._listeners.append(listener)

    def add_violation_listener(self, listener: Callable[[BudgetError], None]) -> None:
        self._violation_listeners.append(listener)

    def _scopes(self, ctx: CallContext) -> Iterable[Tuple[str, Optional[str], str, Optional[Decimal]]]:
        """(scope, name, storage key suffix, limit) for each total this call counts toward."""
        limits = self.limits
        yield "daily", None, "total", limits.daily_total if limits else None  # type: ignore[misc]
        if ctx.user is not None:
            yield "user", ctx.user, f"user:{ctx.user}", limits.for_user(ctx.user) if limits else None
        if ctx.feature is not None:
            yield "feature", ctx.feature, f"feature:{ctx.feature}", limits.for_feature(ctx.feature) if limits else None

    # -- call lifecycle -------------------------------------------------------

    def before_call(
        self, provider: str, kind: str, kwargs: Mapping[str, Any], ctx: CallContext, *, reserve: bool = True
    ) -> Optional[Reservation]:
        """Runs before a request is sent. Raising here stops the request.

        Calls still running count toward each limit at their estimated cost, so a burst of
        parallel calls can't all pass the check before the first one is recorded. The returned
        reservation holds that estimate; pass it to `record_response` or `release` when the call ends.
        """
        model = kwargs.get("model")
        if self.on_unknown_model == "raise" and isinstance(model, str):
            self.pricing.get(model, provider)

        # Also kept without limits: it's what a stream that ends without reporting usage is charged.
        reservation = self._estimate(provider, model, kwargs) if reserve else None
        try:
            if self.limits is not None:
                day = self.today()
                scopes = [s for s in self._scopes(ctx) if s[3] is not None]
                for scope, name, suffix, limit in scopes:
                    spent = self.storage.get(f"spend:{day}:{suffix}")
                    pending_key = f"pending:{day}:{suffix}"
                    if reservation is not None:
                        # add() is atomic, so of two parallel calls only the first sees nothing held before it.
                        held = self.storage.add(pending_key, reservation.amount, _DAY_TTL) - reservation.amount
                        reservation.keys.append(pending_key)
                    else:
                        held = self.storage.get(pending_key)
                    if spent + held >= limit:
                        self._violation(BudgetExceeded(scope, name, limit, spent, held), self.limits.action)

            loops = self.loop_detection
            if loops is not None:
                fp = fingerprint(provider, kind, kwargs, ctx)
                count = self.storage.hit(f"loop:{fp}", loops.window_seconds)
                if count > loops.max_repeats:
                    self._violation(LoopDetected(fp, count, loops.window_seconds, ctx), loops.action)
        except BaseException:
            self.release(reservation)
            raise
        return reservation

    def _estimate(self, provider: str, model: Any, kwargs: Mapping[str, Any]) -> Optional[Reservation]:
        if not isinstance(model, str):
            return None
        try:
            price = self.pricing.get(model, provider)
        except UnknownModelError:
            return None  # unpriced calls aren't counted, so there is nothing to hold
        usage = estimate_usage(kwargs, self.reserve_output_tokens)
        return Reservation(model=model, estimate=usage, amount=calculate_cost(usage, price))

    def release(self, reservation: Optional[Reservation]) -> None:
        """Stop holding a reservation's estimate. Safe to call more than once."""
        if reservation is None:
            return
        keys, reservation.keys = reservation.keys, []
        for key in keys:
            self.storage.add(key, -reservation.amount, _DAY_TTL)

    def _violation(self, error: BudgetError, action: Optional[str]) -> None:
        for listener in self._violation_listeners:
            try:
                listener(error)
            except Exception:
                logger.exception("budget_guard on_violation listener failed")
        if (action or self.on_exceeded) == "raise":
            raise error
        logger.warning("%s The call was allowed because the action is 'warn'.", error)

    def record_response(
        self,
        response: Any,
        provider: str,
        ctx: CallContext,
        fallback_model: Optional[str] = None,
        reservation: Optional[Reservation] = None,
    ) -> Optional[CallRecord]:
        """Price a finished response, add it to the totals and release its reservation. Never raises.

        With no usage to read (an OpenAI stream closed before its last chunk), the call is
        counted at its reserved estimate rather than not at all.
        """
        try:
            if response is None:
                if reservation is not None:
                    logger.warning(
                        "No usage reported for a %s call (a stream closed before its last chunk?); "
                        "counting its estimated cost instead.",
                        provider,
                    )
                    return self.record(provider, reservation.model, reservation.estimate, ctx)
                logger.warning(
                    "No usage reported for a %s call (a stream closed before its last chunk?); it was not counted.",
                    provider,
                )
                return None
            model, usage = extract_usage(response, provider)
            return self.record(provider, model or fallback_model, usage, ctx)
        except Exception:
            logger.exception("budget_guard failed to record a %s call", provider)
            return None
        finally:
            # Released after recording, so the call is never missing from both totals at once.
            try:
                self.release(reservation)
            except Exception:
                logger.exception("budget_guard failed to release a reservation")

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
        for scope, name, suffix, limit in self._scopes(ctx):
            total = self.storage.add(f"spend:{day}:{suffix}", cost, _DAY_TTL)
            if limit is not None and total >= limit > total - cost:
                who = "Daily total" if scope == "daily" else f"Daily budget for {scope} {name!r}"
                logger.warning("%s reached: $%.4f of $%.4f.", who, total, limit)
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
    """Replace the shared guard used by `guard()` with one built from these settings.

    Clients already wrapped with `guard()` pick up the new settings. Unless `storage`
    is given, the previous guard's storage is kept so today's totals aren't lost.
    """
    global _default_guard
    if "storage" not in settings and _default_guard is not None:
        settings["storage"] = _default_guard.storage
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


def remaining(*, user: Optional[str] = None, feature: Optional[str] = None) -> Optional[Decimal]:
    """Budget left today on the shared guard, or None when that scope has no limit."""
    return get_default_guard().remaining(user=user, feature=feature)
