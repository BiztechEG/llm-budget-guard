import asyncio
import logging
from decimal import Decimal

import httpx2
import pytest

from budget_guard import (
    BudgetExceeded,
    BudgetGuard,
    InMemoryStorage,
    Limits,
    LoopDetected,
    LoopDetection,
    budget_context,
    guard,
)
from budget_guard.limits import fingerprint
from budget_guard.context import CallContext

from conftest import chat_completion

M = 1_000_000
MSG = [{"role": "user", "content": "hi"}]


@pytest.fixture
def luna(server):
    """Every call costs $0.10 (1M input tokens of gpt-6-luna)."""
    server.responder = lambda body: httpx2.Response(200, json=chat_completion("gpt-6-luna", M, 0))


def make_guard(clock, **kwargs):
    kwargs.setdefault("loop_detection", None)
    return BudgetGuard(storage=InMemoryStorage(), clock=clock, **kwargs)


def ask(client, text="hi"):
    return client.chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": text}])


# -- spend limits -------------------------------------------------------------------

def test_user_limit_blocks_once_reached(server, openai_client, clock, luna):
    budget = make_guard(clock, limits=Limits(per_user="0.25"))
    client = guard(openai_client, user="u1", budget=budget)

    for i in range(3):  # 0.10, 0.20, 0.30: the third call crosses the cap and is allowed
        ask(client, str(i))
    with pytest.raises(BudgetExceeded) as err:
        ask(client, "again")

    assert err.value.scope == "user" and err.value.name == "u1"
    assert err.value.spent == Decimal("0.30") and err.value.limit == Decimal("0.25")
    assert len(server.requests) == 3  # the blocked call never reached the provider
    ask(client.with_context(user="u2"))  # other users are unaffected


def test_user_override_and_users_without_override(server, openai_client, clock, luna):
    budget = make_guard(clock, limits=Limits(per_user="0.10", users={"vip": "1.00"}))
    client = guard(openai_client, budget=budget)

    with budget_context(user="vip"):
        for i in range(5):
            ask(client, str(i))
    with budget_context(user="regular"):
        ask(client)
        with pytest.raises(BudgetExceeded):
            ask(client, "more")


def test_feature_limit_with_default(server, openai_client, clock, luna):
    budget = make_guard(clock, limits={"per_feature": "0.10", "features": {"chat": "0.30"}})
    client = guard(openai_client, budget=budget)

    summary = client.with_context(feature="summary")
    ask(summary)
    with pytest.raises(BudgetExceeded) as err:
        ask(summary, "2")
    assert err.value.scope == "feature"

    chat = client.with_context(feature="chat")
    for i in range(3):
        ask(chat, str(i))
    with pytest.raises(BudgetExceeded):
        ask(chat, "4")


def test_daily_total_applies_to_everyone(server, openai_client, clock, luna):
    budget = make_guard(clock, limits=Limits(daily_total="0.20"))
    client = guard(openai_client, budget=budget)

    ask(client.with_context(user="a"))
    ask(client.with_context(user="b"))
    with pytest.raises(BudgetExceeded) as err:
        ask(client.with_context(user="c"))
    assert err.value.scope == "daily"


def test_limits_reset_on_a_new_utc_day(server, openai_client, clock, luna):
    from datetime import timedelta

    budget = make_guard(clock, limits=Limits(per_user="0.10"))
    client = guard(openai_client, user="u1", budget=budget)
    ask(client)
    with pytest.raises(BudgetExceeded):
        ask(client, "2")
    clock.now += timedelta(days=1)
    ask(client, "3")


def test_zero_limit_blocks_a_user_entirely(server, openai_client, clock, luna):
    budget = make_guard(clock, limits=Limits(users={"banned": 0}))
    with pytest.raises(BudgetExceeded):
        ask(guard(openai_client, user="banned", budget=budget))
    assert server.requests == []


def test_warn_mode_logs_and_lets_calls_through(server, openai_client, clock, luna, caplog):
    seen = []
    budget = make_guard(clock, limits=Limits(per_user="0.10"), on_exceeded="warn", on_violation=seen.append)
    client = guard(openai_client, user="u1", budget=budget)

    with caplog.at_level(logging.WARNING, logger="budget_guard"):
        ask(client)
        ask(client, "2")

    assert len(server.requests) == 2
    assert budget.spent(user="u1") == Decimal("0.20")
    assert [type(e) for e in seen] == [BudgetExceeded]
    assert any("reached" in r.message for r in caplog.records)  # logged when the cap was crossed
    assert any("allowed" in r.message for r in caplog.records)  # logged when the next call went through


def test_per_rule_action_overrides_the_guard_default(server, openai_client, clock, luna):
    budget = make_guard(clock, limits=Limits(per_user="0.10", action="warn"), on_exceeded="raise")
    client = guard(openai_client, user="u1", budget=budget)
    ask(client)
    ask(client, "2")  # warns instead of raising
    assert len(server.requests) == 2


def test_remaining(server, openai_client, clock, luna):
    budget = make_guard(clock, limits=Limits(per_user="0.25", daily_total="1"))
    client = guard(openai_client, user="u1", budget=budget)
    assert budget.remaining(user="u1") == Decimal("0.25")
    ask(client)
    assert budget.remaining(user="u1") == Decimal("0.15")
    assert budget.remaining() == Decimal("0.90")
    assert budget.remaining(feature="anything") is None
    for i in range(3):
        try:
            ask(client, str(i))
        except BudgetExceeded:
            pass
    assert budget.remaining(user="u1") == 0


def test_limits_apply_to_async_clients(server, async_openai_client, clock, luna):
    budget = make_guard(clock, limits=Limits(per_user="0.10"))
    client = guard(async_openai_client, user="u1", budget=budget)

    async def run():
        await client.chat.completions.create(model="gpt-6-luna", messages=MSG)
        await client.chat.completions.create(model="gpt-6-luna", messages=MSG)

    with pytest.raises(BudgetExceeded):
        asyncio.run(run())


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        BudgetGuard(on_exceeded="explode")
    with pytest.raises(ValueError):
        Limits(action="explode")
    with pytest.raises(ValueError):
        LoopDetection(max_repeats=0)


# -- loop detection -----------------------------------------------------------------

def test_identical_requests_beyond_threshold_are_a_loop(server, openai_client, clock, luna):
    budget = make_guard(clock, loop_detection=LoopDetection(max_repeats=3, window_seconds=60))
    client = guard(openai_client, user="u1", budget=budget)

    for _ in range(3):
        ask(client)
    with pytest.raises(LoopDetected) as err:
        ask(client)

    assert err.value.count == 4
    assert len(server.requests) == 3
    ask(client, "a different question")  # not the same request


def test_loop_counts_are_per_user(server, openai_client, clock, luna):
    budget = make_guard(clock, loop_detection=LoopDetection(max_repeats=2))
    client = guard(openai_client, budget=budget)
    for user in ("a", "b", "c"):
        for _ in range(2):
            ask(client.with_context(user=user))
    assert len(server.requests) == 6


def test_loop_window_slides(server, openai_client, luna):
    now = [0.0]
    storage = InMemoryStorage(clock=lambda: now[0])
    budget = BudgetGuard(storage=storage, loop_detection=LoopDetection(max_repeats=2, window_seconds=10))
    client = guard(openai_client, budget=budget)
    ask(client)
    ask(client)
    now[0] = 11.0
    ask(client)  # the first two have aged out


def test_loop_detection_is_on_by_default(server, openai_client, luna):
    budget = BudgetGuard(storage=InMemoryStorage())
    client = guard(openai_client, budget=budget)
    for _ in range(10):
        ask(client)
    with pytest.raises(LoopDetected):
        ask(client)


def test_loop_detection_warn_mode(server, openai_client, clock, luna):
    budget = make_guard(clock, loop_detection=LoopDetection(max_repeats=1, action="warn"))
    client = guard(openai_client, budget=budget)
    ask(client)
    ask(client)
    assert len(server.requests) == 2


def test_fingerprint_ignores_transport_options():
    ctx = CallContext("u1", "chat")
    base = {"model": "m", "messages": MSG}
    a = fingerprint("openai", "openai_chat", base, ctx)
    assert a == fingerprint("openai", "openai_chat", {**base, "stream": True, "timeout": 5}, ctx)
    assert a == fingerprint("openai", "openai_chat_helper", base, ctx)
    assert a != fingerprint("openai", "openai_chat", {**base, "temperature": 0.5}, ctx)
    assert a != fingerprint("openai", "openai_chat", base, CallContext("u2", "chat"))


def test_numeric_ids_match_text_ids(server, openai_client, clock, luna):
    # request.user.id is often an int; overrides keyed by "42" (or 42) must still apply.
    budget = make_guard(clock, limits=Limits(per_user="0.10", users={"42": "1.00"}, features={7: "0.05"}))
    ask(guard(openai_client, user=42, budget=budget))
    ask(guard(openai_client, user=42, budget=budget), "2")  # under the 1.00 override, not the 0.10 default
    assert budget.spent(user="42") == budget.spent(user=42) == Decimal("0.20")
    assert budget.remaining(user=42) == Decimal("0.80")

    ask(guard(openai_client, feature=7, budget=budget))
    with pytest.raises(BudgetExceeded):
        ask(guard(openai_client, feature="7", budget=budget), "again")


@pytest.mark.parametrize("bad", [float("nan"), "NaN", -1])
def test_nonsense_limits_are_rejected(bad):
    with pytest.raises(ValueError):
        Limits(per_user=bad)
    with pytest.raises(ValueError):
        Limits(users={"u": bad})


def test_generator_arguments_are_fingerprinted_by_content(server, openai_client, clock, luna):
    budget = make_guard(clock, loop_detection=LoopDetection(max_repeats=2))
    client = guard(openai_client, budget=budget)
    generators = [(m for m in MSG) for _ in range(3)]  # all alive at once, so their reprs differ
    for messages in generators[:2]:
        client.chat.completions.create(model="gpt-6-luna", messages=messages)
    with pytest.raises(LoopDetected):
        client.chat.completions.create(model="gpt-6-luna", messages=generators[2])
    assert server.requests[0]["messages"] == MSG  # the SDK still received the messages
