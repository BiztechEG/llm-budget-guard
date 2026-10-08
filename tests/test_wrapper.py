import asyncio
import warnings
from datetime import timedelta
from decimal import Decimal

import httpx2
import pytest

import budget_guard
from budget_guard import BudgetGuard, UnknownModelError, budget_context, guard

from conftest import (
    anthropic_events,
    anthropic_message,
    chat_chunks,
    chat_completion,
    responses_events,
    responses_object,
    sse,
    stream_response,
)

M = 1_000_000
MSG = [{"role": "user", "content": "hi"}]


def reply_json(payload):
    return lambda body: httpx2.Response(200, json=payload)


# -- non-streaming ------------------------------------------------------------

def test_openai_chat_create_records_spend_per_scope(server, openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, M))
    client = guard(openai_client, user="u1", feature="chat", budget=budget)

    result = client.chat.completions.create(model="gpt-6-luna", messages=MSG)

    assert result.choices[0].message.content == "hi"  # caller gets the real SDK object
    assert budget.spent(user="u1") == Decimal("0.60")
    assert budget.spent(feature="chat") == Decimal("0.60")
    assert budget.spent() == Decimal("0.60")
    assert budget.spent(user="someone-else") == 0


def test_anthropic_create(server, anthropic_client, budget):
    server.responder = reply_json(anthropic_message("claude-opus-5-5", M, 100_000, cache_read=M))
    client = guard(anthropic_client, user="u1", budget=budget)

    client.messages.create(model="claude-opus-5-5", max_tokens=10, messages=MSG)

    assert budget.spent(user="u1") == Decimal("6.20")


def test_openai_responses_create(server, openai_client, budget):
    server.responder = reply_json(responses_object("gpt-6.1-sol", M, 100_000))
    client = guard(openai_client, feature="search", budget=budget)

    result = client.responses.create(model="gpt-6.1-sol", input="hi")

    assert result.output_text == "Hello"
    assert budget.spent(feature="search") == Decimal("3")


def test_dated_model_in_response_is_priced(server, openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-6-luna-2026-09-01", M, 0))
    guard(openai_client, budget=budget).chat.completions.create(model="gpt-6-luna", messages=MSG)
    assert budget.spent() == Decimal("0.10")


# -- context --------------------------------------------------------------------

def test_budget_context_overrides_defaults_and_nests(server, openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, 0))
    client = guard(openai_client, user="default", feature="chat", budget=budget)

    with budget_context(user="alice"):
        client.chat.completions.create(model="gpt-6-luna", messages=MSG)
        with budget_context(feature="summary"):
            client.chat.completions.create(model="gpt-6-luna", messages=MSG)
    client.chat.completions.create(model="gpt-6-luna", messages=MSG)

    assert budget.spent(user="alice") == Decimal("0.20")
    assert budget.spent(user="default") == Decimal("0.10")
    assert budget.spent(feature="chat") == Decimal("0.20")
    assert budget.spent(feature="summary") == Decimal("0.10")


def test_with_context_returns_rescoped_copy(server, openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, 0))
    client = guard(openai_client, user="u1", feature="chat", budget=budget)

    client.with_context(user="u2").chat.completions.create(model="gpt-6-luna", messages=MSG)

    assert budget.spent(user="u2") == Decimal("0.10")
    assert budget.spent(user="u1") == 0
    assert budget.spent(feature="chat") == Decimal("0.10")


def test_spend_is_bucketed_by_utc_day(server, openai_client, budget, clock):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, 0))
    client = guard(openai_client, user="u1", budget=budget)

    client.chat.completions.create(model="gpt-6-luna", messages=MSG)
    clock.now += timedelta(days=1)
    client.chat.completions.create(model="gpt-6-luna", messages=MSG)

    assert budget.spent(user="u1") == Decimal("0.10")
    assert budget.spent(user="u1", day="2026-10-08") == Decimal("0.10")


# -- streaming: create(stream=True) ---------------------------------------------

def test_openai_chat_stream_asks_for_usage_and_hides_the_extra_chunk(server, openai_client, budget):
    server.responder = lambda body: stream_response(sse(chat_chunks("gpt-6-luna", M, M, with_usage=True), done=True))
    client = guard(openai_client, user="u1", budget=budget)

    stream = client.chat.completions.create(model="gpt-6-luna", messages=MSG, stream=True)
    text = "".join(chunk.choices[0].delta.content or "" for chunk in stream)  # would IndexError on the usage chunk

    assert text == "Hello"
    assert server.requests[0]["stream_options"] == {"include_usage": True}
    assert budget.spent(user="u1") == Decimal("0.60")


def test_openai_chat_stream_keeps_usage_chunk_when_caller_asked(server, openai_client, budget):
    server.responder = lambda body: stream_response(sse(chat_chunks("gpt-6-luna", M, M, with_usage=True), done=True))
    client = guard(openai_client, budget=budget)

    chunks = list(
        client.chat.completions.create(
            model="gpt-6-luna", messages=MSG, stream=True, stream_options={"include_usage": True}
        )
    )

    assert chunks[-1].usage.prompt_tokens == M
    assert budget.spent() == Decimal("0.60")


def test_stream_closed_early_does_not_raise(server, openai_client, budget):
    # Nothing billed is seen before close, so nothing is counted, but closing must not raise.
    server.responder = lambda body: stream_response(sse(chat_chunks("gpt-6-luna", M, M, with_usage=True), done=True))
    client = guard(openai_client, budget=budget)
    with client.chat.completions.create(model="gpt-6-luna", messages=MSG, stream=True) as stream:
        next(iter(stream))
    assert budget.spent() == 0


def test_openai_responses_stream(server, openai_client, budget):
    server.responder = lambda body: stream_response(sse(responses_events("gpt-6.1-sol", M, 100_000), named=True))
    client = guard(openai_client, budget=budget)

    events = [e.type for e in client.responses.create(model="gpt-6.1-sol", input="hi", stream=True)]

    assert events == ["response.created", "response.completed"]
    assert budget.spent() == Decimal("3")


def test_anthropic_stream_combines_start_and_delta_usage(server, anthropic_client, budget):
    server.responder = lambda body: stream_response(
        sse(anthropic_events("claude-opus-5-5", M, 100_000, cache_read=M), named=True)
    )
    client = guard(anthropic_client, user="u1", budget=budget)

    for _ in client.messages.create(model="claude-opus-5-5", max_tokens=10, messages=MSG, stream=True):
        pass

    assert budget.spent(user="u1") == Decimal("6.20")


# -- streaming: .stream() helpers -------------------------------------------------

def test_anthropic_messages_stream_helper(server, anthropic_client, budget):
    server.responder = lambda body: stream_response(sse(anthropic_events("claude-opus-5-5", M, 100_000), named=True))
    client = guard(anthropic_client, user="u1", budget=budget)

    with client.messages.stream(model="claude-opus-5-5", max_tokens=10, messages=MSG) as stream:
        assert stream.get_final_text() == "Hello"

    assert budget.spent(user="u1") == Decimal("6")


def test_openai_chat_stream_helper(server, openai_client, budget):
    server.responder = lambda body: stream_response(sse(chat_chunks("gpt-6-luna", M, M, with_usage=True), done=True))
    client = guard(openai_client, budget=budget)

    with client.chat.completions.stream(model="gpt-6-luna", messages=MSG) as stream:
        final = stream.get_final_completion()

    assert final.choices[0].message.content == "Hello"
    assert server.requests[0]["stream_options"] == {"include_usage": True}
    assert budget.spent() == Decimal("0.60")


def test_openai_responses_stream_helper(server, openai_client, budget):
    server.responder = lambda body: stream_response(sse(responses_events("gpt-6.1-sol", M, 100_000), named=True))
    client = guard(openai_client, budget=budget)

    with client.responses.stream(model="gpt-6.1-sol", input="hi") as stream:
        stream.get_final_response()

    assert budget.spent() == Decimal("3")


# -- async ----------------------------------------------------------------------

def test_async_openai_create(server, async_openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, M))
    client = guard(async_openai_client, user="u1", budget=budget)

    async def run():
        return await client.chat.completions.create(model="gpt-6-luna", messages=MSG)

    assert asyncio.run(run()).choices[0].message.content == "hi"
    assert budget.spent(user="u1") == Decimal("0.60")


def test_async_anthropic_stream_and_helper(server, async_anthropic_client, budget):
    server.responder = lambda body: stream_response(sse(anthropic_events("claude-opus-5-5", M, 100_000), named=True))
    client = guard(async_anthropic_client, budget=budget)

    async def run():
        stream = await client.messages.create(model="claude-opus-5-5", max_tokens=10, messages=MSG, stream=True)
        async for _ in stream:
            pass
        async with client.messages.stream(model="claude-opus-5-5", max_tokens=10, messages=MSG) as s:
            await s.get_final_message()

    asyncio.run(run())
    assert budget.spent() == Decimal("12")


def test_async_context_is_per_task(server, async_openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, 0))
    client = guard(async_openai_client, budget=budget)

    async def call_as(user):
        with budget_context(user=user):
            await asyncio.sleep(0)
            await client.chat.completions.create(model="gpt-6-luna", messages=MSG)

    async def run():
        await asyncio.gather(call_as("a"), call_as("b"), call_as("a"))

    asyncio.run(run())
    assert budget.spent(user="a") == Decimal("0.20")
    assert budget.spent(user="b") == Decimal("0.10")


# -- unknown models, failures, plumbing --------------------------------------------

def test_unknown_model_warns_once_and_is_not_counted(server, openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-mystery", M, M))
    client = guard(openai_client, budget=budget)

    with pytest.warns(RuntimeWarning, match="gpt-mystery"):
        client.chat.completions.create(model="gpt-mystery", messages=MSG)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        client.chat.completions.create(model="gpt-mystery", messages=MSG)

    assert budget.spent() == 0


def test_unknown_model_raise_mode_blocks_before_sending(server, openai_client, clock):
    strict = BudgetGuard(on_unknown_model="raise", clock=clock)
    client = guard(openai_client, budget=strict)

    with pytest.raises(UnknownModelError):
        client.chat.completions.create(model="gpt-mystery", messages=MSG)
    assert server.requests == []


def test_listener_errors_never_break_the_call(server, openai_client, budget):
    seen = []
    budget.add_listener(lambda record: (_ for _ in ()).throw(RuntimeError("boom")))
    budget.add_listener(seen.append)
    server.responder = reply_json(chat_completion("gpt-6-luna", M, 0))

    guard(openai_client, user="u1", budget=budget).chat.completions.create(model="gpt-6-luna", messages=MSG)

    assert len(seen) == 1
    assert seen[0].model == "gpt-6-luna" and seen[0].context.user == "u1"
    assert seen[0].cost == Decimal("0.10")


def test_with_options_stays_guarded(server, openai_client, budget):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, 0))
    client = guard(openai_client, budget=budget)

    client.with_options(timeout=5).chat.completions.create(model="gpt-6-luna", messages=MSG)

    assert budget.spent() == Decimal("0.10")


def test_untracked_attributes_pass_through(openai_client, budget):
    client = guard(openai_client, budget=budget)
    assert client.api_key == "test"
    assert client.models is openai_client.models
    assert client.unwrapped is openai_client


def test_shared_guard_follows_configure(server, openai_client, clock):
    server.responder = reply_json(chat_completion("gpt-6-luna", M, 0))
    client = guard(openai_client, user="u1")  # created before configure()
    try:
        budget_guard.configure(clock=clock)
        client.chat.completions.create(model="gpt-6-luna", messages=MSG)
        assert budget_guard.spent(user="u1") == Decimal("0.10")
    finally:
        budget_guard.configure()


def test_unrecognised_client_needs_explicit_provider(budget):
    with pytest.raises(TypeError):
        guard(object(), budget=budget)
