from decimal import Decimal
from types import SimpleNamespace

import pytest

from budget_guard import (
    ModelPrice,
    PricingTable,
    Usage,
    UnknownModelError,
    calculate_cost,
    cost_of_response,
    extract_usage,
)


def test_openai_chat_completions_usage_splits_out_cached_tokens():
    response = {
        "model": "gpt-6.1-sol-2026-06-01",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    }
    model, usage = extract_usage(response, "openai")
    assert model == "gpt-6.1-sol-2026-06-01"
    assert usage == Usage(input_tokens=600, output_tokens=200, cache_read_tokens=400)


def test_openai_responses_api_usage_from_sdk_like_object():
    response = SimpleNamespace(
        model="gpt-6-luna",
        usage=SimpleNamespace(
            input_tokens=500,
            output_tokens=50,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    _, usage = extract_usage(response, "openai")
    assert usage == Usage(input_tokens=500, output_tokens=50)


def test_anthropic_usage_with_cache_fields_missing_or_none():
    response = SimpleNamespace(
        model="claude-sonnet-5-5",
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5, cache_read_input_tokens=None
        ),
    )
    _, usage = extract_usage(response, "anthropic")
    assert usage == Usage(input_tokens=10, output_tokens=5)


def test_openai_cost():
    # 600 uncached * $2 + 400 cached * $0.10 + 200 output * $10, per 1M
    response = {
        "model": "gpt-6.1-sol",
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 400},
        },
    }
    cost = cost_of_response(response, "openai")
    assert cost.amount == Decimal("0.00324")
    assert cost.model == "gpt-6.1-sol"


def test_anthropic_cost_with_cache_read_and_write():
    response = {
        "model": "claude-opus-5-5",
        "usage": {
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "cache_read_input_tokens": 2_000_000,
            "cache_creation_input_tokens": 1_000_000,
        },
    }
    # $4 + $2 output + $0.40 cache read + $5 cache write
    assert cost_of_response(response, "anthropic").amount == Decimal("11.4")


def test_long_context_rates_apply_to_whole_request():
    price = PricingTable.default().get("claude-haiku-5-5")
    usage = Usage(input_tokens=200_000, output_tokens=1_000_000)
    # $0.50 * 0.2 + $2.50 * 1
    assert calculate_cost(usage, price) == Decimal("2.6")


def test_missing_cache_rates_fall_back_to_input_rate():
    price = ModelPrice(input=Decimal(1), output=Decimal(2))
    usage = Usage(cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert calculate_cost(usage, price) == Decimal(2)


def test_unknown_model_raises():
    response = {"model": "gpt-unknown", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with pytest.raises(UnknownModelError):
        cost_of_response(response, "openai")


def test_custom_table_and_model_override():
    table = PricingTable({"openai": {"proxy-model": {"input": "1", "output": "1"}}})
    response = {"model": None, "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    assert cost_of_response(response, "openai", pricing=table, model="proxy-model").amount == 1


def test_unsupported_provider():
    with pytest.raises(ValueError):
        extract_usage({}, "gemini")
