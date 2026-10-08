from decimal import Decimal

import pytest

from budget_guard import ModelPrice, PricingTable, UnknownModelError


def test_default_table_loads_both_providers():
    table = PricingTable.default()
    assert table.get("claude-opus-5-5").input == Decimal("4.00")
    assert table.get("gpt-6-luna", provider="openai").output == Decimal("0.50")


def test_dated_snapshots_resolve_to_base_model():
    table = PricingTable.default()
    base = table.get("gpt-6-luna")
    assert table.get("gpt-6-luna-2026-03-01") is base
    assert table.get("claude-opus-5-5-20260401") is table.get("claude-opus-5-5")


def test_different_model_sharing_a_prefix_is_not_matched():
    table = PricingTable.default()
    with pytest.raises(UnknownModelError):
        table.get("gpt-6-astra-pro")


def test_provider_filter():
    table = PricingTable.default()
    with pytest.raises(UnknownModelError):
        table.get("claude-opus-5-5", provider="openai")


def test_set_and_update_override_prices():
    table = PricingTable.default()
    table.set("openai", "my-finetune", {"input": "1", "output": "2"})
    assert table.get("my-finetune") == ModelPrice(input=Decimal(1), output=Decimal(2))

    table.update({"anthropic": {"claude-opus-5-5": {"input": "3", "output": "15"}}})
    assert table.get("claude-opus-5-5").input == Decimal(3)
    assert "claude-sonnet-5-5" in table  # untouched models survive an update


def test_from_file(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text('{"_meta": {}, "openai": {"x-model": {"input": 0.5, "output": 1.5}}}')
    table = PricingTable.from_file(path)
    assert table.get("x-model").input == Decimal("0.5")


def test_long_context_tier():
    price = PricingTable.default().get("claude-haiku-5-5")
    assert price.for_prompt(100_000) is price
    assert price.for_prompt(100_001).input == Decimal("0.50")
