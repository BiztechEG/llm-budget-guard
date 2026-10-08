"""Turn token usage into dollars."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from .pricing import PER_MILLION, ModelPrice, PricingTable
from .usage import Usage, extract_usage


@dataclass(frozen=True)
class Cost:
    model: str
    provider: str
    usage: Usage
    amount: Decimal  # USD


def calculate_cost(usage: Usage, price: ModelPrice) -> Decimal:
    rates = price.for_prompt(usage.prompt_tokens)
    cache_read = rates.cache_read if rates.cache_read is not None else rates.input
    cache_write = rates.cache_write if rates.cache_write is not None else rates.input
    total = (
        usage.input_tokens * rates.input
        + usage.output_tokens * rates.output
        + usage.cache_read_tokens * cache_read
        + usage.cache_write_tokens * cache_write
    )
    return total / PER_MILLION


_default_table: Optional[PricingTable] = None


def default_pricing() -> PricingTable:
    """The bundled table, loaded once and shared, so runtime patches apply everywhere."""
    global _default_table
    if _default_table is None:
        _default_table = PricingTable.default()
    return _default_table


def cost_of_response(
    response: Any,
    provider: str,
    pricing: Optional[PricingTable] = None,
    model: Optional[str] = None,
) -> Cost:
    """Price a single SDK response. `model` overrides the one the response reports."""
    reported_model, usage = extract_usage(response, provider)
    model = model or reported_model
    if not model:
        raise ValueError("Response has no model field; pass model= explicitly.")
    price = (pricing or default_pricing()).get(model, provider)
    return Cost(model=model, provider=provider, usage=usage, amount=calculate_cost(usage, price))
