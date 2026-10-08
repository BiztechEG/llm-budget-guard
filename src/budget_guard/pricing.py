"""Model price table: bundled defaults that can be replaced or patched at runtime."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

PER_MILLION = Decimal(1_000_000)

# A dated snapshot of a known model, e.g. "-2026-01-15", "-20250514" or "@20251101".
_SNAPSHOT_SUFFIX = re.compile(r"^[-@]\d[\d-]*$")


class UnknownModelError(KeyError):
    """Raised when a model has no entry in the pricing table."""

    def __init__(self, model: str, provider: Optional[str] = None):
        self.model = model
        self.provider = provider
        where = f" for provider {provider!r}" if provider else ""
        super().__init__(f"No price for model {model!r}{where}. Add it with PricingTable.set().")


def _dec(value: Any) -> Decimal:
    # Going through str keeps 0.1 as Decimal("0.1") instead of a binary float approximation.
    return value if isinstance(value, Decimal) else Decimal(str(value))


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. Missing cache rates fall back to the input rate."""

    input: Decimal
    output: Decimal
    cache_read: Optional[Decimal] = None
    cache_write: Optional[Decimal] = None
    long_context_threshold: Optional[int] = None
    long_context: Optional["ModelPrice"] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelPrice":
        long_ctx = data.get("long_context")
        return cls(
            input=_dec(data["input"]),
            output=_dec(data["output"]),
            cache_read=_dec(data["cache_read"]) if data.get("cache_read") is not None else None,
            cache_write=_dec(data["cache_write"]) if data.get("cache_write") is not None else None,
            long_context_threshold=data.get("long_context_threshold"),
            long_context=cls.from_dict(long_ctx) if long_ctx else None,
        )

    def for_prompt(self, prompt_tokens: int) -> "ModelPrice":
        """The rates that apply to a prompt of this size."""
        if (
            self.long_context is not None
            and self.long_context_threshold is not None
            and prompt_tokens > self.long_context_threshold
        ):
            return self.long_context
        return self


PriceLike = Union[ModelPrice, Mapping[str, Any]]


class PricingTable:
    """Prices keyed by provider, then model id.

    Lookups accept exact ids and dated snapshots of a known id
    (``gpt-6-luna-2026-03-01`` resolves to ``gpt-6-luna``), but never a
    different model that merely shares a prefix (``gpt-6-astra-pro``).
    """

    def __init__(self, prices: Optional[Mapping[str, Mapping[str, PriceLike]]] = None):
        self._lock = threading.Lock()
        self._prices: Dict[str, Dict[str, ModelPrice]] = {}
        if prices:
            self.update(prices)

    @classmethod
    def default(cls) -> "PricingTable":
        text = resources.files("budget_guard").joinpath("pricing.json").read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PricingTable":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PricingTable":
        return cls({k: v for k, v in data.items() if not k.startswith("_")})

    def set(self, provider: str, model: str, price: PriceLike) -> None:
        if not isinstance(price, ModelPrice):
            price = ModelPrice.from_dict(price)
        with self._lock:
            self._prices.setdefault(provider, {})[model] = price

    def update(self, prices: Mapping[str, Mapping[str, PriceLike]]) -> None:
        """Merge prices in; existing models are overwritten, others kept."""
        for provider, models in prices.items():
            for model, price in models.items():
                self.set(provider, model, price)

    def get(self, model: str, provider: Optional[str] = None) -> ModelPrice:
        with self._lock:
            if provider is not None:
                candidates = [self._prices.get(provider, {})]
            else:
                candidates = list(self._prices.values())
            for models in candidates:
                if model in models:
                    return models[model]
            best: Optional[str] = None
            best_price: Optional[ModelPrice] = None
            for models in candidates:
                for known, price in models.items():
                    if (
                        model.startswith(known)
                        and _SNAPSHOT_SUFFIX.match(model[len(known):])
                        and (best is None or len(known) > len(best))
                    ):
                        best, best_price = known, price
        if best_price is None:
            raise UnknownModelError(model, provider)
        return best_price

    def __contains__(self, model: str) -> bool:
        try:
            self.get(model)
        except UnknownModelError:
            return False
        return True
