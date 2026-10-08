"""Normalise the `usage` block of OpenAI and Anthropic responses into one shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class Usage:
    """Token counts split by how each kind is billed."""

    input_tokens: int = 0  # uncached input only
    output_tokens: int = 0  # includes reasoning tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read, cached or not."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(obj: Any, name: str) -> int:
    return int(_get(obj, name) or 0)


def from_openai(usage: Any) -> Usage:
    """Chat Completions (`prompt_tokens`) and Responses API (`input_tokens`) usage."""
    if _get(usage, "prompt_tokens") is not None:
        prompt = _int(usage, "prompt_tokens")
        output = _int(usage, "completion_tokens")
        details = _get(usage, "prompt_tokens_details")
    else:
        prompt = _int(usage, "input_tokens")
        output = _int(usage, "output_tokens")
        details = _get(usage, "input_tokens_details")
    cached = min(_int(details, "cached_tokens"), prompt)
    # OpenAI counts cached tokens inside the prompt total.
    return Usage(input_tokens=prompt - cached, output_tokens=output, cache_read_tokens=cached)


def from_anthropic(usage: Any) -> Usage:
    """Anthropic reports uncached input and the two cache kinds separately."""
    return Usage(
        input_tokens=_int(usage, "input_tokens"),
        output_tokens=_int(usage, "output_tokens"),
        cache_read_tokens=_int(usage, "cache_read_input_tokens"),
        cache_write_tokens=_int(usage, "cache_creation_input_tokens"),
    )


_PARSERS = {"openai": from_openai, "anthropic": from_anthropic}


def extract_usage(response: Any, provider: str) -> Tuple[Optional[str], Usage]:
    """Return ``(model, usage)`` from an SDK response object or its dict form."""
    try:
        parse = _PARSERS[provider]
    except KeyError:
        raise ValueError(f"Unsupported provider {provider!r}; expected one of {sorted(_PARSERS)}") from None
    return _get(response, "model"), parse(_get(response, "usage"))
