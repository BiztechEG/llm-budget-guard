"""A rough size for a request before it is sent, used to hold budget for calls still in flight."""

from __future__ import annotations

import math
import re
from typing import Any, List, Mapping, Tuple

from .usage import Usage

CHARS_PER_TOKEN = 3  # on the low side: overestimates English a little, close for most other scripts
BLOB_TOKENS = 1000  # an image or file sent inline as base64
_OUTPUT_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")
_SKIPPED_KEYS = frozenset(
    {"model", "stream", "stream_options", "timeout", "extra_headers", "extra_query", "metadata", *_OUTPUT_KEYS}
)
_BASE64 = re.compile(r"[A-Za-z0-9+/=_-]+")


def estimate_usage(kwargs: Mapping[str, Any], default_output_tokens: int) -> Usage:
    """Input from the text in the request; output from its max-tokens setting, else `default_output_tokens`."""
    chars, blobs = _measure([v for k, v in kwargs.items() if k not in _SKIPPED_KEYS])
    output = next(
        (kwargs[k] for k in _OUTPUT_KEYS if isinstance(kwargs.get(k), int) and not isinstance(kwargs[k], bool)),
        default_output_tokens,
    )
    n = kwargs.get("n")
    choices = n if isinstance(n, int) and n > 1 else 1
    return Usage(input_tokens=math.ceil(chars / CHARS_PER_TOKEN) + blobs * BLOB_TOKENS, output_tokens=output * choices)


def _measure(values: List[Any]) -> Tuple[int, int]:
    """(characters of text, number of inline base64 blobs) anywhere in `values`."""
    chars = blobs = 0
    stack = list(values)
    while stack:
        value = stack.pop()
        if isinstance(value, str):
            if value.startswith("data:") or (len(value) > 256 and _BASE64.fullmatch(value)):
                blobs += 1
            else:
                chars += len(value)
        elif isinstance(value, Mapping):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif not isinstance(value, type):
            dump = getattr(value, "model_dump", None)  # SDK objects, e.g. a previous reply passed back in
            if callable(dump):
                stack.append(dump())
    return chars, blobs
