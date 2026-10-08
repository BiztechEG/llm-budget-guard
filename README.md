# llm-budget-guard

Spend limits for OpenAI and Anthropic SDK calls: per user, per feature, and a daily total, plus loop detection.

Status: steps 1 and 2 of 4 done (pricing, cost, storage, SDK wrapper). Limits and loop detection come next.

## Two lines

```python
from budget_guard import guard
client = guard(OpenAI(), user="u1", feature="chat")   # or Anthropic(), AsyncOpenAI(), AsyncAnthropic()
```

The rest of your code stays the same. Each call is priced from the tokens the response actually used and added to today's totals:

```python
import budget_guard
budget_guard.spent(user="u1")        # Decimal USD today (UTC)
budget_guard.spent(feature="chat")
budget_guard.spent()                 # daily total
```

Per-request attribution, for web apps (works with threads and asyncio):

```python
from budget_guard import budget_context
with budget_context(user=request.user.id):
    client.chat.completions.create(...)
```

Or a re-scoped copy: `client.with_context(user="u2")`.

## What is tracked

| Provider | Methods |
|---|---|
| OpenAI | `chat.completions.create/parse/stream`, `responses.create/parse/stream`, `completions.create` |
| Anthropic | `messages.create/parse/stream`, `beta.messages.create/parse/stream` |

Streams are counted when they finish or are closed. For OpenAI chat streams the guard turns on `stream_options.include_usage` and hides the extra usage chunk unless you asked for it yourself.

Not tracked yet: `with_raw_response`, `with_streaming_response`, batches, and Anthropic's `tool_runner`.

## Prices

Prices live in `src/budget_guard/pricing.json` (USD per 1M tokens). Override at runtime:

```python
from budget_guard import default_pricing
default_pricing().set("openai", "my-finetune", {"input": "1.00", "output": "4.00"})
```

Calls to a model with no price go through uncounted with a warning. `budget_guard.configure(on_unknown_model="raise")` blocks them instead.

Run tests: `pip install -e .[dev] && python -m pytest`
