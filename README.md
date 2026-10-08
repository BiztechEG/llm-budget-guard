# llm-budget-guard

Spend limits for OpenAI and Anthropic SDK calls: per user, per feature, and a daily total, plus loop detection.

Status: steps 1 to 3 of 4 done (pricing, cost, storage, SDK wrapper, limits, loop detection). Packaging for PyPI comes next.

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

## Limits and loop detection

Set them once at startup:

```python
import budget_guard
from budget_guard import Limits, LoopDetection

budget_guard.configure(
    limits=Limits(per_user=1.00, users={"vip": 20}, features={"chat": 5}, daily_total=50),
    loop_detection=LoopDetection(max_repeats=10, window_seconds=60),  # this is the default
    on_exceeded="raise",       # or "warn" to log and let the call through
    on_violation=alert_me,     # optional: called with the error in either mode
)
```

All limits are daily, in USD, and reset at midnight UTC. Once a total reaches its cap, the next call
raises `BudgetExceeded` before anything is sent. The call that crosses the cap is still allowed, because
its cost is only known after it returns. Concurrent calls can overshoot by up to one call each.

A loop is the same request (same model, messages and options, same user and feature) sent more than
`max_repeats` times inside `window_seconds`. The next one raises `LoopDetected`. Pass
`loop_detection=None` to turn it off. `Limits` and `LoopDetection` each take `action="raise"|"warn"`
to override `on_exceeded`.

Both errors subclass `BudgetError`. `budget_guard.remaining(user="u1")` returns what is left today.

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
