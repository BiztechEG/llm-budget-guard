# budget-guard-llm

Stop one user, one feature or one runaway loop from burning through your OpenAI or Anthropic bill.

`budget-guard-llm` wraps the official SDK clients, prices every call from the tokens the response actually
used, and enforces daily spend limits per user, per feature and in total. It also catches the same
request being sent over and over in a short time, which is what a stuck retry loop or agent looks like.

```bash
pip install budget-guard-llm
```

The package installs as `budget-guard-llm` and imports as `budget_guard`.

## Two lines

```python
from budget_guard import guard
client = guard(OpenAI(), user="u1", feature="chat")   # or Anthropic(), AsyncOpenAI(), AsyncAnthropic()
```

The rest of your code stays the same: `client.chat.completions.create(...)`,
`client.messages.create(...)`, streaming, async, all as before.

## Limits

Set them once at startup:

```python
import budget_guard
from budget_guard import Limits, LoopDetection

budget_guard.configure(
    limits=Limits(
        per_user=1.00,              # USD per user per day
        users={"vip": 20},          # overrides for specific users
        features={"chat": 5},       # per feature (per_feature= sets a default)
        daily_total=50,             # everything together
    ),
    loop_detection=LoopDetection(max_repeats=10, window_seconds=60),  # the default
    on_exceeded="raise",            # or "warn": log it and let the call through
    on_violation=alert_me,          # optional: called with the error in either mode
)
```

When a total has reached its cap, the next call raises `BudgetExceeded` before anything is sent to the
provider. A loop raises `LoopDetected`. Both subclass `BudgetError`:

```python
from budget_guard import BudgetError

try:
    reply = client.chat.completions.create(model="gpt-6-luna", messages=messages)
except BudgetError as err:
    return "You've hit today's limit, try again tomorrow."
```

`Limits` and `LoopDetection` each accept `action="raise"` or `"warn"` to override `on_exceeded`, so you
can, for example, block loops but only warn on spend.

How the checks behave:

- Limits are daily and reset at midnight UTC.
- The call that crosses a cap is allowed, because its cost is only known once it returns. Calls running
  at the same moment can each overshoot by one call.
- A limit of `0` blocks that user or feature entirely.
- A loop is the same request (model, messages and options) for the same user and feature, sent more than
  `max_repeats` times inside `window_seconds`. Two users asking the same question never count together.
  Pass `loop_detection=None` to turn it off.

## Who a call is for

Set defaults when wrapping, then override per request. `budget_context` works across threads and
asyncio tasks, so it fits request middleware:

```python
from budget_guard import budget_context

with budget_context(user=request.user.id):
    client.chat.completions.create(...)

client.with_context(feature="summary").chat.completions.create(...)   # a re-scoped copy
```

## Reading the totals

```python
budget_guard.spent(user="u1")         # Decimal USD today
budget_guard.spent(feature="chat")
budget_guard.spent()                  # daily total
budget_guard.remaining(user="u1")     # left under the limit, or None if no limit applies
```

To log every priced call, pass `on_record=` to `configure()`. It receives a `CallRecord` with the
provider, model, user, feature, token usage and cost.

## What is tracked

| Provider | Methods |
|---|---|
| OpenAI | `chat.completions.create/parse/stream`, `responses.create/parse/stream`, `completions.create` |
| Anthropic | `messages.create/parse/stream`, `beta.messages.create/parse/stream` |

Streams are counted when they finish or are closed. For OpenAI chat streams the guard turns on
`stream_options.include_usage` and hides the extra usage chunk unless you asked for it yourself.
`client.with_options(...)` stays guarded.

Not tracked yet: `with_raw_response`, `with_streaming_response`, batches, Anthropic's `tool_runner`,
embeddings, images and audio.

## Prices

Prices ship in `budget_guard/pricing.json` (USD per 1M tokens, with cached-input and long-context rates)
and can be changed at runtime:

```python
from budget_guard import default_pricing, PricingTable

default_pricing().set("openai", "my-finetune", {"input": "1.00", "output": "4.00"})
budget_guard.configure(pricing=PricingTable.from_file("my_prices.json"))
```

Dated model names such as `gpt-6-luna-2026-03-01` use the base model's price. A call to a model with no
price goes through uncounted with a warning; `configure(on_unknown_model="raise")` blocks it instead.

## Several servers

Totals are kept in memory, so each process counts on its own and totals reset on restart. To share
them across servers, implement the four methods of `budget_guard.Storage` (`add`, `get`, `hit`, `clear`)
on Redis or a database and pass it as `configure(storage=...)`. A Redis backend is planned.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests run the real OpenAI and Anthropic SDKs against a fake HTTP transport, so nothing is sent and
nothing is billed. Release steps are in [RELEASING.md](RELEASING.md).

## License

MIT
