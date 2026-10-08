# llm-budget-guard

Spend limits for OpenAI and Anthropic SDK calls: per user, per feature, and a daily total, plus loop detection.

Status: step 1 of 4 (pricing, cost calculation, storage). The SDK wrapper comes next.

```python
from budget_guard import cost_of_response

cost = cost_of_response(response, provider="anthropic")
print(cost.amount)  # Decimal USD, from the tokens the response actually used
```

Prices live in `src/budget_guard/pricing.json` (USD per 1M tokens). Override at runtime:

```python
from budget_guard import default_pricing
default_pricing().set("openai", "my-finetune", {"input": "1.00", "output": "4.00"})
```

Run tests: `python -m pytest`
