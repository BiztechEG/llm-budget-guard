# Changelog

## 0.1.2 (unreleased)

Fixes from a code review.

- Calls still running now count toward limits at an estimated cost, so parallel calls can't all pass a limit before the first one is recorded. In a test, 20 parallel calls against a $0.10 limit used to spend $2.00; now one goes through. The estimate comes from the request's text and its `max_tokens` (`reserve_output_tokens=` when it sets none). `BudgetExceeded.reserved` holds the amount held by calls still running.
- An OpenAI stream closed before its last chunk is counted at its estimated cost instead of not at all.
- `parse()` with a Pydantic model (`response_format=`, `text_format=`, `output_format=`) no longer fails with a `TypeError` while loop detection is on.
- `with_raw_response` and `with_streaming_response` calls, and clients made with Anthropic's `with_middleware()`, now go through the limit and loop checks instead of skipping them. Raw calls are not priced yet; a warning says so once.
- A stream dropped before its end (for example, the browser disconnected) now counts the usage it had already reported. Anthropic streams report input tokens at the start, so those are no longer lost.
- `InMemoryStorage` drops expired totals and old request counts every minute, so a long-running process no longer grows without bound.
- Numeric user and feature ids (`user=request.user.id`) now match overrides keyed by the same id as text, and the reverse.
- `Limits` rejects NaN, which made every call fail, and negative amounts.
- Generator arguments (`messages=(m for m in ...)`) are fingerprinted by content, so loop detection sees repeats.
- A `GuardedClient` can be copied with `copy.copy()`.

## 0.1.1 (2026-10-08)

First release on PyPI, as `budget-guard-llm`. (0.1.0 was tagged as `budget-guard` but never published, because PyPI rejected that name as too similar to `budgetguard`.)

- `guard(client, user=..., feature=...)` wraps OpenAI and Anthropic clients, sync and async, including streams and `.stream()` helpers.
- Cost from the tokens each response actually used, including cached input and long-context rates, from an updatable price table.
- Daily (UTC) spend limits per user, per feature and in total, with per-user and per-feature overrides.
- Loop detection for the same request repeated within a short window.
- `on_exceeded="raise"` or `"warn"`, an `on_violation` callback, and `spent()` / `remaining()` helpers.
- In-memory storage behind a small `Storage` interface for other backends.
