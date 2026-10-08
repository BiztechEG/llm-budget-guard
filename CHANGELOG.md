# Changelog

## 0.1.0 (unreleased)

First release.

- `guard(client, user=..., feature=...)` wraps OpenAI and Anthropic clients, sync and async, including streams and `.stream()` helpers.
- Cost from the tokens each response actually used, including cached input and long-context rates, from an updatable price table.
- Daily (UTC) spend limits per user, per feature and in total, with per-user and per-feature overrides.
- Loop detection for the same request repeated within a short window.
- `on_exceeded="raise"` or `"warn"`, an `on_violation` callback, and `spent()` / `remaining()` helpers.
- In-memory storage behind a small `Storage` interface for other backends.
