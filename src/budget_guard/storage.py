"""Where spend totals and request counts live.

`Storage` is deliberately small so a Redis backend can implement it with
INCRBYFLOAT + EXPIRE for `add` and a sorted set for `hit`.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from decimal import Decimal
from typing import Callable, Deque, Dict, Optional, Tuple

ZERO = Decimal(0)


class Storage(ABC):
    @abstractmethod
    def add(self, key: str, amount: Decimal, ttl_seconds: Optional[float] = None) -> Decimal:
        """Add to a running total and return the new total.

        `ttl_seconds` applies when the key is created; later adds don't extend it,
        so a key like ``spend:user:42:2026-10-08`` expires on schedule.
        """

    @abstractmethod
    def get(self, key: str) -> Decimal:
        """Current total, or 0 if the key is missing or expired."""

    @abstractmethod
    def hit(self, key: str, window_seconds: float) -> int:
        """Record one event now and return how many fall inside the last `window_seconds`."""

    @abstractmethod
    def clear(self) -> None:
        """Drop everything."""


class InMemoryStorage(Storage):
    """Thread-safe, single-process storage. Totals are lost on restart.

    Expired totals and request counts are dropped every `sweep_seconds`, so a
    long-running process doesn't keep a key for every request it ever saw.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic, sweep_seconds: float = 60.0):
        self._clock = clock
        self._lock = threading.Lock()
        self._totals: Dict[str, Tuple[Decimal, Optional[float]]] = {}
        self._events: Dict[str, Deque[float]] = {}
        self._windows: Dict[str, float] = {}
        self._sweep_seconds = sweep_seconds
        self._next_sweep = clock() + sweep_seconds

    def _sweep(self, now: float) -> None:
        if now < self._next_sweep:
            return
        self._next_sweep = now + self._sweep_seconds
        for key in [k for k, (_, expires_at) in self._totals.items() if expires_at is not None and expires_at <= now]:
            del self._totals[key]
        for key in [k for k, events in self._events.items() if not events or events[-1] <= now - self._windows[k]]:
            del self._events[key]
            del self._windows[key]

    def _live_total(self, key: str, now: float) -> Optional[Tuple[Decimal, Optional[float]]]:
        entry = self._totals.get(key)
        if entry is not None and entry[1] is not None and entry[1] <= now:
            del self._totals[key]
            return None
        return entry

    def add(self, key: str, amount: Decimal, ttl_seconds: Optional[float] = None) -> Decimal:
        with self._lock:
            now = self._clock()
            self._sweep(now)
            entry = self._live_total(key, now)
            if entry is None:
                expires_at = now + ttl_seconds if ttl_seconds is not None else None
                total = amount
            else:
                total, expires_at = entry[0] + amount, entry[1]
            self._totals[key] = (total, expires_at)
            return total

    def get(self, key: str) -> Decimal:
        with self._lock:
            entry = self._live_total(key, self._clock())
            return entry[0] if entry is not None else ZERO

    def hit(self, key: str, window_seconds: float) -> int:
        with self._lock:
            now = self._clock()
            self._sweep(now)
            events = self._events.setdefault(key, deque())
            self._windows[key] = window_seconds
            cutoff = now - window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            events.append(now)
            return len(events)

    def clear(self) -> None:
        with self._lock:
            self._totals.clear()
            self._events.clear()
            self._windows.clear()
