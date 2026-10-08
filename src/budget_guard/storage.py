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
    """Thread-safe, single-process storage. Totals are lost on restart."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._totals: Dict[str, Tuple[Decimal, Optional[float]]] = {}
        self._events: Dict[str, Deque[float]] = {}

    def _live_total(self, key: str, now: float) -> Optional[Tuple[Decimal, Optional[float]]]:
        entry = self._totals.get(key)
        if entry is not None and entry[1] is not None and entry[1] <= now:
            del self._totals[key]
            return None
        return entry

    def add(self, key: str, amount: Decimal, ttl_seconds: Optional[float] = None) -> Decimal:
        with self._lock:
            now = self._clock()
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
            events = self._events.setdefault(key, deque())
            cutoff = now - window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            events.append(now)
            return len(events)

    def clear(self) -> None:
        with self._lock:
            self._totals.clear()
            self._events.clear()
