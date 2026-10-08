import threading
from decimal import Decimal

from budget_guard import InMemoryStorage


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_add_and_get():
    store = InMemoryStorage()
    assert store.get("k") == 0
    assert store.add("k", Decimal("0.10")) == Decimal("0.10")
    assert store.add("k", Decimal("0.20")) == Decimal("0.30")
    assert store.get("k") == Decimal("0.30")


def test_ttl_set_on_create_and_not_extended():
    clock = FakeClock()
    store = InMemoryStorage(clock=clock)
    store.add("day", Decimal(1), ttl_seconds=10)
    clock.now += 8
    store.add("day", Decimal(1), ttl_seconds=10)
    assert store.get("day") == 2
    clock.now += 2
    assert store.get("day") == 0
    assert store.add("day", Decimal(5), ttl_seconds=10) == 5


def test_hit_sliding_window():
    clock = FakeClock()
    store = InMemoryStorage(clock=clock)
    assert store.hit("req", 10) == 1
    clock.now += 5
    assert store.hit("req", 10) == 2
    clock.now += 6  # first hit is now 11s old
    assert store.hit("req", 10) == 2
    assert store.hit("other", 10) == 1


def test_clear():
    store = InMemoryStorage()
    store.add("k", Decimal(1))
    store.hit("h", 10)
    store.clear()
    assert store.get("k") == 0
    assert store.hit("h", 10) == 1


def test_concurrent_adds_are_not_lost():
    store = InMemoryStorage()

    def worker():
        for _ in range(1000):
            store.add("k", Decimal("0.001"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.get("k") == Decimal("8.000")
