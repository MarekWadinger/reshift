"""Tests for the Rolling wrapper's batch update / revert bookkeeping.

A minimal Rollable that only implements scalar ``update``/``revert`` exercises
the fallback branches (no ``update_many`` / ``revert_many``).
"""

from typing import Any

import numpy as np

from reshift.rolling import Rolling


class _RunningSum:
    """Rollable maintaining a running sum via scalar update/revert only."""

    def __init__(self) -> None:
        self.total = 0.0

    def update(self, *args: Any, **kwargs: Any) -> None:
        # Accepts a scalar (learn_one) or a batch (update_many fallback).
        del kwargs
        self.total += float(np.sum(args[0]))

    def revert(self, *args: Any, **kwargs: Any) -> None:
        del kwargs
        self.total -= float(np.sum(args[0]))


def test_rolling_learn_one_aliases_update():
    obj = _RunningSum()
    r = Rolling(obj, window_size=3)
    r.learn_one(1.0)
    r.learn_one(2.0)
    assert obj.total == 3.0
    assert len(r.window) == 2


def test_rolling_evicts_old_via_revert():
    obj = _RunningSum()
    r = Rolling(obj, window_size=2)
    for x in (1.0, 2.0, 10.0):  # window holds last two; 1.0 reverted out
        r.learn_one(x)
    assert obj.total == 12.0
    assert len(r.window) == 2


def test_rolling_update_many_fallback():
    obj = _RunningSum()
    r = Rolling(obj, window_size=4)
    # update_many with no update_many/revert_many on obj -> scalar fallbacks.
    r.update_many(np.array([1.0, 2.0, 3.0]))
    assert obj.total == 6.0
    r.update_many(np.array([4.0, 5.0]))  # window full -> revert oldest (1.0)
    assert obj.total == 14.0  # window keeps last four: 2+3+4+5
    assert len(r.window) == 4


class _BatchSum:
    """Rollable exposing batch update_many / revert_many."""

    def __init__(self) -> None:
        self.total = 0.0

    def update(self, *args: Any, **kwargs: Any) -> None:
        del kwargs
        self.total += float(np.sum(args[0]))

    def revert(self, *args: Any, **kwargs: Any) -> None:  # Rollable protocol
        del kwargs
        self.total -= float(np.sum(args[0]))

    def update_many(self, *args: Any, **kwargs: Any) -> None:
        del kwargs
        self.total += float(np.sum(args[0]))

    def revert_many(self, *args: Any, **kwargs: Any) -> None:
        del kwargs
        self.total -= float(np.sum(args[0]))


def test_rolling_update_many_batch_methods():
    obj = _BatchSum()
    r = Rolling(obj, window_size=4)
    r.update_many(np.array([1.0, 2.0, 3.0]))  # update_many branch
    assert obj.total == 6.0
    r.update_many(np.array([4.0, 5.0]))  # revert_many branch evicts 1.0
    assert obj.total == 14.0


def test_rolling_learn_many_aliases_update_many():
    obj = _RunningSum()
    r = Rolling(obj, window_size=5)
    r.learn_many(np.array([1.0, 2.0, 3.0]))
    assert obj.total == 6.0
