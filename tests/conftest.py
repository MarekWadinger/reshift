"""Shared fixtures: one known linear system and one synthetic CPD process.

The CPD process is the level-pulse sinusoid from
``examples/11_window_duration_matching.py`` -- a benign carrier with a temporary
level shift that a low-rank linear model cannot reconstruct, so the
reconstruction-error score fires.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: no display needed for plot tests

ROT_THETA = 0.3
ROT_DECAY = 0.99
B_TRUE = np.array([[0.1], [0.2]])


@pytest.fixture
def rotation_A() -> np.ndarray:
    """Stable 2x2 rotation-decay operator with known eigenvalue magnitude."""
    c, s = np.cos(ROT_THETA), np.sin(ROT_THETA)
    return np.array([[c, -s], [s, c]]) * ROT_DECAY


@pytest.fixture
def linear_trajectory(rotation_A: np.ndarray) -> np.ndarray:
    """Snapshots of x_{k+1} = A x_k for the rotation operator, shape (n, 2)."""
    n = 60
    X = np.zeros((n, 2))
    X[0] = [1.0, 0.5]
    for k in range(1, n):
        X[k] = rotation_A @ X[k - 1]
    return X


@pytest.fixture
def controlled_trajectory(
    rotation_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Snapshots of x_{k+1} = A x_k + B u_k with random control, (X, U)."""
    rng = np.random.RandomState(0)
    n = 80
    U = rng.randn(n, 1)
    X = np.zeros((n, 2))
    X[0] = [1.0, 0.5]
    for k in range(1, n):
        X[k] = rotation_A @ X[k - 1] + (B_TRUE @ U[k - 1 : k].T).ravel()
    return X, U


def make_level_pulse(
    n: int, pulses: list[tuple[int, int]], amp: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Sinusoidal carrier with additive level pulses (onset, duration)."""
    rng = np.random.default_rng(seed)
    x = np.sin(2 * np.pi * 0.05 * np.arange(n)) + rng.normal(0, 0.1, n)
    for onset, dur in pulses:
        x[onset : onset + dur] += amp
    return x


@pytest.fixture
def pulse_signal() -> np.ndarray:
    """A single transient level pulse well after the detector's grace period."""
    return make_level_pulse(1500, [(900, 150)])
