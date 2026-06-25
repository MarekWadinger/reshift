"""Tests for the hand-rolled DMD / DMDwC against known linear systems.

DMD recovers the operator A = Y X^+; for exact linear data this is the true A,
so every assertion checks a verifiable numerical value, not just a shape.
"""

import numpy as np
import pytest

from reshift.dmd import DMD, DMDwC
from tests.conftest import B_TRUE, ROT_DECAY


def test_dmd_recovers_operator(linear_trajectory, rotation_A):
    d = DMD()
    d.fit(linear_trajectory)
    assert d.A is not None
    assert np.allclose(d.A.real, rotation_A, atol=1e-6)
    # rotation-decay -> both eigenvalues have magnitude ROT_DECAY
    assert np.allclose(np.abs(d.Lambda), ROT_DECAY, atol=1e-6)


def test_dmd_predict_one_step(linear_trajectory, rotation_A):
    d = DMD()
    d.fit(linear_trajectory)
    pred = d.predict(linear_trajectory[0], forecast=1)
    assert np.allclose(pred[0], rotation_A @ linear_trajectory[0], atol=1e-6)


def test_dmd_predict_multistep(linear_trajectory, rotation_A):
    d = DMD()
    d.fit(linear_trajectory)
    pred = d.predict(linear_trajectory[0], forecast=3)
    expected = rotation_A @ rotation_A @ rotation_A @ linear_trajectory[0]
    assert pred.shape == (3, 2)
    assert np.allclose(pred[-1], expected, atol=1e-6)


def test_dmd_explicit_xy(linear_trajectory, rotation_A):
    # Passing X and Y explicitly must match the auto-shifted path.
    X, Y = linear_trajectory[:-1], linear_trajectory[1:]
    d = DMD()
    d.fit(X, Y)
    assert d.A is not None
    assert np.allclose(d.A.real, rotation_A, atol=1e-6)


def test_dmd_truncated_rank(linear_trajectory):
    # r < m uses the sparse svds branch and yields a rank-1 operator.
    d = DMD(r=1)
    d.fit(linear_trajectory)
    assert d.A is not None
    assert d.A.shape == (2, 2)
    assert np.linalg.matrix_rank(d.A.real, tol=1e-8) == 1


def test_dmd_xi_returns_amplitudes(linear_trajectory):
    d = DMD()
    d.fit(linear_trajectory)
    xi = d.xi
    assert xi.shape == (d.m,)
    assert np.all(np.isfinite(xi))


def test_dmd_vandermonde_C(linear_trajectory):
    d = DMD()
    d.fit(linear_trajectory)
    # First column of a Vandermonde matrix (increasing) is all ones.
    assert d.C.shape == (d.m, d.n)
    assert np.allclose(d.C[:, 0], 1.0)


def test_dmdwc_identifies_A_and_B(controlled_trajectory, rotation_A):
    X, U = controlled_trajectory
    d = DMDwC(r=3)
    d.fit(X, U=U)
    assert d.A is not None
    assert d.B is not None
    assert np.allclose(d.A.real, rotation_A, atol=1e-4)
    assert np.allclose(d.B.real, B_TRUE, atol=1e-4)


def test_dmdwc_predict_with_control(controlled_trajectory, rotation_A):
    X, U = controlled_trajectory
    d = DMDwC(r=3)
    d.fit(X, U=U)
    pred = d.predict(X[0], forecast=1, U=U[0:1])
    expected = rotation_A @ X[0] + (B_TRUE @ U[0:1].T).ravel()
    assert np.allclose(pred[0], expected, atol=1e-4)


def test_dmdwc_known_B(controlled_trajectory, rotation_A):
    X, U = controlled_trajectory
    d = DMDwC(r=2, B=B_TRUE)
    d.fit(X, U=U)
    assert d.A is not None
    assert np.allclose(d.A.real, rotation_A, atol=1e-4)
    pred = d.predict(X[0], forecast=1, U=U[0:1])
    expected = rotation_A @ X[0] + (B_TRUE @ U[0:1].T).ravel()
    assert np.allclose(pred[0], expected, atol=1e-4)


def test_dmdwc_no_control_falls_back_to_dmd(linear_trajectory, rotation_A):
    d = DMDwC(r=2)
    d.fit(linear_trajectory)  # U is None
    assert d.A is not None
    assert np.allclose(d.A.real, rotation_A, atol=1e-6)
    pred = d.predict(linear_trajectory[0], forecast=1)  # U None -> super()
    assert np.allclose(pred[0], rotation_A @ linear_trajectory[0], atol=1e-6)


def test_dmdwc_mismatched_timesteps_raises(rotation_A):
    X = np.random.RandomState(1).randn(20, 2)
    U = np.random.RandomState(2).randn(10, 1)
    # Known B skips the X|U hstack, so the explicit time-step check fires.
    with pytest.raises(ValueError, match="same number of time steps"):
        DMDwC(r=2, B=B_TRUE).fit(X[:-1], Y=X[1:], U=U)


def test_dmd_predict_before_fit_raises():
    d = DMD()
    d.A = None  # simulate unfitted operator
    with pytest.raises(RuntimeError, match="Fit the model"):
        d.predict(np.zeros(2))


def test_dmdwc_predict_before_fit_raises():
    d = DMDwC(r=2, B=B_TRUE)
    d.A = None
    with pytest.raises(RuntimeError, match="Fit the model"):
        d.predict(np.zeros(2), U=np.zeros((1, 1)))


def test_dmdwc_predict_forecast_mismatch_raises(controlled_trajectory):
    X, U = controlled_trajectory
    d = DMDwC(r=3)
    d.fit(X, U=U)
    with pytest.raises(ValueError, match="forecast number of time steps"):
        d.predict(X[0], forecast=3, U=U[0:1])
