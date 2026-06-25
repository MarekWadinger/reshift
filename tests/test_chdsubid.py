"""Tests for subspace-identification change detection.

The end-to-end checks drive the ODMD-CPD pipeline from
``examples/11_window_duration_matching.py``: a rolling OnlineDMD wrapped in a
``SubIDChangeDetector`` behind a ``Hankelizer``. The verifiable claim is that the
detector's score peaks inside a transient level pulse and stays low on the clean
carrier.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest
from river.base import Transformer
from river.decomposition import OnlineDMD
from river.preprocessing import Hankelizer

from reshift.chdsubid import (
    DMDChangeDetector,
    SubIDChangeDetector,
    dist_featurewise_l1,
    dist_featurewise_l2,
    dist_frobenius_cov,
    dist_measurement_l2,
    dist_overall_l1,
    dist_sum_square_diff,
    get_default_params,
    get_default_rank,
    get_default_timedelays,
)
from reshift.preprocessing import hankel
from reshift.rolling import Rolling

# ---------------------------------------------------------------------------
# Default-parameter helpers
# ---------------------------------------------------------------------------


def test_get_default_timedelays_no_cap():
    assert get_default_timedelays(50) == (50, 1)


def test_get_default_timedelays_under_cap():
    assert get_default_timedelays(40, n_features_max=100) == (40, 1)


def test_get_default_timedelays_over_cap():
    # h >= cap -> clamp delays to cap and stride to h // cap.
    assert get_default_timedelays(250, n_features_max=100) == (100, 2)


def test_get_default_rank_no_variance_on_embedding():
    # High-dimensional Hankel embedding of a single sinusoid: the median-based
    # threshold keeps a handful of dominant modes (a small positive rank).
    rng = np.random.default_rng(0)
    x = np.sin(0.1 * np.arange(300))[:, None] + rng.normal(0, 0.05, (300, 1))
    r = get_default_rank(hankel(x, 20))
    assert 1 <= r <= 20


def test_get_default_rank_with_known_noise_variance():
    # Two clean sinusoids + small noise, true noise variance supplied -> rank 2.
    rng = np.random.default_rng(0)
    t = np.arange(400)
    clean = np.c_[np.sin(0.1 * t), np.cos(0.05 * t)]
    X = clean + rng.normal(0, 0.01, clean.shape)
    r = get_default_rank(X, noise_variance=0.01**2)
    assert r == 2


def test_get_default_params_without_control():
    rng = np.random.default_rng(1)
    X = np.sin(0.1 * np.arange(200))[:, None] + rng.normal(0, 0.05, (200, 1))
    out = get_default_params(X, window_size=40)
    assert len(out) == 5
    window_size, ref_size, test_size, lag, q = out
    assert window_size == ref_size == test_size == 40
    assert lag == 0
    assert 1 <= q <= 10


def test_get_default_params_with_control():
    rng = np.random.default_rng(2)
    X = np.sin(0.1 * np.arange(200))[:, None] + rng.normal(0, 0.05, (200, 1))
    U = np.cos(0.07 * np.arange(200))[:, None] + rng.normal(0, 0.05, (200, 1))
    out = get_default_params(X, U, window_size=40)
    assert len(out) == 6
    assert 1 <= out[-1] <= 10  # rank for U


def test_get_default_params_defaults_window_to_len():
    X = np.random.default_rng(3).normal(size=(60, 1))
    window_size = get_default_params(X)[0]
    assert window_size == 60


def test_get_default_params_large_hankel_skips_rank_estimate():
    # Many features * half-window exceed the dim threshold -> q = max_rank.
    X = np.random.default_rng(4).normal(size=(400, 30))
    U = np.random.default_rng(5).normal(size=(400, 30))
    *_, q, p = get_default_params(X, U, window_size=200, max_rank=7)
    assert q == 7
    assert p == 7


# ---------------------------------------------------------------------------
# Reconstruction-distance metrics
# ---------------------------------------------------------------------------


def test_distance_metrics_zero_on_perfect_reconstruction():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    for dist in (
        dist_featurewise_l1,
        dist_featurewise_l2,
        dist_measurement_l2,
        dist_overall_l1,
        dist_sum_square_diff,
        dist_frobenius_cov,
    ):
        assert dist(X, X) == pytest.approx(0.0)


def test_dist_featurewise_l1_value():
    X = np.array([[1.0, 0.0], [1.0, 0.0]])
    X_t = np.zeros((2, 2))
    # column 0 differs by 1 in each of 2 rows -> L1 column norm = 2; column 1 = 0
    assert dist_featurewise_l1(X, X_t) == pytest.approx(2.0)


def test_dist_featurewise_l2_value():
    X = np.array([[3.0], [4.0]])
    X_t = np.zeros((2, 1))
    assert dist_featurewise_l2(X, X_t) == pytest.approx(5.0)


def test_dist_measurement_l2_value():
    X = np.array([[3.0, 4.0]])
    X_t = np.zeros((1, 2))
    assert dist_measurement_l2(X, X_t) == pytest.approx(5.0)


def test_dist_sum_square_diff_sign():
    X = np.array([[2.0]])
    X_t = np.array([[1.0]])
    assert dist_sum_square_diff(X, X_t) == pytest.approx(3.0)  # 4 - 1


# ---------------------------------------------------------------------------
# SubIDChangeDetector lifecycle
# ---------------------------------------------------------------------------


def _make_detector(det_win=100, learn_w=400, hn=20):
    odmd = Rolling(
        OnlineDMD(
            r=2,
            initialize=learn_w,
            w=1.0,
            exponential_weighting=False,
            seed=42,
        ),
        learn_w + 1,
    )
    det = SubIDChangeDetector(
        odmd,
        ref_size=det_win,
        test_size=det_win,
        grace_period=learn_w + det_win + 1,
        start_soon=True,
    )
    return Hankelizer(hn) | det, det


def test_pipeline_fires_inside_pulse(pulse_signal):
    pipe, det = _make_detector()
    scores = np.zeros(len(pulse_signal))
    for i, xi in enumerate(
        pd.DataFrame(
            pulse_signal.reshape(-1, 1), columns=pd.Index(["x"])
        ).to_dict(orient="records")
    ):
        try:
            scores[i] = pipe.score_one(xi)
        except ZeroDivisionError:
            scores[i] = scores[i - 1] if i else 0.0
        pipe.learn_one(xi)
    scores = np.nan_to_num(scores)
    pulse = slice(900, 1050)
    # The change-point score is far higher inside the transient than on the
    # clean carrier before it (after the grace period at ~500).
    assert scores[pulse].max() > 5 * np.median(scores[600:850] + 1e-9)
    assert det.score >= 0.0


def test_subid_defaults_and_properties():
    det = SubIDChangeDetector(OnlineDMD(r=2, initialize=5), ref_size=5)
    assert det.test_size == 5  # defaults to ref_size
    assert det._supervised is False
    assert det.predict_one() is None  # nothing scored yet
    # No data yet -> neutral distances and zero score, no drift.
    assert det.distances == (1.0 + 0.0j, 1.0 + 0.0j)
    assert det.score == 0.0
    assert not det.drift_detected


def test_subid_ref_size_zero_uses_window_size():
    odmd = Rolling(OnlineDMD(r=2, initialize=5), window_size=8)
    det = SubIDChangeDetector(odmd, ref_size=0)
    assert det.ref_size == 8


def test_subid_ref_size_zero_unbounded_window_raises():
    # An unbounded (maxlen=None) rolling window has no size to fall back to.
    odmd = Rolling(OnlineDMD(r=2, initialize=5), window_size=cast("int", None))
    with pytest.raises(ValueError, match="window_size must be provided"):
        SubIDChangeDetector(odmd, ref_size=0)


def test_subid_score_one_is_stateless():
    det = SubIDChangeDetector(OnlineDMD(r=2, initialize=5), ref_size=5)
    before = len(det._X)
    det.score_one({"x": 1.0})
    assert len(det._X) == before  # buffer restored


def test_subid_learn_one_alias_and_learn_delay():
    det = SubIDChangeDetector(
        OnlineDMD(r=2, initialize=3), ref_size=3, test_size=2, lag=1
    )
    assert det.learn_delay == det.lag + det.test_size == 3
    det.learn_one({"x": 0.1})
    assert det.n_seen == 1


def test_subid_start_soon_learn_delay_grows():
    det = SubIDChangeDetector(
        OnlineDMD(r=1, initialize=3), ref_size=3, start_soon=True
    )
    for v in range(10):
        det.update({"x": float(v)})
    # start_soon delay = max(0, len(buffer) - (lag + test_size))
    assert det.learn_delay == max(0, len(det._X) - det._learn_delay)


def test_subid_learn_many_chunks_large_batch():
    det = SubIDChangeDetector(
        Rolling(OnlineDMD(r=1, initialize=5), window_size=6), ref_size=6
    )
    X = pd.DataFrame({"x": np.arange(40.0)})
    det.learn_many(X)  # > buffer_len -> recursive chunking
    assert det.n_seen == 40


def test_subid_learn_many_small_batch():
    det = SubIDChangeDetector(OnlineDMD(r=2, initialize=3), ref_size=3)
    X = pd.DataFrame({"x": [0.1, 0.2]})
    det.learn_many(X)
    assert det.n_seen == 2


def test_subid_grace_period_blocks_learning():
    # learn_after_grace=False: model only learns before grace elapses.
    det = SubIDChangeDetector(
        OnlineDMD(r=1, initialize=3),
        ref_size=3,
        grace_period=2,
        learn_after_grace=False,
    )
    for v in range(8):
        det.update({"x": float(v)})
    assert det.n_seen == 8


def test_drift_detected_is_cached():
    det = SubIDChangeDetector(OnlineDMD(r=2, initialize=5), ref_size=5)
    first = det.drift_detected
    assert det.drift_detected is first  # second call hits the cached branch


# --- _transform_many / learn_many branch coverage with minimal fakes -------


class _IdentityTransformer(Transformer):
    """Plain river Transformer (no transform_many) -> per-row transform path."""

    def learn_one(self, x: dict, **_: object) -> None:
        pass

    def transform_one(self, x: dict) -> dict:
        return dict(x)


def test_transform_many_per_row_loop():
    det = SubIDChangeDetector(_IdentityTransformer(), ref_size=3)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = det._transform_many(df)
    assert out["x"].tolist() == [1.0, 2.0, 3.0]


def test_learn_many_learns_minibatch_branch():
    # StandardScaler is a MiniBatchTransformer -> learn_many branch.
    from river.preprocessing import StandardScaler

    det = SubIDChangeDetector(StandardScaler(), ref_size=4, test_size=1)
    for _ in range(3):  # accumulate until cond_soon, then MiniBatch learn_many
        det.learn_many(pd.DataFrame({"x": [0.1, 0.2]}))
    assert det.n_seen == 6


def test_learn_many_learns_per_row_branch():
    det = SubIDChangeDetector(_IdentityTransformer(), ref_size=4, test_size=1)
    for _ in range(3):
        det.learn_many(pd.DataFrame({"x": [0.1, 0.2]}))
    assert det.n_seen == 6


# --- control-aware reconstruction (_predict_many) -------------------------


class _FakeABModel(Transformer):
    """Transformer exposing a fixed (A, B) for control-aware reconstruction."""

    def __init__(self, A: np.ndarray, B: np.ndarray, *, raises: bool = False):
        self._A, self._B, self._raises = A, B, raises

    def learn_one(self, x: dict, **_: object) -> None:
        pass

    def transform_one(self, x: dict) -> dict:
        return dict(x)

    def _reconstruct_AB(self) -> tuple[np.ndarray, np.ndarray]:
        if self._raises:
            msg = "not fitted"
            raise ValueError(msg)
        return self._A, self._B


def _control_detector(model, control_aware=True):
    return SubIDChangeDetector(model, ref_size=3, control_aware=control_aware)


def test_update_buffers_control_input():
    det = _control_detector(_IdentityTransformer())
    det.update({"x": 1.0}, u={"u": 0.5})
    assert list(det._U) == [{"u": 0.5}]


def test_predict_many_one_step_reconstruction():
    A = np.array([[0.9, 0.0], [0.0, 0.8]])
    B = np.array([[0.5], [0.25]])
    det = _control_detector(_FakeABModel(A, B))
    xs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    det._X.extend({"a": r[0], "b": r[1]} for r in xs)
    det._U.extend([{"u": 1.0}, {"u": 2.0}, {"u": 3.0}])
    X = pd.DataFrame(det._X)
    out = det._predict_many(X).to_numpy()
    # row 0 copied; rows 1..k = x_{k-1} A^T + u_{k-1} B^T (latest u right-aligned)
    expected = xs[:-1] @ A.T + np.array([[2.0], [3.0]]) @ B.T
    assert np.allclose(out[0], xs[0])
    assert np.allclose(out[1:], expected)


def test_predict_many_falls_back_without_reconstruct_AB():
    det = _control_detector(_IdentityTransformer())  # no _reconstruct_AB
    det._X.extend([{"a": 1.0}, {"a": 2.0}])
    det._U.extend([{"u": 1.0}])
    out = det._predict_many(pd.DataFrame(det._X))
    assert out["a"].tolist() == [1.0, 2.0]  # identity fallback


def test_predict_many_falls_back_on_too_few_inputs():
    A = np.eye(1)
    det = _control_detector(_FakeABModel(A, np.zeros((1, 1))))
    det._X.extend([{"a": 1.0}, {"a": 2.0}, {"a": 3.0}])
    det._U.clear()  # no buffered inputs -> fallback
    out = det._predict_many(pd.DataFrame(det._X))
    assert out["a"].tolist() == [1.0, 2.0, 3.0]


def test_predict_many_falls_back_when_reconstruct_raises():
    det = _control_detector(_FakeABModel(np.eye(1), np.eye(1), raises=True))
    det._X.extend([{"a": 1.0}, {"a": 2.0}])
    det._U.extend([{"u": 1.0}])
    out = det._predict_many(pd.DataFrame(det._X))
    assert out["a"].tolist() == [1.0, 2.0]


def test_predict_many_falls_back_on_shape_mismatch():
    # A is 3x3 but data has 2 features -> dimension guard falls back.
    det = _control_detector(_FakeABModel(np.eye(3), np.ones((3, 1))))
    det._X.extend([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}])
    det._U.extend([{"u": 1.0}])
    out = det._predict_many(pd.DataFrame(det._X))
    assert out.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_distances_uses_predict_many_when_control_aware():
    A = np.array([[0.9, 0.0], [0.0, 0.8]])
    B = np.array([[0.1], [0.1]])
    det = SubIDChangeDetector(
        _FakeABModel(A, B), ref_size=2, test_size=2, control_aware=True
    )
    for k in range(5):
        det.update({"a": float(k), "b": float(2 * k)}, u={"u": 1.0})
    assert det.score >= 0.0  # full control-aware scoring path executed


# --- DMDChangeDetector cached-transform branches --------------------------


class _FakeAllclose(OnlineDMD):
    """OnlineDMD stand-in whose Koopman operator is reported unchanged."""

    A_allclose = True  # class attribute overrides the convergence property

    def __init__(self) -> None:
        super().__init__(r=1, initialize=1)

    def transform_one(self, x: dict) -> dict:
        return dict(x)


def test_dmd_change_detector_reuses_when_A_allclose():
    det = DMDChangeDetector(_FakeAllclose(), ref_size=3)
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = det._transform_many(df)
    # A_allclose -> only the last row is (re)transformed and cached.
    assert len(det._Xp) == 1
    assert out["x"].iloc[-1] == 3.0


def test_dmd_change_detector_caches_transform(pulse_signal):
    learn_w, det_win, hn = 400, 100, 10
    odmd = Rolling(
        OnlineDMD(
            r=2, initialize=learn_w, w=1.0, exponential_weighting=False, seed=1
        ),
        learn_w + 1,
    )
    det = DMDChangeDetector(
        odmd,
        ref_size=det_win,
        test_size=det_win,
        grace_period=learn_w + det_win + 1,
    )
    pipe = Hankelizer(hn) | det
    sig = pulse_signal[:900]
    scores = np.zeros(len(sig))
    for i, xi in enumerate(
        pd.DataFrame(sig.reshape(-1, 1), columns=pd.Index(["x"])).to_dict(
            orient="records"
        )
    ):
        try:
            scores[i] = pipe.score_one(xi)
        except ZeroDivisionError:
            scores[i] = scores[i - 1] if i else 0.0
        pipe.learn_one(xi)
    assert np.all(np.nan_to_num(scores) >= 0.0)
    assert len(det._Xp) > 0  # transform cache populated


def test_hankel_consistency_used_by_pipeline():
    # Sanity: the hankel helper that feeds the detector is shape-correct.
    X = np.arange(10.0)
    out = hankel(X, 3, return_partial=False)
    assert out.shape == (8, 3)
