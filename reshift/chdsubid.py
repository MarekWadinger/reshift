"""Change Detection based on Subspace Identification algorithm."""

from collections import deque
from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np
import pandas as pd
from river.anomaly.base import AnomalyDetector
from river.base import MiniBatchTransformer, Transformer
from river.decomposition.rust_rolling_dmd import (
    RustRollingDMD,
    RustRollingDMDwC,
)

from .preprocessing import hankel
from .rolling import Rolling

if TYPE_CHECKING:
    from river.decomposition import OnlineDMD, OnlineDMDwC
    from river.utils.rolling import Rolling as RiverRolling

_RustTypes = (RustRollingDMD, RustRollingDMDwC)
_RollingTypes = (Rolling, RustRollingDMD, RustRollingDMDwC)

_HANKEL_DIM_THRESHOLD = 100


# # Default parameters
def get_default_timedelays(
    h: int,
    n_features_max: int | None = None,
) -> tuple[int, int]:
    """Compute default time-delay embedding parameters from window half-length.

    Args:
        h: Half the window size, used as the maximum number of time delays.
        n_features_max: Maximum number of features in the Hankel matrix. When
            provided, the number of delays and step size are adjusted so the
            embedded matrix does not exceed this column budget.

    Returns:
        Tuple of (number of delays, step size).

    """
    if n_features_max is None:
        return h, 1
    if h < n_features_max:
        h_ = h
        step = 1
    else:
        h_ = n_features_max
        step = (h) // n_features_max
    return h_, step


def get_default_rank(
    X: np.ndarray | pd.DataFrame,
    noise_variance: float | None = None,
) -> int:
    """Get default rank for the given data matrix.

    Args:
        X (np.ndarray): Data matrix.
        noise_variance (float | None, optional): Known noise variance. When ``None``,
            the optimal threshold is estimated from the median singular value.

    Returns:
        int: Default rank

    References:
        [1] Gavish, M., and Donoho L. D. (2014). The Optimal Hard Threshold for Singular Values is 4/sqrt(3). IEEE
        Transactions on Information Theory 60.8 (2014): 5040-5053.
        doi:[10.1109/TIT.2014.2323359](https://doi.org/10.1109/TIT.2014.2323359).

    """
    n, m = X.shape
    beta = m / n
    s = np.linalg.svd(X.T, compute_uv=False)
    if noise_variance is None:
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau = omega * np.median(s)
    else:
        lambda_opt = np.sqrt(
            2 * (beta + 1)
            + (8 * beta) / ((beta + 1) + np.sqrt(beta**2 + 14 * beta + 1)),
        )

        tau = lambda_opt * np.sqrt(n * noise_variance)
    return sum(s > tau)


@overload
def get_default_params(
    X: np.ndarray,
    U: None = None,
    window_size: int = 0,
    max_rank: int = 10,
) -> tuple[int, int, int, int, int]: ...
@overload
def get_default_params(
    X: np.ndarray,
    U: np.ndarray,
    window_size: int = 0,
    max_rank: int = 10,
) -> tuple[int, int, int, int, int, int]: ...
def get_default_params(
    X: np.ndarray,
    U: np.ndarray | None = None,
    window_size: int = 0,
    max_rank: int = 10,
) -> tuple[int, int, int, int, int] | tuple[int, int, int, int, int, int]:
    """Get default parameters for the given dataset and window size.

    Args:
        X (np.ndarray): Data matrix.
        U (np.ndarray | None, optional): Control input matrix. When provided, a rank for U is also returned.
        window_size (int): Window size. What kind of structural changes are we looking for?
        max_rank (int): Upper bound on the returned rank(s).

    References:
        [2] Moskvina, V., & Zhigljavsky, A. (2003). An Algorithm Based on Singular Spectrum Analysis for
        Change-Point Detection.
        Communications in Statistics - Simulation and Computation, 32(2), 319-352.
        doi:[10.1081/SAC-120017494](https://doi.org/10.1081/SAC-120017494).

    """
    if window_size == 0:
        window_size = len(X)
    # If window_size is not very large, then take half
    hn = window_size // 2
    hn, step = get_default_timedelays(hn, 100)
    # Base size
    ref_size = window_size
    lag = 0
    test_size = window_size
    # Optimal low-rank representation of signal with unknown noise variance
    if hn * X.shape[1] < _HANKEL_DIM_THRESHOLD:
        q = min(get_default_rank(hankel(X, hn, step)), max_rank)
    else:
        q = max_rank
    if U is not None:
        if hn * U.shape[1] < _HANKEL_DIM_THRESHOLD:
            p = min(get_default_rank(hankel(U, hn, step)), max_rank)
        else:
            p = max_rank

        return window_size, ref_size, test_size, lag, q, p

    return window_size, ref_size, test_size, lag, q


# --- Reconstruction-distance metrics ---------------------------------------
# Each maps an original window X and its reconstruction X_t to a scalar
# dissimilarity used for change scoring. SubIDChangeDetector uses
# `dist_featurewise_l1` by default; the alternatives were explored during
# development (issue #12) and are kept as named, testable functions rather
# than commented-out code. All operate on plain numpy arrays.
def dist_featurewise_l1(X: np.ndarray, X_t: np.ndarray) -> float:
    """Sum of per-feature (column-wise) L1 reconstruction errors (default).

    The 1-norm is sensitive to the number of measurements.
    """
    return float(np.sum(np.linalg.norm(X - X_t, axis=0, ord=1)))


def dist_featurewise_l2(X: np.ndarray, X_t: np.ndarray) -> float:
    """Sum of per-feature (column-wise) L2 reconstruction errors.

    The 2-norm is insensitive to the number of features.
    """
    return float(np.sum(np.linalg.norm(X - X_t, axis=0, ord=2)))


def dist_measurement_l2(X: np.ndarray, X_t: np.ndarray) -> float:
    """Sum of per-measurement (row-wise) L2 errors (total measurement error)."""
    return float(np.sum(np.linalg.norm(X - X_t, axis=1, ord=2)))


def dist_overall_l1(X: np.ndarray, X_t: np.ndarray) -> float:
    """Overall L1 error; scales with the magnitude of change."""
    return float(np.linalg.norm(X - X_t, ord=1).real)


def dist_sum_square_diff(X: np.ndarray, X_t: np.ndarray) -> float:
    """Difference of summed squares (squared-L1 energy), after Kawahara (2007)."""
    return float(np.sum(X**2) - np.sum(X_t**2))


def dist_frobenius_cov(X: np.ndarray, X_t: np.ndarray) -> float:
    """Frobenius norm of the data-vs-reconstruction covariance difference.

    Gives scores comparable to projecting and differencing covariances, but
    is cheaper since it skips the projection step.
    """
    return float(
        np.linalg.norm(np.inner(X, X) - np.inner(X_t, X_t), ord="fro"),
    )


class SubIDChangeDetector(AnomalyDetector):
    """Change-Point Detection on Subspace Identification.

    This class implements a change-point detection algorithm based on subspace identification. It uses a subspace
    identification algorithm to transform the data and then computes the distance between the original data and the
    transformed data. The distance is then used to detect changes in the data distribution.

    Args:
        subid (MiniBatchTransformer | Transformer | Rolling): Subspace identification algorithm
        ref_size (int): Size of the reference window
        test_size (int, optional): Size of the test window. Defaults to None -> ref_size.
        threshold (float, optional): Detection threshold. Defaults to 0.25.
        lag (int, optional): Time lag. Defaults to 0.
        grace_period (int, optional): Grace period. Defaults to 0.
        learn_after_grace (bool, optional): Learn after grace period. Defaults to True.
        start_soon (bool, optional): Start detection as soon as possible. Defaults to False.
        True (test: o, ref: x, both: w, none: .):
            3: www      (started)
            4: xwwo
            5: xxwoo
            6: xxxooo
            7: xxx.ooo
            8: .xxx.ooo
        False (test: o, ref: x, both: w, none: .):
            3: ooo
            4: .ooo
            5: x.ooo
            6: xx.ooo
            7: xxx.ooo  (started)
            8: .xxx.ooo

    """

    def __init__(
        self,
        subid: MiniBatchTransformer
        | Transformer
        | OnlineDMD
        | OnlineDMDwC
        | Rolling
        | RiverRolling
        | RustRollingDMD
        | RustRollingDMDwC,
        ref_size: int,
        test_size: int | None = None,
        threshold: float = 0.25,
        lag: int = 0,
        grace_period: int = 0,
        *,
        learn_after_grace: bool = True,
        start_soon: bool = False,
        control_aware: bool = False,
    ) -> None:
        r"""Initialize SubIDChangeDetector with subspace model and detection parameters.

        When ``control_aware`` is True and the subspace model exposes the
        learned control matrix (``_reconstruct_AB``, i.e. ``OnlineDMDwC``), the
        reconstruction used for scoring is the one-step prediction
        :math:`\\hat{x}_k = A x_{k-1} + B u_{k-1}` rather than the input-blind
        mode-subspace projection :math:`\\Phi\\Phi^\\top x`. This makes the score
        regress out the exogenous input, so a known forcing no longer leaks into
        the change statistic. Falls back to the projection score whenever the
        model has no control matrix or no inputs have been buffered.

        Examples:
            Control-aware scoring buffers the inputs and runs the prediction
            path (``OnlineDMDwC``), yielding a finite, non-negative score:

            >>> import numpy as np
            >>> from river.decomposition import OnlineDMDwC
            >>> from reshift.rolling import Rolling
            >>> det = SubIDChangeDetector(
            ...     Rolling(OnlineDMDwC(p=2, q=1, initialize=20, w=1.0), 40),
            ...     ref_size=20, test_size=20, grace_period=0,
            ...     start_soon=True, control_aware=True,
            ... )
            >>> for t in range(120):
            ...     u = float(np.sin(0.3 * t))
            ...     x = {"a": np.cos(0.1 * t) + 0.5 * u, "b": np.sin(0.1 * t)}
            ...     _ = det.score_one(x)
            ...     det.learn_one(x, u={"u": u})
            >>> bool(det.score >= 0.0) and len(det._U) > 0
            True
        """
        self.subid = subid
        self.threshold = threshold
        if ref_size == 0 and isinstance(subid, _RollingTypes):
            ref_size = subid.window_size
            # Since window_size is maxlen of deque in Rolling it may be None
            if ref_size is None:
                msg = "window_size must be provided for Rolling subid"
                raise ValueError(
                    msg,
                )
        self.ref_size = ref_size
        self.test_size = test_size if test_size is not None else ref_size
        self.lag = lag
        assert self.ref_size > 0
        assert self.test_size > 0
        assert self.test_size + self.lag >= 0
        self._learn_delay = self.lag + self.test_size
        # assert self.grace_period < self.test_size
        # TODO(MarekWadinger): omit grace period; start detection once Transformer is fitted (#11)
        self.grace_period = grace_period
        self.learn_after_grace = learn_after_grace
        self.start_soon = start_soon
        self.control_aware = control_aware
        self.n_seen = 0

        self._score: float | None = None
        self._distances: tuple[complex, complex] | None = None
        self._drift_detected: bool | None = None

        self._X: deque[dict] = deque(
            maxlen=self.ref_size + self.lag + self.test_size,
        )
        # Buffer of control inputs, appended in lockstep with ``_X`` during
        # ``update``; only populated/used when ``control_aware`` is set.
        self._U: deque[dict | None] = deque(
            maxlen=self.ref_size + self.lag + self.test_size,
        )

    @property
    def distances(self) -> tuple[complex, complex]:
        """Return the per-sample reconstruction distances for the reference and test windows.

        Returns:
            Tuple of (D_train, D_test) where each value is the mean reconstruction
            distance over the respective window. Returns (1+0j, 1+0j) when there is
            insufficient data or the grace period has not elapsed.

        """
        if self._distances is None:
            # Do inference after grace period and enough data is available
            lenght_X = len(self._X)
            if self.start_soon:
                # Ensure enough snapshots to fill both windows
                cond_soon = lenght_X > max(self.ref_size, self.test_size)
            else:
                cond_soon = lenght_X == self.ref_size + self._learn_delay

            if self.n_seen >= self.grace_period and cond_soon:
                X = pd.DataFrame(self._X)
                X_p = (
                    self._predict_many(X)
                    if self.control_aware
                    else self._transform_many(X)
                )
                D_train = (
                    self._compute_distance(
                        X.iloc[: self.ref_size, :],
                        X_p.iloc[: self.ref_size, :],
                    )
                    / self.ref_size
                )
                D_test = (
                    self._compute_distance(
                        X.iloc[-self.test_size :, :],
                        X_p.iloc[-self.test_size :, :],
                    )
                    / self.test_size
                )
                distances = (D_train, D_test)
            else:
                distances = (1.0 + 0.0j, 1.0 + 0.0j)
            self._distances = distances
        else:
            distances = self._distances
        return distances

    @property
    def drift_detected(self) -> bool:
        """Indicate whether a change-point has been detected.

        Returns:
            True when the anomaly score exceeds the threshold, False otherwise.

        """
        if self._drift_detected is None:
            drift_detected = self.score > self.threshold
            self._drift_detected = drift_detected
        else:
            drift_detected = self._drift_detected
        return drift_detected

    @property
    def score(self) -> float:
        """Compute the anomaly score as the relative increase in reconstruction error.

        Returns:
            Non-negative float; values above ``threshold`` indicate a change-point.

        """
        if self._score is None:
            # Under some circumstances score < 0
            #  - lower test noise
            #  - running normalization
            #  - ...
            D_train, D_test = self.distances
            score = (D_test / D_train) - 1
            # TODO(MarekWadinger): explore scoring options (#12) -- e.g. an
            # absolute distance difference, weighting individual terms, and a
            # proper use of the score's imaginary part
            if isinstance(score, complex):
                score: float = score.real + np.abs(score.imag)
            # TODO(MarekWadinger): document score shaping (#12)
            score = max(score, 0.0)
            self._score = score
        else:
            score = self._score
        return score

    @property
    def _supervised(self) -> bool:
        """Indicates whether or not the estimator is supervised or not.

        This is useful internally for determining if an estimator expects to be provided with a `y`
        value in it's `learn_one` method. For instance we use this in a pipeline to know whether or
        not we should pass `y` to an estimator or not.

        """
        return False

    def _compute_distance(self, X: pd.DataFrame, X_t: pd.DataFrame) -> float:
        """Compute the distance between the data matrix and its transformation.

        This formulation computes a measure of how much information in the dataset represented by Y is preserved or
        retained when projected onto the space spanned by W. The difference between the covariance matrix of Y and the
        projected version is computed, and the sum of all elements in this difference matrix gives an overall measure
        of dissimilarity or distortion.

        Args:
            X: data matrix
            X_t: Transformed data matrix

        Returns:
            Distance between the data matrix and its transformation.

        """
        # Default metric; see the dist_* functions above for the alternatives
        # explored during development (issue #12).
        return dist_featurewise_l1(X.to_numpy(), X_t.to_numpy())

    def _reset_score(self) -> None:
        self._score = None
        self._distances = None
        self._drift_detected = None

    def _transform_many(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.subid, MiniBatchTransformer) or (
            not isinstance(self.subid, Transformer)
            and hasattr(self.subid, "transform_many")
        ):
            X_p = pd.DataFrame(self.subid.transform_many(X))
        else:
            X_p = pd.DataFrame(
                [
                    self.subid.transform_one(x)
                    for x in X.to_dict(orient="records")
                ],
            )
        return X_p

    def _predict_many(self, X: pd.DataFrame) -> pd.DataFrame:
        r"""Control-aware reconstruction: one-step prediction with the learned B.

        Returns a frame the same shape as ``X`` whose row ``k`` is the model's
        one-step prediction :math:`\hat{x}_k = A x_{k-1} + B u_{k-1}` (the first
        row is copied from ``X``, having no predecessor). Falls back to the
        input-blind projection reconstruction when the model exposes no control
        matrix, no inputs are buffered, or the dimensions do not line up.
        """
        model = (
            self.subid.obj if isinstance(self.subid, Rolling) else self.subid
        )
        # getattr (not direct access) keeps this tolerant of models without a
        # control matrix and avoids reaching into a typed private attribute.
        reconstruct_ab = getattr(model, "_reconstruct_AB", None)
        us = [u for u in self._U if u is not None]
        xs = X.to_numpy()
        # Need a control matrix, buffered inputs, and at least one transition.
        if reconstruct_ab is None or len(us) < len(xs) - 1 or len(xs) < 2:  # noqa: PLR2004
            return self._transform_many(X)
        try:
            A, B = reconstruct_ab()  # original-space (n,n), (n,m)
        except AttributeError, ValueError:
            return self._transform_many(X)
        u_arr = np.array([list(u.values()) for u in us], dtype=float)
        # Right-align inputs with the predictor rows X[:-1]: the most recent
        # buffered input drives the most recent transition, regardless of how
        # the bounded buffers have been clipped or temporarily extended.
        u_al = u_arr[-(len(xs) - 1) :]
        if A.shape[0] != xs.shape[1] or B.shape[0] != xs.shape[1]:
            return self._transform_many(X)
        pred = (xs[:-1] @ A.T + u_al @ B.T).real
        X_p = xs.copy().astype(float)
        X_p[1:] = pred
        return pd.DataFrame(X_p, index=X.index, columns=X.columns)

    def learn_one(self, x: dict, **params: Any) -> None:  # noqa: ANN401
        """Allias for update method for interoperability with Pipeline."""
        self.update(x, **params)

    @property
    def learn_delay(self) -> int:
        """Return the effective number of samples between observation and learning.

        Returns:
            Number of samples that must be buffered before a new observation is
            used to update the subspace model.

        """
        if self.start_soon:
            delay = max(0, len(self._X) - self._learn_delay)
        else:
            delay = self._learn_delay
        return delay

    def learn_many(self, X: pd.DataFrame, **params: Any) -> None:  # noqa: ANN401
        """Update the subspace model with a mini-batch of observations.

        Args:
            X: DataFrame of new observations, one row per time step.
            **params: Extra keyword arguments forwarded to the underlying model.

        """
        n = len(X)
        # If buffer is too small, learn in chunks
        buffer_len = self.ref_size + self.lag + self.test_size
        if n > buffer_len:
            for X_part in [
                X[i : i + buffer_len] for i in range(0, X.shape[0], buffer_len)
            ]:
                self.learn_many(X_part, **params)
            return
        # This would discard samples beyond window size, but we make chunks
        self._X.extend(X.to_dict(orient="records"))

        X_ = pd.DataFrame(self._X)

        # Learn the model if data past the time lag and test size is availabe
        # If learn_after_grace is False learn only when grace period is not yet over
        cond_grace = self.learn_after_grace or self.n_seen < self.grace_period
        cond_soon = len(self._X) >= self.learn_delay + n
        if cond_soon and cond_grace:
            idx_start = -self.learn_delay - n
            idx_end = -self.learn_delay if self.learn_delay > 0 else None
            if isinstance(self.subid, Rolling):
                self.subid.update_many(
                    X_.iloc[idx_start:idx_end],
                    **params,
                )
            elif isinstance(self.subid, MiniBatchTransformer):
                self.subid.learn_many(
                    X_.iloc[idx_start:idx_end],
                    **params,
                )
            else:
                for x in X_.iloc[idx_start:idx_end].to_dict(orient="records"):
                    self.subid.learn_one(x, **params)
        self.n_seen += n

    def predict_one(self) -> bool | None:
        """Return the cached drift detection result without consuming a new sample.

        Returns:
            True if drift was detected on the last scored sample, False if not,
            or None when no score has been computed yet.

        """
        return self._drift_detected

    def score_one(self, x: dict) -> float:
        """Score a single observation without permanently updating the buffer.

        Args:
            x: Feature dictionary for the new observation.

        Returns:
            Anomaly score; values above ``threshold`` indicate a change-point.

        """
        # Temporarily add the new sample to the buffer
        self._X.append(x)

        self._reset_score()
        score = self.score

        # Preserve stateless behavior
        self._X.pop()
        return score

    def update(self, x: dict, **params: Any) -> None:  # noqa: ANN401
        """Append one observation and update the subspace model when ready.

        Args:
            x: Feature dictionary for the new observation.
            **params: Extra keyword arguments forwarded to the underlying model.

        """
        self._X.append(x)
        if self.control_aware:
            self._U.append(params.get("u"))
        # Learn the model if data past the time lag and test size is availabe
        # If learn_after_grace is False learn only when grace period is not yet over
        cond_grace = self.learn_after_grace or self.n_seen < self.grace_period
        cond_soon = len(self._X) > self.learn_delay
        if cond_soon and cond_grace:
            idx = -self.learn_delay - 1
            if isinstance(self.subid, _RollingTypes):
                self.subid.update(self._X[idx], **params)
            else:
                self.subid.learn_one(self._X[idx], **params)
        self.n_seen += 1


class DMDChangeDetector(SubIDChangeDetector):
    """Change-Point Detection on Subspace Identification with Online DMD.

    This class implements is optimized for the OnlineDMD and OnlineDMDwC classes,
    where computation of eigenvalues during transformation creates a bottleneck.
    It stores transformed data and only recomputes the transformation when the
    Koopman operator changes.

    The computation time is approx. 20% lower for the OnlineDMD (80 features).
    This has however, impact on the overall performance and adds positive trand
    in D_train - D_test score.

    Args:
        SubIDChangeDetector (_type_): _description_

    """

    def __init__(
        self,
        subid: OnlineDMD
        | OnlineDMDwC
        | Rolling
        | RustRollingDMD
        | RustRollingDMDwC,
        ref_size: int,
        test_size: int | None = None,
        threshold: float = 0.25,
        lag: int = 0,
        grace_period: int = 0,
        *,
        learn_after_grace: bool = True,
    ) -> None:
        """Initialize DMDChangeDetector with subspace model and detection parameters."""
        super().__init__(
            subid=subid,
            ref_size=ref_size,
            test_size=test_size,
            threshold=threshold,
            lag=lag,
            grace_period=grace_period,
            learn_after_grace=learn_after_grace,
        )
        self.subid = subid
        self._Xp: deque[dict] = deque(
            maxlen=self.ref_size + self.lag + self.test_size,
        )

    def _transform_many(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.subid, Rolling):
            subid_ = cast("Transformer", self.subid.obj)
        else:
            subid_ = self.subid
        if hasattr(subid_, "A_allclose") and subid_.A_allclose:
            self._Xp.append(dict(subid_.transform_one(X.iloc[-1].to_dict())))
        else:
            X_p = super()._transform_many(X)
            self._Xp.extend(X_p.to_dict(orient="records"))

        return pd.DataFrame(self._Xp)
