"""Adapter + shared evaluation utilities for the CPDMD baseline comparison.

Compares ODMD-CPD (this repo) against the external CPDMD baseline of V. Khamesi,
"Online Changepoint Detection in Multivariate Seasonal Time Series via Dynamic
Mode Decomposition", https://github.com/vkhamesi/cpdmd-python

The baseline's source is used unmodified. Notebooks clone it on demand into
``examples/.baselines/cpdmd-python`` (gitignored); :func:`ensure_cpdmd` performs
the clone and puts it on ``sys.path``. This module only wraps the baseline so it
consumes the same data layout and is scored by the same metric as ODMD-CPD.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

CPDMD_URL = "https://github.com/vkhamesi/cpdmd-python"
CPDMD_REV = "main"
DEFAULT_DIR = Path(__file__).resolve().parent / ".baselines" / "cpdmd-python"

# Ensure the repo root is importable (for `reshift.*`) regardless of the
# notebook's cwd or sys.path -- this module lives in examples/.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def ensure_cpdmd(dest: Path = DEFAULT_DIR, rev: str = CPDMD_REV) -> Path:
    """Clone the CPDMD baseline on demand and add it to ``sys.path``.

    Args:
        dest: Target directory for the clone (gitignored).
        rev: Branch or commit to check out.

    Returns:
        Path to the baseline checkout.
    """
    if not (dest / "cpdmd.py").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                rev,
                CPDMD_URL,
                str(dest),
            ],
            check=True,
        )
    if str(dest) not in sys.path:
        sys.path.append(str(dest))
    return dest


# --- baseline adapter -----------------------------------------------------


def run_cpdmd(
    X: np.ndarray,
    burn_in: int,
    grid: dict[str, list[int]],
    gamma: float = 0.1,
    ell: float = 3.0,
) -> list[int]:
    """Run the CPDMD baseline and return detected change-point indices.

    Args:
        X: Data array of shape ``(n_samples, n_features)`` (same layout as the
            ODMD-CPD streaming loop). Transposed internally to the baseline's
            ``(n_features, n_samples)`` convention.
        burn_in: Initial samples used for hyperparameter selection / warm-up.
        grid: Grid-search ranges with keys ``window``, ``order``, ``rank``.
        gamma: Adaptive-EWMA learning rate (``lam`` in the baseline).
        ell: Adaptive-EWMA control limit (``L`` in the baseline).

    Returns:
        Sorted list of detected change-point indices on the same announcement
        convention as :func:`run_odmd_cpd` (the sample at which the alarm is
        raised). The baseline labels detections at ``t-1`` -- the trailing edge
        of the window -- one sample behind the iteration that actually fires;
        we add 1 so both methods report the announcement sample, removing the
        baseline's 1-sample labelling head start.
    """
    # Imported dynamically: the baseline is cloned on demand by ensure_cpdmd().
    multiple_changes = importlib.import_module("cpdmd").multiple_changes
    grid_search = importlib.import_module("optimisation").grid_search

    def optimiser(objective: Callable, data: np.ndarray) -> dict:
        return grid_search(objective, data, grid)

    cps = multiple_changes(
        X.T,
        burn_in=burn_in,
        gamma=gamma,
        ell=ell,
        optimiser=optimiser,
    )
    return sorted(int(c) + 1 for c in cps)


def cpdmd_error_series(
    X: np.ndarray,
    window: int,
    order: int,
    rank: int,
) -> np.ndarray:
    """Per-step normalized reconstruction error of the baseline's windowed DMD.

    Provides a score curve comparable to ODMD-CPD's for plotting. Uses the
    baseline's own ``dmd``/``hankel``/``unroll`` unmodified.

    Args:
        X: Data array of shape ``(n_samples, n_features)``.
        window: Sliding-window size.
        order: Time-delay (Hankel) order.
        rank: SVD truncation rank.

    Returns:
        Array of length ``n_samples``; entries before the first full window are 0.
    """
    # Imported dynamically: the baseline is cloned on demand by ensure_cpdmd().
    _dmd = importlib.import_module("dmd")
    dmd, hankel, unroll = _dmd.dmd, _dmd.hankel, _dmd.unroll

    data = X.T
    num_features, n = data.shape
    err = np.zeros(n)
    for t in range(window, n):
        Xw = data[:, t - window : t]
        H = hankel(Xw, order)
        modes, dyn, amp = dmd(H, rank)
        Xhat = unroll((modes @ amp @ dyn).real, order)
        err[t] = np.linalg.norm(Xw - Xhat) ** 2 / (num_features * window)
    return err


# --- shared evaluation ----------------------------------------------------


def peaks(scores: np.ndarray, threshold: float, refractory: int) -> list[int]:
    """Collapse a score curve into discrete detections.

    The first index whose score exceeds ``threshold`` triggers a detection;
    further triggers within ``refractory`` samples are suppressed (one detection
    per change region).

    Args:
        scores: Per-step detection score.
        threshold: Detection threshold.
        refractory: Minimum gap (samples) between successive detections.

    Returns:
        Sorted list of detection indices.
    """
    out: list[int] = []
    last = -(10**9)
    for i in np.flatnonzero(scores > threshold):
        if i - last > refractory:
            out.append(int(i))
            last = int(i)
    return out


def match_changepoints(
    true_cps: list[int],
    pred_cps: list[int],
    tol: int,
    pre: int = 0,
) -> dict[str, float]:
    """Greedily match predictions to ground truth in a causal forward window.

    A prediction ``p`` is a true positive for change ``t`` only if it lands in
    ``[t - pre, t + tol]`` -- i.e. at the change or after it (a causal detector
    cannot fire before the change manifests). ``pre`` allows a few samples of
    grace for windowing/indexing jitter; with ``pre=0`` any pre-change detection
    is a false positive, so reported delays are non-negative and measured on
    true positives only.

    Args:
        true_cps: Ground-truth change-point indices.
        pred_cps: Detected change-point indices.
        tol: Forward detection window (max admissible delay, in samples).
        pre: Samples of pre-change grace for jitter (default 0).

    Returns:
        Dict with tp, fp, fn, precision, recall, f1, mean_delay (mean over true
        positives, ``>= -pre``; NaN if none).
    """
    remaining = list(true_cps)
    delays: list[int] = []
    tp = 0
    for p in sorted(pred_cps):
        best = min(
            (t for t in remaining if -pre <= p - t <= tol),
            key=lambda t: (
                p - t
            ),  # earliest valid (smallest non-negative delay)
            default=None,
        )
        if best is not None:
            tp += 1
            delays.append(p - best)
            remaining.remove(best)
    fp = len(pred_cps) - tp
    fn = len(remaining)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
    }


# --- synthetic benchmarks -------------------------------------------------


def gen_steps(
    n: int,
    changes: list[float],
    sigma: float,
    seed: int | None,
    coupling: float = 0.7,
) -> tuple[np.ndarray, list[int]]:
    """Bivariate piecewise-constant signal with evenly spaced level steps.

    Channel 2 is a scaled copy of channel 1 (correlated, low-rank). Matches the
    generator used in ``01_synthetic_steps.ipynb``.

    Args:
        n: Number of samples.
        changes: Per-change-point level increments applied in order.
        sigma: Gaussian noise std.
        seed: RNG seed.
        coupling: Channel-2 scaling of channel-1 increments.

    Returns:
        ``(X, cps)`` with ``X`` of shape ``(n, 2)`` and the true change indices.
    """

    def one(
        deltas: list[float],
        sd: int | None,
    ) -> tuple[np.ndarray, list[int]]:
        rng = np.random.default_rng(sd)
        x = np.zeros(n)
        interval = n // (len(deltas) + 1)
        cps = np.arange(interval, n, interval)[: len(deltas)]
        level = 0.0
        for i in range(n):
            hit = np.where(cps == i)[0]
            if len(hit):
                level += deltas[hit[0]]
            x[i] = level + rng.normal(0, sigma)
        return x, [int(c) for c in cps]

    c1, cps = one(changes, seed)
    c2, _ = one(
        [c * coupling for c in changes],
        None if seed is None else seed + 1,
    )
    return np.vstack([c1, c2]).T, cps


def gen_seasonal(
    n: int,
    freqs: list[float],
    amps: list[float],
    sigma: float,
    seed: int | None,
) -> tuple[np.ndarray, list[int]]:
    """Bivariate seasonal signal with per-segment frequency and amplitude.

    Change points are the segment boundaries; both the oscillation frequency and
    amplitude switch at each one (amplitude shifts give the reconstruction-error
    transient that ODMD-CPD's ratio statistic responds to).

    Args:
        n: Number of samples.
        freqs: Per-segment frequencies (cycles/sample).
        amps: Per-segment amplitudes (same length as ``freqs``).
        sigma: Gaussian noise std.
        seed: RNG seed.

    Returns:
        ``(X, cps)`` with ``X`` of shape ``(n, 2)`` and the true change indices.
    """
    rng = np.random.default_rng(seed)
    seg = n // len(freqs)
    cps = [seg * i for i in range(1, len(freqs))]
    f = np.zeros(n)
    a = np.zeros(n)
    for i, (fr, am) in enumerate(zip(freqs, amps, strict=False)):
        f[i * seg : (i + 1) * seg] = fr
        a[i * seg : (i + 1) * seg] = am
    phase = np.cumsum(2 * np.pi * f)
    c1 = a * np.sin(phase) + rng.normal(0, sigma, n)
    c2 = (
        a * np.sin(phase + 0.5)
        + 0.5 * a * np.cos(2 * phase)
        + rng.normal(0, sigma, n)
    )
    return np.vstack([c1, c2]).T, cps


# --- ODMD-CPD runner ------------------------------------------------------


def run_odmd_cpd(
    X: np.ndarray,
    window: int,
    hankel_n: int,
    ref_size: int,
    test_size: int,
    rank: int,
    threshold: float,
    refractory: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Stream X through the ODMD-CPD detector and return scores + detections.

    Uses the fixed-threshold detector as published: a detection fires when the
    score exceeds ``threshold``; triggers within ``refractory`` samples collapse
    into one change point.

    Args:
        X: Data array ``(n_samples, n_features)``.
        window: OnlineDMD rolling window.
        hankel_n: Hankel (time-delay) embedding order.
        ref_size: Reference (base) window size.
        test_size: Test window size.
        rank: DMD truncation rank.
        threshold: Detection threshold on the score.
        refractory: Suppression gap between detections (defaults to ref+test).

    Returns:
        ``(scores, cps)`` with per-step score array and detected change indices.
    """
    from river.decomposition import OnlineDMD
    from river.preprocessing import Hankelizer

    from reshift.chdsubid import SubIDChangeDetector
    from reshift.rolling import Rolling

    odmd = Rolling(
        OnlineDMD(
            r=rank,
            initialize=window,
            w=1.0,
            exponential_weighting=False,
            seed=42,
        ),
        window + 1,
    )
    detector = SubIDChangeDetector(
        odmd,
        ref_size=ref_size,
        test_size=test_size,
        threshold=threshold,
        grace_period=window + test_size + 1,
    )
    pipe = Hankelizer(hankel_n) | detector
    n = X.shape[0]
    scores = np.zeros(n)
    for i, x in enumerate(pd.DataFrame(X).to_dict(orient="records")):
        scores[i] = pipe.score_one(x)
        pipe.learn_one(x)
    refr = refractory if refractory is not None else ref_size + test_size
    return scores, peaks(scores, threshold, refr)


# --- reporting helpers ----------------------------------------------------


def metrics_table(
    true_cps: list[int],
    ours: list[int],
    base: list[int],
    tol: int,
) -> pd.DataFrame:
    """Tabulate match metrics for both methods on one benchmark.

    Args:
        true_cps: Ground-truth change indices.
        ours: ODMD-CPD detections.
        base: CPDMD detections.
        tol: Matching tolerance in samples.

    Returns:
        DataFrame indexed by method with precision/recall/F1/delay and counts.
    """
    rows: dict[str, dict[str, float]] = {}
    for name, pred in [("ODMD-CPD", ours), ("CPDMD", base)]:
        m = match_changepoints(true_cps, pred, tol)
        rows[name] = {
            "precision": m["precision"],
            "recall": m["recall"],
            "F1": m["f1"],
            "mean_delay": m["mean_delay"],
            "TP": m["tp"],
            "FP": m["fp"],
            "FN": m["fn"],
        }
    return pd.DataFrame(rows).T.round(3)


def _detection_legend() -> list[Line2D]:
    return [
        Line2D([], [], color="k", ls="--", label="true CP"),
        Line2D([], [], color="C0", label="ODMD-CPD detection"),
        Line2D([], [], color="C3", ls="--", label="CPDMD detection"),
    ]


def plot_comparison(
    X: np.ndarray,
    scores: np.ndarray,
    true_cps: list[int],
    ours: list[int],
    base: list[int],
    threshold: float,
    title: str,
) -> Figure:
    """Plot the signal and ODMD-CPD score with both methods' detections.

    Args:
        X: Data array ``(n_samples, n_features)``.
        scores: ODMD-CPD per-step score.
        true_cps: Ground-truth change indices (dashed verticals on the signal).
        ours: ODMD-CPD detections (solid blue verticals).
        base: CPDMD detections (dashed red verticals).
        threshold: ODMD-CPD detection threshold (dotted line on the score axis).
        title: Plot title.

    Returns:
        The created figure.
    """
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(11, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axs[0].plot(X, lw=0.7)
    axs[0].set_ylabel("signal")
    axs[0].set_title(title)
    for c in true_cps:
        axs[0].axvline(c, color="k", ls="--", lw=0.8, alpha=0.6)
    axs[1].plot(scores, color="C0", lw=0.8, label="ODMD-CPD score")
    axs[1].axhline(threshold, color="C0", ls=":", lw=0.8)
    for c in ours:
        axs[1].axvline(c, color="C0", lw=1.4)
    for c in base:
        axs[1].axvline(c, color="C3", lw=1.4, ls="--")
    axs[1].set_ylabel("score")
    axs[1].set_xlabel("sample")
    axs[1].legend(handles=_detection_legend(), loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_bess(
    X: np.ndarray,
    scores: np.ndarray,
    fault: np.ndarray,
    ours: list[int],
    base: list[int],
    threshold: float,
) -> Figure:
    """Plot BESS temperatures and score with the fault region shaded.

    Args:
        X: Temperature array ``(n_samples, n_features)``.
        scores: ODMD-CPD per-step score.
        fault: Binary fault-active label per sample.
        ours: ODMD-CPD detections.
        base: CPDMD detections.
        threshold: ODMD-CPD detection threshold.

    Returns:
        The created figure.
    """
    n = X.shape[0]
    fig, axs = plt.subplots(
        2,
        1,
        figsize=(11, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axs[0].plot(X, lw=0.6)
    axs[0].fill_between(
        range(n),
        X.min(),
        X.max(),
        where=fault > 0,
        color="grey",
        alpha=0.2,
        label="fan-fault active",
    )
    axs[0].set_title("BESS module temperatures")
    axs[0].set_ylabel("temp (norm.)")
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].plot(scores, color="C0", lw=0.8)
    axs[1].axhline(threshold, color="C0", ls=":", lw=0.8)
    for c in ours:
        axs[1].axvline(c, color="C0", lw=1.4)
    for c in base:
        axs[1].axvline(c, color="C3", lw=1.4, ls="--")
    axs[1].fill_between(
        range(n),
        0,
        scores.max(),
        where=fault > 0,
        color="grey",
        alpha=0.2,
    )
    axs[1].set_ylabel("score")
    axs[1].set_xlabel("sample (resampled)")
    axs[1].legend(
        handles=_detection_legend()[1:],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    return fig


# --- analyses (notebook appendices) ---------------------------------------


def robustness(
    gen_fn: Callable[[int], tuple[np.ndarray, list[int]]],
    odmd_kw: dict[str, int],
    threshold: float,
    tol: int,
    n_cp: int,
    grid: dict[str, list[int]],
    seeds: range = range(1, 7),
    burn_in: int = 700,
) -> pd.DataFrame:
    """Multi-seed mean +/- std for ODMD-CPD, CPDMD, and a random detector.

    Re-runs the full benchmark over several noise seeds (same fixed parameters
    every run) to show whether the single-seed numbers are stable. The random
    detector (``n_cp`` uniform draws past ``burn_in``) calibrates the metric's
    chance floor for the given tolerance.

    Args:
        gen_fn: ``seed -> (X, true_cps)`` benchmark generator.
        odmd_kw: Keyword args for :func:`run_odmd_cpd` (without ``threshold``).
        threshold: ODMD-CPD detection threshold.
        tol: Forward matching tolerance (samples).
        n_cp: Number of random detections to draw for the chance baseline.
        grid: CPDMD grid (use single-value lists for fixed, fast, fair params).
        seeds: Seeds to average over.
        burn_in: CPDMD burn-in / lower bound for random draws.

    Returns:
        DataFrame indexed by method with mean and std of precision, recall, F1,
        and mean detection delay.
    """
    acc: dict[str, list[dict[str, float]]] = {
        "ODMD-CPD": [],
        "CPDMD": [],
        "random": [],
    }
    for seed in seeds:
        X, cps = gen_fn(seed)
        _, ours = run_odmd_cpd(X, threshold=threshold, **odmd_kw)
        base = run_cpdmd(X, burn_in=burn_in, grid=grid)
        rng = np.random.default_rng(seed + 99)
        rnd = sorted(rng.integers(burn_in, len(X), n_cp).tolist())
        acc["ODMD-CPD"].append(match_changepoints(cps, ours, tol))
        acc["CPDMD"].append(match_changepoints(cps, base, tol))
        acc["random"].append(match_changepoints(cps, rnd, tol))

    def stat(ms: list[dict[str, float]], key: str) -> tuple[float, float]:
        vals = [
            m[key]
            for m in ms
            if not (key == "mean_delay" and np.isnan(m[key]))
        ]
        return (
            (float(np.mean(vals)), float(np.std(vals)))
            if vals
            else (
                float("nan"),
                0.0,
            )
        )

    rows: dict[str, dict[str, float]] = {}
    for name, ms in acc.items():
        f1_m, f1_s = stat(ms, "f1")
        p_m, _ = stat(ms, "precision")
        r_m, r_s = stat(ms, "recall")
        d_m, _ = stat(ms, "mean_delay")
        rows[name] = {
            "F1": f1_m,
            "F1_std": f1_s,
            "precision": p_m,
            "recall": r_m,
            "recall_std": r_s,
            "mean_delay": d_m,
        }
    return pd.DataFrame(rows).T.round(3)


def latency(
    X: np.ndarray,
    odmd_kw: dict[str, int],
    grid: dict[str, list[int]],
    threshold: float = 0.5,
    burn_in: int = 700,
) -> pd.DataFrame:
    """Per-sample wall-clock cost of ODMD-CPD vs the CPDMD baseline.

    Args:
        X: Data array ``(n_samples, n_features)``.
        odmd_kw: Keyword args for :func:`run_odmd_cpd` (without ``threshold``).
        grid: CPDMD grid (single-value lists for a fixed-param timing).
        threshold: ODMD-CPD threshold (irrelevant to timing).
        burn_in: CPDMD burn-in.

    Returns:
        DataFrame indexed by method with total seconds, ms per sample, and Hz.
    """
    n = X.shape[0]
    t0 = time.perf_counter()
    run_odmd_cpd(X, threshold=threshold, **odmd_kw)
    t_odmd = time.perf_counter() - t0
    t1 = time.perf_counter()
    run_cpdmd(X, burn_in=burn_in, grid=grid)
    t_cpdmd = time.perf_counter() - t1
    rows = {
        "ODMD-CPD": {
            "total_s": t_odmd,
            "ms_per_sample": 1e3 * t_odmd / n,
            "Hz": n / t_odmd,
        },
        "CPDMD": {
            "total_s": t_cpdmd,
            "ms_per_sample": 1e3 * t_cpdmd / n,
            "Hz": n / t_cpdmd,
        },
    }
    return pd.DataFrame(rows).T.round(3)


def causality_test(
    tau: int = 1000,
    n: int = 1400,
    window: int = 200,
    order: int = 10,
    rank: int = 2,
    burn_in: int = 700,
    sigma: float = 0.1,
    jumps: tuple[float, ...] = (3.0, 1.0, 0.5, 0.2),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Demonstrate the CPDMD baseline is causal (no look-ahead) for delay 0.

    Runs the baseline's ``single_change`` on data progressively truncated around
    ``tau``: if it detects the change only once the array extends past ``tau``
    (and the firing window reads no sample beyond ``tau``), the detection is
    causal, not written a posteriori. A second sweep over jump magnitudes shows
    the delay grows as the change becomes subtler -- i.e. delay reflects
    accumulated evidence, and delay 0 only occurs for abrupt, large jumps.

    Args:
        tau: True change index.
        n: Sequence length for the jump sweep.
        window: CPDMD window.
        order: CPDMD time-delay order.
        rank: CPDMD rank.
        burn_in: CPDMD burn-in (must be < tau to allow detection).
        sigma: Noise std for the truncation sweep.
        jumps: Jump magnitudes (as absolute level shifts) for the delay sweep.
        seed: RNG seed.

    Returns:
        ``(truncation_df, jump_df)``. ``truncation_df`` maps the highest data
        index available to the detected index; ``jump_df`` maps jump size (in
        sigmas, using the jump sweep's own noise) to detected index and delay.
    """
    single_change = importlib.import_module("cpdmd").single_change
    rng = np.random.default_rng(seed)
    sig = np.concatenate(
        [rng.normal(0, sigma, tau), rng.normal(3, sigma, n - tau)],
    )
    data = np.vstack([sig, 0.7 * sig])
    trows = []
    for end in (tau - 1, tau, tau + 1, tau + 2, tau + 5, tau + 50, n):
        det = single_change(
            data[:, :end],
            burn_in=burn_in,
            window=window,
            order=order,
            rank=rank,
            gamma=0.1,
            ell=3.0,
        )
        trows.append({"data_through_index": end - 1, "detected": det})
    trunc_df = pd.DataFrame(trows)

    jrows = []
    jsigma = 0.3
    for jump in jumps:
        s = np.concatenate(
            [rng.normal(0, jsigma, tau), rng.normal(jump, jsigma, n - tau)],
        )
        det = single_change(
            np.vstack([s, 0.7 * s]),
            burn_in=burn_in,
            window=window,
            order=order,
            rank=rank,
            gamma=0.1,
            ell=3.0,
        )
        jrows.append(
            {
                "jump_sigmas": round(jump / jsigma, 1),
                "detected": det,
                "delay": None if det is None else det - tau,
            },
        )
    return trunc_df, pd.DataFrame(jrows)


def guideline_params(
    X: np.ndarray,
    window_size: int,
    rank_from: int = 1000,
) -> dict[str, int]:
    """Derive ODMD-CPD parameters via the paper's systematic selection.

    Wraps ``reshift.chdsubid.get_default_params`` /
    ``get_default_timedelays`` (sec:systematic-parameters): ``h = window/2``
    capped at 100 delays, ``ref = test = window``, ``lag = 0``, rank by
    Gavish-Donoho.

    Args:
        X: Data array ``(n_samples, n_features)``.
        window_size: Base/structural-change window.
        rank_from: Number of leading samples used to estimate the rank.

    Returns:
        Keyword dict ready to splat into :func:`run_odmd_cpd` (minus threshold).
    """
    from reshift.chdsubid import (
        get_default_params,
        get_default_timedelays,
    )

    win, ref, test, _lag, rank = get_default_params(
        X[:rank_from],
        window_size=window_size,
    )
    hn, _step = get_default_timedelays(window_size // 2, 100)
    return {
        "window": win,
        "hankel_n": hn,
        "ref_size": ref,
        "test_size": test,
        "rank": rank,
    }


if __name__ == "__main__":
    # ponytail: smoke check on a single synthetic step, asserts both the
    # adapter and the matcher run end-to-end. Not the notebook's full benchmark.
    ensure_cpdmd()
    rng = np.random.default_rng(0)
    n = 2000
    sig = np.concatenate([rng.normal(0, 0.3, 1000), rng.normal(3, 0.3, 1000)])
    X = np.vstack([sig, 0.7 * sig]).T
    cps = run_cpdmd(
        X,
        burn_in=400,
        grid={"window": [100, 200], "order": [5, 10], "rank": [2]},
    )
    m = match_changepoints([1000], cps, tol=150)
    print("detected", cps, "metrics", m)
    assert m["tp"] >= 1, "baseline failed to detect the single obvious step"
    print("ok")
