"""M2 quantitative CPD benchmark: toDMDc (OnlineDMD) vs OSVD-CPD (OnlineSVD).

Builds the peer-review "M2" comparison table for the ECC paper.

For two experiments (synthetic step-change signal and the nonlinear two-tank
delay-control system) it runs Monte-Carlo repetitions and, for both the
toDMDc detector (OnlineDMD-based) and the OnlineSVD reference baseline
(OSVD-CPD), computes precision / recall / F1, mean detection delay, and
false-alarm rate against the known ground-truth change-points, using a
tolerance window equal to the deterministic delay budget c = test_size.

The proposed and reference detectors share the *same* SubIDChangeDetector
wrapper (so the only difference is the subspace model: OnlineDMD vs OnlineSVD),
exactly as in examples/01_synthetic_steps.ipynb. The M4 reconstruction-projector
fix in reshift/chdsubid.py is reflected automatically (normal import).

PINNED PARAMETERS
-----------------
Synthetic (canonical, matching examples/config.py MODELS dict):
    Hankelizer(80) | SubIDChangeDetector(
        Rolling(OnlineDMD(r=2, initialize=300, w=1.0,
                          exponential_weighting=False), 301),
        ref_size=100, test_size=100, grace_period=300+100+1=401,
        threshold=0.25)
    OSVD-CPD mirrors this with Rolling(OnlineSVD(n_components=2,
        initialize=300), 301) inside the identical detector wrapper.
    Signal: the two-column step signal of 01_synthetic_steps.ipynb
        (columns [1,-1,2,-2,3,-3,4,-4,5] and [1]*9), n=10000, sigma=0.1.
    Ground-truth change-points: every 1000 samples -> [1000..9000].

Nonlinear two-tank (control case, mirrors 08_nonlinear_delay_control.ipynb):
    PolynomialExtender(2) | Hankelizer(hm=15) | SubIDChangeDetector(
        Rolling(OnlineDMDwC(p=2, q=1, initialize=1999, w=1.0,
                            exponential_weighting=False, eig_rtol=None), 2000),
        ref_size=200, test_size=200, grace_period=2000, start_soon=True,
        threshold=0.25)
    OSVD-CPD: PolynomialExtender(2) | Hankelizer(hm+hl=45) |
        SubIDChangeDetector(OnlineSVD(p+q=3, initialize=2000,
        force_orth=False), ref_size=200, test_size=200,
        grace_period=2000+200+1=2201, threshold=0.25)
    Ground-truth change-points: [3998, 4998, 7598, 8598, 9798].

Run with:
  cd /Users/mw/pyprojects/phd/researcher/odmd-ecc-fixes && \
  PYTHONPATH=$PWD \
  /Users/mw/pyprojects/phd/researcher/odmd-subid-cpd/.venv/bin/python \
  benchmarks/m2_cpd_metrics.py
(the odmd-subid-cpd venv interpreter resolves vendored river; run from the
examples dir is not required because data paths below are absolute).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from river.decomposition import OnlineDMD, OnlineDMDwC, OnlineSVD
from river.feature_extraction import PolynomialExtender
from river.preprocessing import Hankelizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from reshift.chdsubid import (  # noqa: E402
    SubIDChangeDetector,
    get_default_rank,
    get_default_timedelays,
    hankel,
)
from reshift.rolling import Rolling  # noqa: E402

EXAMPLES_DIR = ROOT / "examples"

# -- Benchmark configuration ------------------------------------------------
N_RUNS = 50
MASTER_SEED = 42
THRESHOLD = 0.25  # config.py / SubIDChangeDetector default


# ===========================================================================
# Data generation (copied from examples/01_synthetic_steps.ipynb)
# ===========================================================================
SYN_DATA_DIR = EXAMPLES_DIR / "data/synthetic-steps"
SYN_N_FILES = 5  # published realizations y0.txt .. y4.txt


def load_real_synthetic(idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the published synthetic realization ``y{idx}.txt`` (no generation).

    These are the exact files the paper's synthetic experiment uses (cell 7 of
    01_synthetic_steps.ipynb loads ``y0.txt``). Each is a single-channel signal
    of 10000 samples with 9 step changes every 1000 samples. Returns
    ``(x[n,1], change_points)``.
    """
    y = np.genfromtxt(SYN_DATA_DIR / f"y{idx}.txt")
    n = len(y)
    interval = n // 10  # 9 changes -> 10 segments of 1000
    change_points = np.arange(interval, n, interval)[:9]
    return y.reshape(-1, 1), change_points


# ===========================================================================
# Detection -> metrics
# ===========================================================================
def detections_from_drift(drift_flags: np.ndarray) -> np.ndarray:
    """Convert per-sample boolean drift flags into detection onset indices.

    A detection event is the rising edge of the drift flag (a contiguous block
    of drift==True yields a single detection at its first sample).
    """
    drift = drift_flags.astype(bool)
    return np.where(drift & ~np.concatenate(([False], drift[:-1])))[0]


def score_detections(
    detections: np.ndarray,
    change_points: np.ndarray,
    tolerance: int,
    n_samples: int,
) -> dict:
    """Match detections to ground-truth change-points within a tolerance window.

    A detection counts as a TP if it falls within ``(cp, cp + tolerance]`` after
    a true change-point. Multiple detections inside one window collapse to a
    single TP. Detections outside every window are FP; a change-point window
    with no detection is a FN.

    False-alarm rate is defined as FP per 1000 change-free (nominal) samples,
    where nominal samples = n_samples - (union of the tolerance windows).
    """
    cps = np.sort(np.asarray(change_points))
    dets = np.sort(np.asarray(detections))

    matched_cp = np.zeros(len(cps), dtype=bool)
    delays: list[int] = []
    fp = 0

    for d in dets:
        cand = np.where((cps < d) & (d <= cps + tolerance))[0]
        if len(cand) == 0:
            fp += 1
            continue
        ci = cand[-1]  # nearest preceding change-point
        if not matched_cp[ci]:
            matched_cp[ci] = True
            delays.append(int(d - cps[ci]))
        # additional detections within an already-matched window are neither a
        # new TP nor an FP -- still "explained" by that change-point.

    tp = int(matched_cp.sum())
    fn = int((~matched_cp).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    nominal = max(n_samples - len(cps) * tolerance, 1)
    far = 1000.0 * fp / nominal  # FP per 1000 nominal samples
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "delays": delays,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "far": far,
    }


def stable_variance(
    scores: np.ndarray,
    change_points: np.ndarray,
    tolerance: int,
    grace: int,
) -> float:
    """Variance of the detection statistic in stable (non-change) regions.

    Stable samples are those past the grace period and outside every
    ``(cp, cp + tolerance]`` window. Returns the variance of the scores there.
    """
    n = len(scores)
    mask = np.ones(n, dtype=bool)
    mask[:grace] = False
    for cp in change_points:
        lo = max(cp, 0)
        hi = min(cp + tolerance + 1, n)
        mask[lo:hi] = False
    vals = scores[mask]
    return float(np.var(vals)) if len(vals) > 1 else float("nan")


# Calibration: number of stable-region std-devs above the stable mean used as
# the per-run detection threshold. Chosen so the stable-region false-alarm
# rate is ~0; reported alongside the results.
CALIB_K = 4.0
MIN_CALIB_SAMPLES = 2  # need >=2 points for std with ddof=1


def calibrated_threshold(
    scores: np.ndarray,
    grace: int,
    first_cp: int,
    k: float = CALIB_K,
) -> float:
    """Per-run threshold = mean + k*std over the first stable calibration window.

    The window is ``[grace, first_cp)`` -- post warm-up, pre first change.
    """
    calib = scores[grace:first_cp]
    calib = calib[np.isfinite(calib)]
    if calib.size < MIN_CALIB_SAMPLES:
        return float("inf")
    return float(calib.mean() + k * calib.std(ddof=1))


def roc_auc(
    scores: np.ndarray,
    change_points: np.ndarray,
    tolerance: int,
    grace: int,
) -> float:
    """Threshold-free AUC of the score as a per-sample change discriminant.

    Positives are samples in any ``(cp, cp + tolerance]`` window; negatives are
    the remaining post-grace samples. Computed via the rank-based
    Mann-Whitney identity (average ranks, so exact score ties are handled).
    """
    n = len(scores)
    label = np.zeros(n, dtype=bool)
    for cp in change_points:
        label[cp + 1 : min(cp + tolerance + 1, n)] = True
    keep = np.zeros(n, dtype=bool)
    keep[grace:] = True
    keep &= np.isfinite(scores)
    s, y = scores[keep], label[keep]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_pos = ranks[y].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def metrics_for_method(
    scores: np.ndarray,
    change_points: np.ndarray,
    tolerance: int,
    grace: int,
    n: int,
) -> dict:
    """Compute calibrated-threshold detection metrics + AUC + stable variance.

    Detections are rising edges of ``score > calibrated_threshold`` past the
    grace period. Returns the score_detections dict augmented with ``auc``,
    ``stable_var``, ``thr`` and ``_tol``.
    """
    first_cp = int(np.min(change_points))
    thr = calibrated_threshold(scores, grace, first_cp)
    flags = (scores > thr) & (np.arange(n) >= grace)
    res = score_detections(
        detections_from_drift(flags),
        change_points,
        tolerance,
        n,
    )
    res["auc"] = roc_auc(scores, change_points, tolerance, grace)
    res["stable_var"] = stable_variance(
        scores,
        change_points,
        tolerance,
        grace,
    )
    res["thr"] = thr
    res["_tol"] = tolerance
    return res


# ===========================================================================
# Experiment 1: synthetic step changes  (PINNED config.py parameters)
# ===========================================================================
SYN_N_SAMPLES = 10000
SYN_HN = 80
SYN_R = 2
SYN_INIT = 300
SYN_REF = 100
SYN_TEST = 100
SYN_ROLL = 301
SYN_GRACE = SYN_INIT + SYN_TEST + 1  # 401


def build_synthetic_pipelines() -> tuple:
    """Build toDMDc and OSVD-CPD pipelines with the pinned config.py params."""
    odmd = Rolling(
        OnlineDMD(
            r=SYN_R,
            initialize=SYN_INIT,
            w=1.0,
            exponential_weighting=False,
            seed=42,
        ),
        SYN_ROLL,
    )
    osvd = Rolling(
        OnlineSVD(n_components=SYN_R, initialize=SYN_INIT, seed=42),
        SYN_ROLL,
    )
    subid_dmd = SubIDChangeDetector(
        odmd,
        ref_size=SYN_REF,
        test_size=SYN_TEST,
        grace_period=SYN_GRACE,
        threshold=THRESHOLD,
    )
    subid_svd = SubIDChangeDetector(
        osvd,
        ref_size=SYN_REF,
        test_size=SYN_TEST,
        grace_period=SYN_GRACE,
        threshold=THRESHOLD,
    )
    pipeline_dmd = Hankelizer(SYN_HN) | subid_dmd
    pipeline_svd = Hankelizer(SYN_HN) | subid_svd
    return pipeline_dmd, subid_dmd, pipeline_svd, subid_svd


def run_synthetic_once(seed: int) -> dict:
    """Run both detectors once on a freshly generated synthetic signal."""
    x, change_points = load_real_synthetic(seed % SYN_N_FILES)
    n = x.shape[0]

    pipeline_dmd, _subid_dmd, pipeline_svd, _subid_svd = (
        build_synthetic_pipelines()
    )

    score_dmd = np.zeros(n, dtype=float)
    score_svd = np.zeros(n, dtype=float)
    for i, xi in enumerate(pd.DataFrame(x).to_dict(orient="records")):
        score_dmd[i] = pipeline_dmd.score_one(xi)
        pipeline_dmd.learn_one(xi)
        score_svd[i] = pipeline_svd.score_one(xi)
        pipeline_svd.learn_one(xi)

    res_dmd = metrics_for_method(
        score_dmd,
        change_points,
        SYN_TEST,
        SYN_GRACE,
        n,
    )
    res_svd = metrics_for_method(
        score_svd,
        change_points,
        SYN_TEST,
        SYN_GRACE,
        n,
    )
    return {"dmd": res_dmd, "svd": res_svd, "tol": SYN_TEST}


# ===========================================================================
# Experiment 2: nonlinear two-tank delay control (mirrors notebook 08)
# ===========================================================================
NL_WINDOW = 2000
NL_REF = 200
NL_TEST = 200


def load_nonlinear_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the deterministic two-tank simulation (states X, inputs U)."""
    train_data = pd.read_pickle(  # noqa: S301  trusted local pickle
        EXAMPLES_DIR / "data/nonlinear-delay-control/train_sim.pkl",
    )
    X = pd.DataFrame(train_data["X"][:12000])
    U = pd.DataFrame(train_data["U"][:12000])
    return X, U


def run_nonlinear_once(seed: int) -> dict:
    """Run both detectors once; vary the additive measurement-noise seed.

    The two-tank dynamics (the pickle) are fully deterministic; the only
    stochastic component in the notebook is the additive measurement noise,
    seeded with ``np.random.seed(42)``. Monte-Carlo therefore varies that seed.
    """
    X_base, U = load_nonlinear_base()
    X = X_base.copy()
    Y = np.zeros(X.shape[0])

    np.random.seed(seed)
    X += np.random.normal(0, 0.35, X.shape)
    X.iloc[3998:4998] += 1.0
    Y[3998:4998] = 1
    X.iloc[7598:8598] *= 2.0
    Y[7598:8598] = 1
    X.iloc[9798:12000] *= np.linspace(
        1.0,
        2.0,
        X.iloc[9798:12000].shape[0],
    )[:, None]
    Y[9798:] = 1

    change_points = np.where(np.abs(np.diff(Y, prepend=0)) == 1)[0]
    n = X.shape[0]

    hm, hm_step = get_default_timedelays(200, 30 // X.shape[1])
    hl, hl_step = get_default_timedelays(30, 30 // U.shape[1])
    p = min(X.shape[1], get_default_rank(hankel(X[:NL_WINDOW], hm, hm_step)))
    q = min(U.shape[1], get_default_rank(hankel(U[:NL_WINDOW], hl, hl_step)))

    U_ = pd.DataFrame(hankel(U, hl, hl_step))
    init_size = NL_WINDOW

    odmd = Rolling(
        OnlineDMDwC(
            p=p,
            q=q,
            initialize=init_size - 1,
            w=1.0,
            exponential_weighting=False,
            eig_rtol=None,
        ),
        init_size,
    )
    subid_dmd = SubIDChangeDetector(
        odmd,
        ref_size=NL_REF,
        test_size=NL_TEST,
        grace_period=init_size,
        start_soon=True,
        threshold=THRESHOLD,
    )
    pipeline_dmd = PolynomialExtender(2) | Hankelizer(hm) | subid_dmd

    osvd = OnlineSVD(p + q, initialize=init_size, force_orth=False)
    subid_svd = SubIDChangeDetector(
        osvd,
        ref_size=NL_REF,
        test_size=NL_TEST,
        grace_period=init_size + NL_TEST + 1,
        threshold=THRESHOLD,
    )
    pipeline_svd = PolynomialExtender(2) | Hankelizer(hm + hl) | subid_svd

    score_dmd = np.zeros(n, dtype=float)
    score_svd = np.zeros(n, dtype=float)
    for i, (x, u) in enumerate(
        zip(
            X.to_dict(orient="records"),
            U_.to_dict(orient="records"),
            strict=False,
        ),
    ):
        score_dmd[i] = pipeline_dmd.score_one(x)
        pipeline_dmd.learn_one(x, u=u)

        # NOTE: the notebook never scores the SVD pipeline (it only calls
        # learn_one on it). We additionally call score_one here to obtain an
        # OSVD-CPD baseline; the pipeline itself is built verbatim. SVD learns
        # on x only (no control), matching the notebook's learn_one(x).
        score_svd[i] = pipeline_svd.score_one(x)
        pipeline_svd.learn_one(x)

    res_dmd = metrics_for_method(
        score_dmd,
        change_points,
        NL_TEST,
        init_size,
        n,
    )
    res_svd = metrics_for_method(
        score_svd,
        change_points,
        NL_TEST,
        init_size + NL_TEST + 1,
        n,
    )
    return {"dmd": res_dmd, "svd": res_svd, "tol": NL_TEST}


# ===========================================================================
# Aggregation
# ===========================================================================
def ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, 95% half-width) using 1.96 * std / sqrt(N)."""
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, 0.0
    half = 1.96 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
    return mean, half


def aggregate(per_run: list[dict]) -> dict:
    """Aggregate per-run metric dicts into mean +- 95% CI summaries."""
    f1_m, f1_ci = ci95([r["f1"] for r in per_run])
    far_m, far_ci = ci95([r["far"] for r in per_run])
    prec_m, _ = ci95([r["precision"] for r in per_run])
    rec_m, _ = ci95([r["recall"] for r in per_run])

    delays_per_run = [
        float(np.mean(r["delays"])) for r in per_run if len(r["delays"]) > 0
    ]
    all_delays = [d for r in per_run for d in r["delays"]]
    delay_m, delay_ci = ci95(delays_per_run)
    delay_std = (
        float(np.std(all_delays, ddof=1)) if len(all_delays) > 1 else 0.0
    )

    stable_vars = [r["stable_var"] for r in per_run if "stable_var" in r]
    sv_m, _ = ci95(stable_vars) if stable_vars else (float("nan"), 0.0)
    auc_m, auc_ci = ci95([r["auc"] for r in per_run if "auc" in r])
    thr_m, _ = ci95([r["thr"] for r in per_run if "thr" in r])

    return {
        "n": len(per_run),
        "f1_mean": f1_m,
        "f1_ci": f1_ci,
        "auc_mean": auc_m,
        "auc_ci": auc_ci,
        "far_mean": far_m,
        "far_ci": far_ci,
        "precision_mean": prec_m,
        "recall_mean": rec_m,
        "delay_mean": delay_m,
        "delay_ci": delay_ci,
        "delay_std": delay_std,
        "stable_var_mean": sv_m,
        "thr_mean": thr_m,
        "n_detected_delays": len(all_delays),
    }


# ===========================================================================
# Reporting
# ===========================================================================
def fmt(mean: float, ci: float, decimals: int = 3) -> str:
    """Format mean +- CI for plain text."""
    if math.isnan(mean):
        return "n/a"
    return f"{mean:.{decimals}f} +- {ci:.{decimals}f}"


def fmt_tex(mean: float, ci: float, decimals: int = 3) -> str:
    r"""Format mean +- CI for LaTeX (\pm)."""
    if math.isnan(mean):
        return "n/a"
    return f"${mean:.{decimals}f} \\pm {ci:.{decimals}f}$"


def print_block(method: str, agg: dict, tol: int) -> None:
    """Print a plain-text metric block for one method/experiment."""
    print(f"  [{method}]  (n={agg['n']}, tolerance=(cp, cp+{tol}] samples)")
    print(f"    Precision        : {agg['precision_mean']:.3f}")
    print(f"    Recall           : {agg['recall_mean']:.3f}")
    print(f"    F1 (calibrated)  : {fmt(agg['f1_mean'], agg['f1_ci'])}")
    print(f"    ROC-AUC          : {fmt(agg['auc_mean'], agg['auc_ci'])}")
    print(
        f"    Calib. threshold : {agg['thr_mean']:.4g}  "
        f"(mean + {CALIB_K:g}*std of stable score)",
    )
    print(
        f"    Mean delay [smpl]: {fmt(agg['delay_mean'], agg['delay_ci'], 1)}"
        f"  (pooled std={agg['delay_std']:.1f}, "
        f"n_detected={agg['n_detected_delays']})",
    )
    print(
        f"    False-alarm rate : {fmt(agg['far_mean'], agg['far_ci'], 4)}"
        "  (FP / 1000 change-free samples)",
    )
    if not math.isnan(agg["stable_var_mean"]):
        print(f"    Stable-region var: {agg['stable_var_mean']:.6g}")


def write_latex(results: dict, tolerances: dict) -> str:
    """Write the booktabs LaTeX table to benchmarks/m2_table.tex."""
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Change-point detection performance of the proposed "
        r"toDMDc detector against the OnlineSVD reference baseline "
        r"(OSVD-CPD), over $N$ Monte-Carlo runs. Values are "
        r"mean~$\pm$~95\,\% confidence interval ($1.96\,\sigma/\sqrt{N}$). "
        r"A detection is a true positive if it falls within the "
        r"$(t_c,\,t_c+c]$ tolerance window after a true change-point $t_c$, "
        r"with $c=\text{test\_size}$ the deterministic delay budget. F1 uses "
        r"a per-run threshold calibrated to $\mu+4\sigma$ of the score over a "
        r"stable pre-change window; ROC-AUC is threshold-free (score as a "
        r"per-sample change discriminant). False-alarm rate is false "
        r"positives per $1000$ change-free samples.}",
        r"  \label{tab:m2-cpd}",
        r"  \begin{tabular}{lllll}",
        r"    \toprule",
        r"    Method & F1 & ROC-AUC & Mean delay [samples] & "
        r"False-alarm rate \\",
        r"    \midrule",
    ]
    exp_titles = {
        "synthetic": (
            rf"Synthetic step changes ($N={results['synthetic']['n']}$, "
            rf"$c={tolerances['synthetic']}$)"
        ),
        "nonlinear": (
            rf"Nonlinear two-tank ($N={results['nonlinear']['n']}$, "
            rf"$c={tolerances['nonlinear']}$)"
        ),
    }
    method_labels = {"dmd": "toDMDc (proposed)", "svd": "OSVD-CPD"}
    for exp in ("synthetic", "nonlinear"):
        lines.append(
            rf"    \multicolumn{{5}}{{l}}{{\textit{{{exp_titles[exp]}}}}} \\",
        )
        for m in ("dmd", "svd"):
            a = results[exp][m]
            lines.append(
                "    "
                + method_labels[m]
                + " & "
                + fmt_tex(a["f1_mean"], a["f1_ci"])
                + " & "
                + fmt_tex(a["auc_mean"], a["auc_ci"])
                + " & "
                + fmt_tex(a["delay_mean"], a["delay_ci"], 1)
                + " & "
                + fmt_tex(a["far_mean"], a["far_ci"], 4)
                + r" \\",
            )
        if exp == "synthetic":
            lines.append(r"    \midrule")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    text = "\n".join(lines)
    (Path(__file__).resolve().parent / "m2_table.tex").write_text(text)
    return text


# ===========================================================================
# Resumable per-run cache
# ===========================================================================
CACHE_DIR = Path(__file__).resolve().parent / ".m2_cache"
RUNNERS = {"synthetic": run_synthetic_once, "nonlinear": run_nonlinear_once}


def _cache_path(exp: str) -> Path:
    return CACHE_DIR / f"{exp}.json"


def _load_cache(exp: str) -> dict:
    p = _cache_path(exp)
    if p.exists():
        return json.loads(p.read_text())
    return {"tol": None, "runs": {}}


def _save_cache(exp: str, cache: dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    _cache_path(exp).write_text(json.dumps(cache))


def ensure_runs(exp: str, seeds: list[int]) -> dict:
    """Run any missing seeds for ``exp`` and return the populated cache."""
    cache = _load_cache(exp)
    runner = RUNNERS[exp]
    for k, s in enumerate(seeds):
        key = str(s)
        if key in cache["runs"]:
            continue
        out = runner(int(s))
        cache["tol"] = int(out["tol"])
        cache["runs"][key] = {"dmd": out["dmd"], "svd": out["svd"]}
        _save_cache(exp, cache)
        print(
            f"  [{exp}] run {k + 1}/{len(seeds)} (seed={s}) cached",
            flush=True,
        )
    return cache


def _agg_from_cache(cache: dict, seeds: list[int], method: str) -> dict:
    return aggregate([cache["runs"][str(s)][method] for s in seeds])


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    """Run both experiments over Monte-Carlo seeds and emit reports."""
    import argparse

    parser = argparse.ArgumentParser(description="M2 CPD benchmark")
    parser.add_argument(
        "--exp",
        choices=["synthetic", "nonlinear", "all"],
        default="all",
        help="Which experiment(s) to run/refresh before reporting.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(MASTER_SEED)
    nl_seeds = [
        int(s) for s in rng.integers(0, 2**31 - 1, size=N_RUNS).tolist()
    ]
    # Synthetic uses the published real realizations y0..y4 (no generation),
    # so its "seeds" are the file indices. Nonlinear varies measurement-noise.
    exp_seeds = {
        "synthetic": list(range(SYN_N_FILES)),
        "nonlinear": nl_seeds,
    }

    to_run = ["synthetic", "nonlinear"] if args.exp == "all" else [args.exp]
    for exp in to_run:
        seeds = exp_seeds[exp]
        print(
            f"Ensuring {exp} experiment ({len(seeds)} runs)...",
            flush=True,
        )
        ensure_runs(exp, seeds)

    results: dict = {}
    tolerances: dict = {}
    for exp in ("synthetic", "nonlinear"):
        seeds = exp_seeds[exp]
        cache = _load_cache(exp)
        missing = [s for s in seeds if str(s) not in cache["runs"]]
        if missing:
            print(
                f"\n[!] {exp}: {len(missing)}/{len(seeds)} runs still missing; "
                f"re-run with --exp {exp} to finish before the final table.",
                flush=True,
            )
            return
        results[exp] = {
            "dmd": _agg_from_cache(cache, seeds, "dmd"),
            "svd": _agg_from_cache(cache, seeds, "svd"),
            "n": len(seeds),
        }
        tolerances[exp] = cache["tol"]

    tol_syn = tolerances["synthetic"]
    tol_nl = tolerances["nonlinear"]

    # --- Plain-text summary ---
    print("\n" + "=" * 72)
    print("M2 CPD BENCHMARK RESULTS")
    print("=" * 72)
    print(
        f"Master seed: {MASTER_SEED} | N_runs: {N_RUNS} | threshold eta: "
        f"{THRESHOLD}",
    )
    print(
        "Tolerance window: (cp, cp+c] with c = test_size (the deterministic\n"
        "delay budget). TP: >=1 detection in window (collapsed, no double\n"
        "count); FP: detection outside every window; FN: window with no\n"
        "detection. False-alarm rate = FP per 1000 change-free samples\n"
        "(change-free = total samples - union of tolerance windows).",
    )
    print("-" * 72)
    print(f"\nEXPERIMENT 1: Synthetic step changes (c={tol_syn})")
    print_block("toDMDc", results["synthetic"]["dmd"], tol_syn)
    print_block("OSVD-CPD", results["synthetic"]["svd"], tol_syn)

    sv_dmd = results["synthetic"]["dmd"]["stable_var_mean"]
    sv_svd = results["synthetic"]["svd"]["stable_var_mean"]
    print("\n  Stable-region detection-statistic variance:")
    print(f"    toDMDc   : {sv_dmd:.6g}")
    print(f"    OSVD-CPD : {sv_svd:.6g}")
    if sv_dmd > 0:
        ratio = sv_svd / sv_dmd
        print(
            f"    Ratio OSVD/toDMDc : {ratio:.4f}  "
            f"(OSVD has {100 * (ratio - 1):+.1f}% variance vs toDMDc)",
        )

    print(f"\nEXPERIMENT 2: Nonlinear two-tank (c={tol_nl})")
    print_block("toDMDc", results["nonlinear"]["dmd"], tol_nl)
    print_block("OSVD-CPD", results["nonlinear"]["svd"], tol_nl)

    # Corrected synthetic delay (mean +- std), pooled over detected CPs.
    a_dmd = results["synthetic"]["dmd"]
    print("\n" + "-" * 72)
    print(
        "Corrected synthetic detection delay (toDMDc, pooled over detected "
        f"CPs):\n    mean = {a_dmd['delay_mean']:.1f} samples, "
        f"pooled std = {a_dmd['delay_std']:.1f} samples "
        f"(n_detected={a_dmd['n_detected_delays']})",
    )

    # --- LaTeX ---
    tex = write_latex(results, tolerances)
    print("\n" + "=" * 72)
    print("LaTeX table written to benchmarks/m2_table.tex:")
    print("=" * 72)
    print(tex)


if __name__ == "__main__":
    main()
