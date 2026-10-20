"""Standardized TEP benchmark sweep for the ODMD SubID change detector.

Threshold is calibrated on a held-out set of fault-free testing runs (no
knowledge of any fault onset), then applied unchanged to every (fault, run)
pair of the faulty testing set -- 20 faults x 500 runs, following the
conventional Rieth et al. (2017) evaluation protocol. Results are cached to
``results/.tep/tep_sweep.pkl``.
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from river.decomposition import OnlineDMD
from river.preprocessing import Hankelizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from reshift.chdsubid import SubIDChangeDetector
from reshift.rolling import Rolling

if TYPE_CHECKING:
    from river.compose import Pipeline

TEP_PATH = "/Users/mw/Developer/datasets/TEP"
FAULT_ONSET = 160  # sample (1-based) at which the fault is injected
# Hyperparameters were scanned against a *correctly labeled* clean AUC (real
# held-out fault-free scores as negatives, not the pre-onset scores of the
# faulty run itself -- see GRACE_PERIOD note below for why that self-labeling
# is invalid). Under the corrected metric, none of INIT_SIZE (50-125), R
# (2-30), or REF_SIZE/TEST_SIZE (10-100) move AUC meaningfully off ~0.53;
# ref=test=30 was the (marginal) local best of those tried. An earlier scan
# that used the contaminated self-labeled AUC wrongly concluded INIT_SIZE=130
# was a big improvement (0.740 -> 0.755) -- that "gain" was entirely an
# artifact of shrinking the genuine pre-onset window, not real.
INIT_SIZE = 100
REF_SIZE = 30
TEST_SIZE = 30
R = 10
N_CALIB_RUNS = 100  # fault-free runs used only to calibrate the threshold
CALIB_QUANTILE = 0.999
# Literature-standard operating point (e.g. Heo & Lee 2019, *Processes* 7(7):411)
# for the sample-level fault detection rate (FDR) comparison: FAR = 1%.
LIT_FAR_QUANTILE = 0.99

# Samples before this index score exactly 0.0 (the model hasn't finished its
# warm-up), so they must never be used as a "known normal" label -- doing so
# previously produced a badly inflated AUC (0.759) by comparing real
# post-onset scores against these trivial zeros instead of genuine fault-free
# noise. The assert is a guard against that bug class recurring silently.
GRACE_PERIOD = INIT_SIZE + TEST_SIZE + 1
assert GRACE_PERIOD < FAULT_ONSET - 1, (
    "grace_period must leave a genuine pre-onset scoring margin"
)

RESULTS_PATH = (
    Path(__file__).resolve().parent / "results" / ".tep" / "tep_sweep.pkl"
)


def build_pipeline() -> Pipeline:
    """Build the ODMD-based change-point detection pipeline."""
    odmd = Rolling(
        OnlineDMD(
            r=R,
            initialize=INIT_SIZE,
            w=1.0,
            exponential_weighting=False,
            eig_rtol=1e-1,
        ),
        GRACE_PERIOD,
    )
    subid = SubIDChangeDetector(
        odmd,
        ref_size=REF_SIZE,
        test_size=TEST_SIZE,
        grace_period=GRACE_PERIOD,
    )
    return Hankelizer(1) | subid


def score_run(X: pd.DataFrame) -> np.ndarray:
    """Run the pipeline over a single (fault, run) sequence."""
    pipeline = build_pipeline()
    scores = np.empty(len(X))
    for i, x in enumerate(X.to_dict(orient="records")):
        scores[i] = pipeline.score_one(x)
        pipeline.learn_one(x)
    return scores


def _score_free(args: tuple[float, pd.DataFrame]) -> tuple[float, np.ndarray]:
    run_id, X = args
    return run_id, score_run(X)


def _score_faulty(
    args: tuple[int, float, pd.DataFrame],
) -> tuple[int, float, np.ndarray]:
    fault, run_id, X = args
    return fault, run_id, score_run(X)


def main() -> None:
    """Calibrate the threshold and sweep all TEP faults/runs, then cache results."""
    import pyreadr

    print("Loading fault-free testing set...")
    free = next(
        iter(
            pyreadr.read_r(f"{TEP_PATH}/TEP_FaultFree_Testing.RData").values(),
        ),
    )
    print("Loading faulty testing set...")
    faulty = next(
        iter(pyreadr.read_r(f"{TEP_PATH}/TEP_Faulty_Testing.RData").values()),
    )
    feature_cols = [
        c
        for c in free.columns
        if c not in ("faultNumber", "simulationRun", "sample")
    ]

    free_runs = sorted(free["simulationRun"].unique())
    calib_runs, holdout_runs = (
        free_runs[:N_CALIB_RUNS],
        free_runs[N_CALIB_RUNS:],
    )

    def free_run(run_id: float) -> pd.DataFrame:
        sub = free[free["simulationRun"] == run_id]
        return sub.sort_values("sample")[feature_cols]

    calib_args = [(r, free_run(r)) for r in calib_runs]
    holdout_args = [(r, free_run(r)) for r in holdout_runs]

    print(f"Calibrating threshold on {len(calib_args)} fault-free runs...")
    with ProcessPoolExecutor() as ex:
        calib_results = list(ex.map(_score_free, calib_args, chunksize=4))
    calib_scores = np.concatenate([s for _, s in calib_results])
    threshold = float(np.quantile(calib_scores, CALIB_QUANTILE))
    lit_threshold = float(np.quantile(calib_scores, LIT_FAR_QUANTILE))
    print(
        f"threshold={threshold:.5f}  lit_threshold(FAR=1%)={lit_threshold:.5f}",
    )

    print(f"Evaluating FAR on {len(holdout_args)} held-out fault-free runs...")
    with ProcessPoolExecutor() as ex:
        holdout_results = list(
            ex.map(_score_free, holdout_args, chunksize=4),
        )
    holdout_scores = dict(holdout_results)
    far_sample_rate = float(
        np.mean([(s > threshold).mean() for _, s in holdout_results]),
    )
    far_run_rate = float(
        np.mean([(s > threshold).any() for _, s in holdout_results]),
    )
    lit_far_sample_rate = float(
        np.mean([(s > lit_threshold).mean() for _, s in holdout_results]),
    )

    faults = sorted(int(f) for f in faulty["faultNumber"].unique())
    runs = sorted(faulty["simulationRun"].unique())
    faulty_args = [
        (
            f,
            r,
            faulty[
                (faulty["faultNumber"] == f) & (faulty["simulationRun"] == r)
            ].sort_values("sample")[feature_cols],
        )
        for f in faults
        for r in runs
    ]

    print(f"Evaluating {len(faulty_args)} faulty runs...")
    records = []
    faulty_scores: dict[tuple[int, float], np.ndarray] = {}
    with ProcessPoolExecutor() as ex:
        for fault, run_id, s in ex.map(
            _score_faulty,
            faulty_args,
            chunksize=4,
        ):
            faulty_scores[fault, run_id] = s
            alarms = s > threshold
            pre = alarms[: FAULT_ONSET - 1]
            post = alarms[FAULT_ONSET - 1 :]
            detected = bool(post.any())
            # Sample-level fault detection rate at the literature FAR=1%
            # operating point (e.g. Heo & Lee 2019) -- the metric needed for
            # an apples-to-apples comparison, as opposed to the run-level
            # "detected at least once" view above.
            lit_post = s[FAULT_ONSET - 1 :] > lit_threshold
            records.append(
                {
                    "fault": fault,
                    "run": run_id,
                    "pre_false_alarms": int(pre.sum()),
                    "detected": detected,
                    "delay": int(np.argmax(post)) if detected else None,
                    "sample_fdr": float(lit_post.mean()),
                },
            )

    # Clean, threshold-free AUC per fault: post-onset scores of that fault's
    # runs (positive) against genuinely-scored held-out fault-free samples
    # (negative). Never use a faulty run's own pre-onset samples as the
    # negative class -- GRACE_PERIOD < FAULT_ONSET guarantees those are
    # score-valid, but they're still the same run under test, not an
    # independent normal-operation reference.
    from sklearn.metrics import roc_auc_score

    neg_scores = np.concatenate(
        [s[GRACE_PERIOD:] for s in holdout_scores.values()],
    )
    clean_auc_per_fault = {}
    for fault in faults:
        pos_scores = np.concatenate(
            [
                s[FAULT_ONSET - 1 :]
                for (f, _run), s in faulty_scores.items()
                if f == fault
            ],
        )
        y = np.concatenate(
            [np.zeros(len(neg_scores)), np.ones(len(pos_scores))],
        )
        clean_auc_per_fault[fault] = float(
            roc_auc_score(y, np.concatenate([neg_scores, pos_scores])),
        )

    results = {
        "threshold": threshold,
        "far_sample_rate": far_sample_rate,
        "far_run_rate": far_run_rate,
        "lit_threshold": lit_threshold,
        "lit_far_sample_rate": lit_far_sample_rate,
        "n_calib_runs": len(calib_args),
        "n_holdout_runs": len(holdout_args),
        "records": pd.DataFrame(records),
        "clean_auc_per_fault": clean_auc_per_fault,
        # Raw per-sample scores, cached so metrics at a different operating
        # point / protocol can be recomputed later without re-running the
        # (expensive) online pipeline over all 10500 runs again.
        "faulty_scores": faulty_scores,
        "calib_scores": dict(calib_results),
        "holdout_scores": holdout_scores,
        "params": {
            "init_size": INIT_SIZE,
            "ref_size": REF_SIZE,
            "test_size": TEST_SIZE,
            "r": R,
            "fault_onset": FAULT_ONSET,
            "grace_period": GRACE_PERIOD,
            "calib_quantile": CALIB_QUANTILE,
            "lit_far_quantile": LIT_FAR_QUANTILE,
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
