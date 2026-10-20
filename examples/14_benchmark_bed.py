# %% [markdown]
# # A plug-and-play benchmark bed
#
# `cpdmd_compare.run_benchmark` treats every method identically: it calls
# `method(X) -> list[int]` and scores the result with the same causal
# `match_changepoints` metric. There is no "ours" vs. "baseline" branch --
# adding a method or a dataset is just adding a registry entry below.
#
# Four methods: **ODMD-CPD** (this repo), **CPDMD** (Khamesi baseline),
# **TS2Vec** and **SN-TS2Vec** (Bazarova et al. 2024,
# [arXiv:2410.13637](https://arxiv.org/abs/2410.13637)) -- spectral
# normalization applied to every conv layer of TS2Vec. SN-TS2Vec has no
# published code, so it's reproduced here as a subclass that spectral-norms
# the official [TS2Vec](https://github.com/zhihanyue/ts2vec) encoder
# (installed unmodified as the `ts2vec` package) -- same "wrap the external
# method, don't fork it" pattern `cpdmd_compare.py` uses for CPDMD. This is a
# good-faith reproduction, not the paper's exact eval protocol (they use a
# theoretically-grounded dynamic threshold; here it's a fixed quantile like
# the other methods' fixed thresholds).
#
# Both of the paper's datasets are included: **USC-HAD** is freely
# downloadable directly; **Yahoo A4Benchmark** normally requires signing
# Yahoo's Webscope data agreement, so it's pulled here from a third-party
# GitHub mirror
# ([harris0704/nbaData16-17](https://github.com/harris0704/nbaData16-17)) --
# convenient, but unofficial and unverified against the original release, so
# treat results on it as indicative rather than authoritative.
#
# Methods and datasets are plain `Callable`/tuple registries, not pydantic
# models: this is in-process code calling in-process code, not a request
# crossing a trust boundary (the case pydantic is for), and every value is
# already a `numpy.ndarray` -- there's nothing to parse or coerce, just a
# shape to satisfy, which a type checker verifies for free and pydantic would
# need `arbitrary_types_allowed` workarounds for anyway.
#
# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.nn.utils.parametrizations import spectral_norm
from ts2vec import TS2Vec

sys.path.append("..")  # repo root (for `reshift` / `cpdmd_compare`)
import cpdmd_compare as cc

cc.ensure_cpdmd()  # clone the CPDMD baseline into .baselines/ if absent

# %% [markdown]
# ## Methods
#
# Every entry is a zero-argument-except-`X` callable: `Callable[[np.ndarray],
# list[int]]`. Hyperparameters are bound here via closures, so the benchmark
# loop itself never sees a method-specific parameter.
# %%
GRID_FIXED = {"window": [200], "order": [20], "rank": [4]}


def odmd_cpd_scores(X: np.ndarray) -> np.ndarray:
    """ODMD-CPD's per-step test/base error-ratio score."""
    scores, _ = cc.run_odmd_cpd(
        X,
        window=300,
        hankel_n=20,
        ref_size=300,
        test_size=300,
        rank=2,
        threshold=0.5,
    )
    return scores


def odmd_cpd(X: np.ndarray) -> list[int]:
    """ODMD-CPD, fixed-threshold detector."""
    return cc.peaks(odmd_cpd_scores(X), 0.5, 300 + 300)


def cpdmd_scores(X: np.ndarray) -> np.ndarray:
    """CPDMD baseline's per-step normalized DMD reconstruction error (same fixed grid point)."""
    window, order, rank = (v[0] for v in GRID_FIXED.values())
    return cc.cpdmd_error_series(X, window=window, order=order, rank=rank)


def cpdmd(X: np.ndarray) -> list[int]:
    """CPDMD baseline, fixed grid."""
    return cc.run_cpdmd(X, burn_in=400, grid=GRID_FIXED)


def _apply_spectral_norm(module: torch.nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, torch.nn.Conv1d):
            spectral_norm(m)


class SNTS2Vec(TS2Vec):
    """TS2Vec with spectral normalization on every conv layer (Bazarova et al., 2024)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Build the base TS2Vec encoder, then spectral-norm its conv layers."""
        super().__init__(*args, **kwargs)
        _apply_spectral_norm(self._net)
        # rebuild the SWA-averaged inference net used by encode() so it picks
        # up the spectral-norm parametrization added above
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)


def _ts2vec_score(
    X: np.ndarray,
    model_cls: type[TS2Vec],
    window: int = 100,
    epochs: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """Embedding-shift score: |repr(t) - repr(t - window)|."""
    torch.manual_seed(seed)
    np.random.seed(
        seed,
    )  # ts2vec's augmentation masking draws from the global numpy RNG
    model = model_cls(
        input_dims=X.shape[1],
        output_dims=64,
        hidden_dims=64,
        depth=4,
        device="cpu",
    )
    model.fit(X[None].astype(np.float32), n_epochs=epochs, verbose=False)
    reprs = model.encode(
        X[None].astype(np.float32),
        causal=True,
        sliding_length=1,
        sliding_padding=window,
    )[0]
    score = np.zeros(len(reprs))
    score[window:] = np.linalg.norm(reprs[window:] - reprs[:-window], axis=1)
    return score


def ts2vec_scores(X: np.ndarray) -> np.ndarray:
    """Vanilla TS2Vec embedding-shift score."""
    return _ts2vec_score(X, TS2Vec)


def sn_ts2vec_scores(X: np.ndarray) -> np.ndarray:
    """SN-TS2Vec (Bazarova et al. 2024) embedding-shift score."""
    return _ts2vec_score(X, SNTS2Vec)


def ts2vec(X: np.ndarray) -> list[int]:
    """Vanilla TS2Vec, fixed-quantile threshold."""
    score = ts2vec_scores(X)
    return cc.peaks(score, np.quantile(score, 0.98), 100)


def sn_ts2vec(X: np.ndarray) -> list[int]:
    """SN-TS2Vec, fixed-quantile threshold."""
    score = sn_ts2vec_scores(X)
    return cc.peaks(score, np.quantile(score, 0.98), 100)


METHODS = {
    "ODMD-CPD": odmd_cpd,
    "CPDMD": cpdmd,
    "TS2Vec": ts2vec,
    "SN-TS2Vec": sn_ts2vec,
}

SCORERS = {
    "ODMD-CPD": odmd_cpd_scores,
    "CPDMD": cpdmd_scores,
    "TS2Vec": ts2vec_scores,
    "SN-TS2Vec": sn_ts2vec_scores,
}


# %% [markdown]
# ## Datasets
#
# Every entry is `(X, true_cps)`. Swap in real data by registering any other
# `(X, true_cps)` pair here -- the simulator doesn't care where it came from.
#
# `load_usc_had` concatenates `n_segments` randomly chosen activities (3-axis
# accelerometer only, z-normalized) from one subject; boundaries between
# activities are the true change points -- the same construction the SN-TS2Vec
# paper describes for its USC-HAD benchmark. `load_yahoo_a4` concatenates
# `n_series` randomly chosen Yahoo A4Benchmark series and reuses each series'
# own labeled `changepoint` column -- no boundary construction needed there,
# the ground truth ships with the data.
# %%
def load_usc_had(
    root: str,
    subject: int = 1,
    n_segments: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Concatenate random USC-HAD activity trials; segment boundaries are the true change points."""
    rng = np.random.default_rng(seed)
    activities = rng.choice(np.arange(1, 13), size=n_segments, replace=False)
    trial = int(rng.integers(1, 6))
    segments: list[np.ndarray] = []
    cps: list[int] = []
    pos = 0
    for a in activities:
        mat = loadmat(Path(root) / f"Subject{subject}" / f"a{a}t{trial}.mat")
        segments.append(
            mat["sensor_readings"][:, :3].astype(float),
        )  # 3-axis accelerometer
        pos += len(segments[-1])
        cps.append(pos)
    X = np.vstack(segments)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, cps[:-1]


def load_yahoo_a4(
    root: str,
    n_series: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Concatenate randomly chosen Yahoo A4Benchmark series; labeled `changepoint` rows are the true change points."""
    files = sorted(
        Path(root).glob("A4Benchmark-TS*.csv"),
        key=lambda p: int(p.stem.removeprefix("A4Benchmark-TS")),
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(files), size=n_series, replace=False)
    segments: list[np.ndarray] = []
    cps: list[int] = []
    pos = 0
    for i in chosen:
        df = pd.read_csv(files[i])
        segments.append(df["value"].to_numpy(dtype=float)[:, None])
        cps.extend(
            (np.flatnonzero(df["changepoint"].to_numpy()) + pos).tolist(),
        )
        pos += len(segments[-1])
    X = np.vstack(segments)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, sorted(cps)


DATASETS = {
    "steps": cc.gen_steps(
        10_000,
        [1, -1, 2, -2, 3, -3, 4, -4, 5],
        sigma=0.1,
        seed=42,
    ),
    "seasonal": cc.gen_seasonal(
        10_000,
        freqs=[0.02, 0.05, 0.03, 0.08],
        amps=[1.0, 1.0, 2.0, 1.5],
        sigma=0.1,
        seed=42,
    ),
    "usc-had": load_usc_had("data/usc-had/USC-HAD"),
    "yahoo-a4": load_yahoo_a4("data/yahoo-a4"),
}

# %% [markdown]
# ## Run
#
# One call, no branching: every method runs on every dataset through the same
# causal-forward-window metric.
# %%
results = cc.run_benchmark(DATASETS, METHODS, tol=250)
results.round(3)

# %% [markdown]
# ## Threshold-free: ROC AUC
#
# `run_benchmark` above scores one fixed threshold per method -- it can't
# separate "the score is uninformative" from "the threshold is wrong for this
# data". `run_auc_benchmark` sidesteps that: every `SCORERS` entry returns a
# raw per-step score, each timestep is labeled 1/0 by whether it falls in a
# true change's causal window, and ROC AUC summarizes ranking quality across
# *every* threshold at once. If two methods differ on `results` but not on
# `auc_results`, the difference is in threshold calibration, not in what the
# underlying score knows.
# %%
auc_results = cc.run_auc_benchmark(DATASETS, SCORERS, tol=250)
auc_results.pivot_table(
    index="dataset",
    columns="method",
    values="roc_auc",
).round(3)

# %% [markdown]
# ## Extending
#
# - New method: write a `Callable[[np.ndarray], list[int]]` and add it to
#   `METHODS`; if it has a natural continuous score, also add a
#   `Callable[[np.ndarray], np.ndarray]` to `SCORERS`.
# - New dataset: add an `(X, true_cps)` pair to `DATASETS` (synthetic or
#   loaded from disk).
#
# `run_benchmark` and `run_auc_benchmark` are unchanged either way.
