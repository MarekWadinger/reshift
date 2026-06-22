"""Talk figures for the ECC26 toDMDc presentation.

Renders projector-ready (large-font, high-contrast, annotated) versions of the
paper figures, tuned to convey ONE point each. Reuses the published cached
scores / real data -- no re-derivation, so figures match the paper.

Run: ``uv run python publications/ECC26/talk/figs/make_talk_figs.py``
Outputs PNG (preview) + PDF (slides) into this directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RESULTS = REPO / "examples" / "results" / ".bess"
DATA = REPO / "examples" / "data"

# --- talk style: big, clean, projector-legible -----------------------------
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 17,
        "axes.labelsize": 18,
        "axes.titlesize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.6,
        "figure.dpi": 130,
    },
)
BLUE, RED, GREY, GREEN = "#1f6fb2", "#d1495b", "#9aa0a6", "#2a9d8f"


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# --- shared runners (reproduce the paper detectors) ------------------------
def _gen_steps_1d(
    n: int,
    n_changes: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Univariate step signal: dᵢ = 0.5·i every n/(c+1) samples (paper synthetic)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    interval = n // (n_changes + 1)
    cps = np.arange(interval, n, interval)[:n_changes]
    level = 0.0
    for i in range(n):
        if i in cps:
            level += 0.5 * (np.where(cps == i)[0][0] + 1)
        x[i] = level + rng.normal(0, sigma)
    return x.reshape(-1, 1)


def _run_detector(
    X: np.ndarray,
    kind: str,
    *,
    window: int,
    hn: int,
    ref: int,
    test: int,
    rank: int,
) -> np.ndarray:
    """Stream X through toDMDc (OnlineDMD) or the SVD-reference (OnlineSVD)."""
    from river.decomposition import OnlineDMD, OnlineSVD
    from river.preprocessing import Hankelizer

    from reshift.chdsubid import SubIDChangeDetector
    from reshift.rolling import Rolling

    if kind == "dmd":
        core = Rolling(
            OnlineDMD(
                r=rank,
                initialize=window,
                w=1.0,
                exponential_weighting=False,
                seed=42,
            ),
            window + 1,
        )
    else:
        core = Rolling(
            OnlineSVD(n_components=rank, initialize=window, seed=42),
            window + 1,
        )
    det = SubIDChangeDetector(
        core,
        ref_size=ref,
        test_size=test,
        threshold=0.5,
        grace_period=window + test + 1,
    )
    pipe = Hankelizer(hn) | det
    s = np.zeros(X.shape[0])
    for i, x in enumerate(pd.DataFrame(X).to_dict(orient="records")):
        s[i] = pipe.score_one(x)
        pipe.learn_one(x)
    return s


# --- BESS hero (slide 10) --------------------------------------------------
ONSET = 7824  # ground-truth fan-fault onset (Aug 23); = 65.2 h from start


def fig_bess(variant: str = "hx10", *, crop: bool = True) -> None:
    """Real battery fault. crop=True zooms to the onset (clean detection-at-c
    story); crop=False shows the full run with late return-to-normal peaks.
    """
    res = json.load(
        (
            RESULTS
            / f"bess-chd_p10-l2880_b240_t240roll_2880-dmd_w1.0-{variant}.json"
        ).open(),
    )
    q = np.nan_to_num(np.array(res["scores_dmd"]))
    temp = pd.read_csv(DATA / "kokam_norm.csv").iloc[:, 1:].to_numpy()
    n = len(q)
    test_c = 240
    h = np.arange(n) * 30 / 3600.0
    peak = ONSET + int(
        np.argmax(q[ONSET : ONSET + 600]),
    )  # detection peak after onset
    # pre-onset noise floor (for an honest "bumps near floor" band, not a cherry-picked arrow)
    floor_p95 = float(np.percentile(q[3219:ONSET], 95))
    thr = 0.8

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    ax0.plot(h, temp, lw=0.8, alpha=0.85)
    ax0.axvspan(h[ONSET], h[-1], color=RED, alpha=0.06)
    ax0.set_ylabel("module temp\n(normalised)")
    ax0.set_title(
        "Real BESS cooling-fan fault — detected on data the model never saw",
    )

    ax1.plot(h, q, color=BLUE, lw=1.7)
    ax1.axhline(thr, color=GREY, ls=":", lw=1.4)
    ax1.text(h[3300], thr + 0.05, "threshold", color=GREY, fontsize=13)
    ax1.axvline(h[ONSET], color=RED, lw=2.0)
    ax1.text(
        h[ONSET] - 0.3,
        q.max() * 0.96,
        "fault onset",
        color=RED,
        fontsize=14,
        ha="right",
        va="top",
        fontweight="bold",
    )
    ax1.annotate(
        f"detected {peak - ONSET} samples\nafter onset  ≈  c = {test_c}",
        xy=(h[peak], q[peak]),
        xytext=(h[peak] + 2.0, q[peak] * 0.9),
        color=BLUE,
        fontsize=14,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.8},
    )
    # honest pre-onset context: shade the noise-floor band instead of an arrow
    ax1.axhspan(0, floor_p95, color=GREY, alpha=0.12)
    ax1.text(
        h[3400],
        floor_p95 * 0.5,
        "pre-onset noise floor\n(bumps here are not reliable)",
        color=GREY,
        fontsize=11,
        va="center",
    )
    ax1.set_ylabel("change score $Q_k$")
    ax1.set_xlabel("time since start (hours)")
    ax1.set_ylim(0, q.max() * 1.1)
    if crop:
        ax1.set_xlim(h[ONSET - 3000], h[min(ONSET + 2500, n - 1)])
        name = f"fig_bess_{variant}_crop"
    else:
        ax1.set_xlim(h[3219], h[-1])
        # label the late peaks honestly as return-to-normal (bidirectional)
        for li in [i for i in (18122, 20881) if i < n and q[i] > 1.0]:
            ax1.annotate(
                "late peak\n(after fault window;\nuncertain origin)",
                xy=(h[li], q[li]),
                xytext=(h[li] - 7, q[li]),
                color=GREY,
                fontsize=11,
                arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.2},
            )
        name = f"fig_bess_{variant}_full"
    fig.align_ylabels((ax0, ax1))
    fig.tight_layout()
    _save(fig, name)


# --- Robustness / slide 7 --------------------------------------------------
def fig_stability() -> None:
    """ToDMDc vs SVD-reference on the synthetic steps: truncated DMD gives a
    lower-variance statistic that resolves small early changes the SVD baseline
    buries in noise (the paper's ~25%-lower-variance robustness claim).
    """
    X = _gen_steps_1d(10000, 9, sigma=1.5, seed=42)
    cps = [1000 * i for i in range(1, 10)]
    kw = {"window": 300, "hn": 20, "ref": 300, "test": 300, "rank": 2}
    s_dmd = _run_detector(X, "dmd", **kw)
    s_svd = _run_detector(X, "svd", **kw)
    warm = 601
    v_dmd, v_svd = np.nanstd(s_dmd[warm:]), np.nanstd(s_svd[warm:])
    redux = round(100 * (v_svd - v_dmd) / v_svd)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    for c in cps:
        for ax in (ax0, ax1):
            ax.axvline(c, color=RED, lw=1.2, alpha=0.5)
    ax0.plot(s_dmd, color=BLUE, lw=1.4)
    ax0.set_ylabel("toDMDc $Q_k$")
    ax0.set_title(
        f"Truncation → a calmer statistic ({redux}% lower variance than the SVD baseline)",
    )
    ax1.plot(s_svd, color=GREY, lw=1.4)
    ax1.set_ylabel("SVD-reference")
    ax1.set_xlabel("sample")
    ax0.annotate(
        "small early changes\nstay visible",
        xy=(1000, s_dmd[900:1100].max()),
        xytext=(1400, ax0.get_ylim()[1] * 0.7),
        color=BLUE,
        fontsize=13,
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.5},
    )
    fig.align_ylabels((ax0, ax1))
    fig.tight_layout()
    _save(fig, "fig_stability_variance")
    print(f"    std: toDMDc={v_dmd:.3f}  SVD={v_svd:.3f}  ({redux}% lower)")


# --- Two-tank spine proof (slide 9) ----------------------------------------
def fig_twotank() -> None:
    """Controlled nonlinear plant: the score stays flat through every control
    command and fires only at the doubled-response event (same input → different
    response). This is the control-vs-fault claim, proven.
    """
    from river.decomposition import OnlineDMDwC
    from river.feature_extraction import PolynomialExtender
    from river.preprocessing import Hankelizer

    from reshift.chdsubid import (
        SubIDChangeDetector,
        get_default_rank,
        get_default_timedelays,
    )
    from reshift.preprocessing import hankel
    from reshift.rolling import Rolling

    td = pd.read_pickle(DATA / "nonlinear-delay-control" / "train_sim.pkl")  # noqa: S301
    X = pd.DataFrame(td["X"][:12000])
    U = pd.DataFrame(td["U"][:12000])
    rng = np.random.default_rng(42)
    X += rng.normal(0, 0.35, X.shape)
    X.iloc[3998:4998] += 1.0  # sensor bias
    X.iloc[7598:8598] *= 2.0  # doubled response  <-- the proof
    X.iloc[9798:12000] *= np.linspace(1.0, 2.0, 2202)[:, None]  # drift
    window, ref, test = 2000, 200, 200
    hm, hm_step = get_default_timedelays(200, 30 // X.shape[1])
    hl, hl_step = get_default_timedelays(30, 30 // U.shape[1])
    p = min(X.shape[1], get_default_rank(hankel(X[:window], hm, hm_step)))
    qd = min(U.shape[1], get_default_rank(hankel(U[:window], hl, hl_step)))
    U_ = pd.DataFrame(hankel(U, hl, hl_step))
    odmd = Rolling(
        OnlineDMDwC(
            p=p,
            q=qd,
            initialize=window - 1,
            w=1.0,
            exponential_weighting=False,
            eig_rtol=None,
        ),
        window,
    )
    det = SubIDChangeDetector(
        odmd,
        ref_size=ref,
        test_size=test,
        grace_period=window,
        start_soon=True,
    )
    pipe = PolynomialExtender(2) | Hankelizer(hm) | det
    n = X.shape[0]
    s = np.zeros(n)
    xs = X.to_dict(orient="records")
    us = U_.to_dict(orient="records")
    for i in range(n):
        s[i] = pipe.score_one(xs[i])
        pipe.learn_one(xs[i], u=us[i])
    np.save(HERE / "_twotank_score.npy", s)  # cache for re-styling

    # three injected faults (honest labels matching the data manipulation)
    events = [
        (3998, 4998, "sensor\nbias"),
        (7598, 8598, "input→state\ngain ×2"),
        (9798, 12000, "drift"),
    ]
    sn = np.nan_to_num(s)
    fig, axs = plt.subplots(
        3,
        1,
        figsize=(12, 6.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 0.7, 1]},
    )
    axs[0].plot(X.to_numpy(), lw=0.7)
    axs[0].set_ylabel("tank levels")
    axs[0].set_title(
        "Nonlinear controlled plant — detects sensor bias, a gain change, and drift",
    )
    axs[1].plot(U.to_numpy(), lw=0.8, color=GREY)
    axs[1].set_ylabel("control $u$")
    axs[2].plot(sn, color=BLUE, lw=1.4)
    axs[2].set_ylabel("change score $Q_k$")
    axs[2].set_xlabel("sample")
    for e0, e1, lab in events:
        for ax in axs:
            ax.axvspan(e0, e1, color=RED, alpha=0.10)
        pk = e0 + int(np.nanargmax(sn[e0 : min(e1 + 300, n)]))
        axs[2].annotate(
            lab,
            xy=(pk, sn[pk]),
            xytext=(pk, sn[pk] + 6),
            color=RED,
            fontsize=12,
            ha="center",
            fontweight="bold",
        )
    axs[2].set_xlim(window, n)
    fig.align_ylabels(axs)
    fig.tight_layout()
    _save(fig, "fig_twotank_proof")


# --- The trap (slide 4) ----------------------------------------------------
def fig_trap() -> None:
    """Concept: a control step (NORMAL big swing) vs the same input producing a
    different response (FAULT). A flat threshold on the signal fires on both.
    """
    rng = np.random.default_rng(3)
    t = np.arange(600)
    y = np.full(600, 1.0)
    u = np.zeros(600)
    u[150:300] = 1.0  # control step -> normal swing
    y[150:300] = 3.0
    u[420:570] = 1.0  # SAME control step ...
    y[420:570] = 5.0  # ... different (faulty) response
    y += rng.normal(0, 0.18, 600)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(11, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 0.5]},
    )
    ax0.plot(t, y, color=BLUE, lw=1.8)
    ax0.axhline(2.0, color=GREY, ls=":", lw=1.4)
    ax0.text(5, 2.1, "signal threshold", color=GREY, fontsize=12)
    ax0.annotate(
        "control acts\nNORMAL",
        xy=(225, 3),
        xytext=(150, 5.6),
        color=GREEN,
        fontsize=13,
        ha="center",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.6},
    )
    ax0.annotate(
        "same input,\ndifferent response → FAULT",
        xy=(495, 5),
        xytext=(360, 6.2),
        color=RED,
        fontsize=13,
        ha="center",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.6},
    )
    ax0.set_ylabel("output $x$")
    ax0.set_ylim(0, 8)
    ax0.set_title(
        "A signal-watcher can't tell the two apart — both cross the line",
    )
    ax1.plot(t, u, color=GREY, lw=1.8)
    ax1.fill_between(t, 0, u, color=GREY, alpha=0.2)
    ax1.set_ylabel("input $u$")
    ax1.set_xlabel("time")
    ax1.set_yticks([0, 1])
    # make the key point unmissable: the two input pulses are IDENTICAL
    for c in (225, 495):
        ax1.annotate(
            "",
            xy=(c - 75, 1.15),
            xytext=(c + 75, 1.15),
            arrowprops={"arrowstyle": "<->", "color": "black", "lw": 1.0},
        )
    ax1.text(
        360,
        1.45,
        "identical input",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylim(0, 1.8)
    fig.align_ylabels((ax0, ax1))
    fig.tight_layout()
    _save(fig, "fig_trap")


def fig_cold_open() -> None:
    """Slide 1: a single battery temperature creeping up — stakes in one trace."""
    temp = pd.read_csv(DATA / "kokam_norm.csv").iloc[:, 1].to_numpy()
    h = np.arange(len(temp)) * 30 / 3600.0
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(h, temp, color=RED, lw=1.3)
    ax.set_ylabel("battery module\ntemperature")
    ax.set_xlabel("time (hours)")
    ax.set_yticks([])
    ax.set_title(
        "Its cooling is failing — and a standard monitor can't tell that\n"
        "from the controller doing its job",
        fontsize=16,
    )
    fig.tight_layout()
    _save(fig, "fig_cold_open")


if __name__ == "__main__":
    print("Rendering talk figures...")
    fig_cold_open()
    fig_trap()
    fig_bess("hx10", crop=True)
    fig_bess("hx10", crop=False)
    fig_bess("hx20", crop=True)
    fig_twotank()
    print("Done.")
