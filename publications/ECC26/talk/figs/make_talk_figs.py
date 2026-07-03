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


def _save(fig: plt.Figure, name: str, *, tight: bool = True) -> None:
    # tight=False keeps the full fixed canvas (so two figures of equal figsize
    # render at the same scale on a slide — identical font sizes).
    kw = {"bbox_inches": "tight"} if tight else {}
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", **kw)
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
    thr = 0.8

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    ax0.plot(h, temp, lw=0.8, alpha=0.85)
    ax0.set_ylabel("temperature\n(normalised)")

    ax1.plot(h, q, color=BLUE, lw=1.7)
    ax1.axhline(thr, color=GREY, ls=":", lw=1.4)
    # reference (blue) / test (green) windows + gray dashed onset, matching the
    # companion plots; onset coloured like the companion's change marker.
    lo, hi = max(0, ONSET - test_c), min(n - 1, ONSET + test_c)
    for ax in (ax0, ax1):
        ax.axvspan(h[lo], h[ONSET], color=APP_REF, alpha=0.13)
        ax.axvspan(h[ONSET], h[hi], color=APP_TEST, alpha=0.16)
        ax.axvline(h[ONSET], color=APP_PLANT, ls="--", lw=1.6)
    ax1.annotate(
        f"detected {peak - ONSET} samples\nafter onset  ≈  c = {test_c}",
        xy=(h[peak], q[peak]),
        xytext=(h[peak] + 2.0, q[peak] * 0.9),
        color=BLUE,
        fontsize=14,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.8},
    )
    ax1.set_ylabel("score")
    ax1.set_xlabel("time since start (hours)")
    ax1.set_ylim(0, q.max() * 1.1)
    if crop:
        ax1.set_xlim(h[ONSET - 3000], h[min(ONSET + 2500, n - 1)])
        name = f"fig_bess_{variant}_crop"
    else:
        ax1.set_xlim(h[3219], h[-1])
        name = f"fig_bess_{variant}_full"
    fig.align_ylabels((ax0, ax1))
    # full canvas + shared left/right (tight=False): 12-inch width and identical
    # plot extent as the companion figs → same on-slide font size AND width.
    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, top=0.985, bottom=0.10, hspace=0.13)
    _save(fig, name, tight=False)


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


# --- Companion two-tank (slide 9) ------------------------------------------
# window_explorer.html constants, mirrored so the talk figures match the app.
APP_PLANT, APP_THRESH = "#999999", "#111111"  # plant gray, threshold black
APP_REF, APP_TEST = "#2f86d6", "#27ae60"  # reference blue, test green bands
APP_METHODS = ["DMD", "DMDc", "onlineDMD", "onlineDMDc", "toDMDc"]
# diverges from the app's default palette on purpose: projector-legible,
# high-contrast, and toDMDc carries the deck's emphasis blue.
APP_COLOR = {
    "DMD": "#999999",
    "DMDc": "#e0a800",
    "onlineDMD": "#d1495b",
    "onlineDMDc": "#9467bd",
    "toDMDc": "#1f6fb2",
}
APP_LABEL = {"onlineDMD": "oDMD", "onlineDMDc": "oDMDc"}
APP_REF_N = APP_TEST_N = 60  # default ref/test window size
APP_THR = 0.25  # default threshold (slider 25 / 100)
COMPANION_FIGSIZE = (12, 7.0)  # shared by both companion figs → equal slide scale
# shared plot-area left/right (figure fraction) so every data plot spans the SAME
# horizontal extent on its slide — only thing that makes the x-axes line up.
AX_LEFT, AX_RIGHT = 0.11, 0.965


def _app_lbl(m: str) -> str:
    """Display label for a method (window_explorer.html lbl())."""
    return APP_LABEL.get(m, m)


def _web_score(
    r: np.ndarray,
    ref: int,
    test: int,
    lag: int = 0,
    warmup: int = 0,
) -> np.ndarray:
    """The window_explorer.html score, verbatim: max(D_test/D_ref − 1, 0) over
    sliding ref/test windows (see scores() in examples/window_explorer.html).
    Samples before the windows fill are NaN (the app's null), excluded downstream.
    """
    L = ref + lag + test
    out = np.full(len(r), np.nan)
    for t in range(max(L - 1, warmup), len(r)):
        start = t - L + 1
        out[t] = max(
            r[t - test + 1 : t + 1].mean() / r[start : start + ref].mean() - 1,
            0.0,
        )
    return out


def _web_auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC with tie-averaged ranks (window_explorer.html auc())."""
    if not pos or not neg:
        return float("nan")
    allv = sorted(
        [(v, 1) for v in pos] + [(v, 0) for v in neg],
        key=lambda a: a[0],
    )
    i, rank_sum, m = 0, 0.0, len(allv)
    while i < m:
        j = i
        while j < m and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2
        rank_sum += sum(avg for k in range(i, j) if allv[k][1] == 1)
        i = j
    np_, nn = len(pos), len(neg)
    return (rank_sum - np_ * (np_ + 1) / 2) / (np_ * nn)


def _web_region_bounds(
    clean: np.ndarray,
    ref: int,
    test: int,
    lag: int,
    onset: int | None,
) -> tuple[int, int, int]:
    """(N, posS, posE) detection region (window_explorer.html regionBounds())."""
    clean = np.asarray(clean)
    nn = len(clean)
    d_eps = 0.005 * ((clean.max() - clean.min()) or 1.0)
    cs = ce = -1
    for t in range(1, nn):
        if abs(clean[t] - clean[t - 1]) > d_eps:
            cs = t if cs < 0 else cs
            ce = t
    pos_s = onset if onset is not None else (0 if cs < 0 else cs)
    true_e = 0 if ce < 0 else max(ce, pos_s)
    return nn, pos_s, min(nn - 1, true_e + ref + test + lag)


def _web_metrics(
    sc: np.ndarray,
    clean: np.ndarray,
    ref: int,
    test: int,
    lag: int,
    thr: float,
    onset: int | None,
) -> dict:
    """Per-model detection metrics (window_explorer.html metrics())."""
    nn, pos_s, pos_e = _web_region_bounds(clean, ref, test, lag, onset)
    peak, peak_idx, cross_idx = -1.0, -1, -1
    for t in range(pos_s, pos_e + 1):
        s = sc[t]
        if np.isnan(s):
            continue
        if s > peak:
            peak, peak_idx = s, t
        if cross_idx < 0 and s > thr:
            cross_idx = t
    fp = nrm = 0
    pos_v: list[float] = []
    nrm_v: list[float] = []
    for t in range(nn):
        s = sc[t]
        if np.isnan(s):
            continue
        if pos_s <= t <= pos_e:
            pos_v.append(s)
        else:
            nrm += 1
            nrm_v.append(s)
            fp += s > thr
    return {
        "far": fp / nrm if nrm else 0.0,
        "auc": _web_auc(pos_v, nrm_v),
        "detected": peak > thr,
        "delayPeak": peak_idx - pos_s if peak_idx >= 0 else None,
        "delayCross": cross_idx - pos_s if cross_idx >= 0 else None,
    }


def _companion_series(method: str, width: float = 60.0) -> dict:
    """Run one app identifier on the fouling plant; return its score + metrics
    alongside the raw residual backend payload. The single reusable block both
    companion figures build on."""
    from examples.plant_residual_server import LEARN_W, compute_residual

    d = compute_residual(method, "permanent", width, 0.08)
    sc = _web_score(
        np.array(d["r"]),
        APP_REF_N,
        APP_TEST_N,
        0,
        LEARN_W,
    )
    met = _web_metrics(
        sc,
        np.array(d["clean"]),
        APP_REF_N,
        APP_TEST_N,
        0,
        APP_THR,
        d["onset"],
    )
    return {"key": method, "raw": d, "score": sc, "metrics": met}


def fig_twotank_companion() -> None:
    """The exact plant + identifier the live companion runs (window_explorer.html):
    a controlled two-tank whose outflow coefficient slowly fouls. Mirrors the app's
    look: split gray plant levels (h1, h2) with the toDMDc one-step prediction
    overlaid, gray inflow q, the toDMDc change score, and the reference/test window
    undercolors. The inflow never changes (same input) yet the response departs as
    the outflow fouls — the spine claim, on the plant the audience plays with."""
    from examples.plant_residual_server import CP, LEARN_W

    ser = _companion_series("toDMDc")
    d = ser["raw"]
    states, pred = np.array(d["states"]), np.array(d["pred"])
    q, s = np.array(d["inputs"]), ser["score"]
    n = len(s)
    x = np.arange(n)
    # app band positions (bandShapes in window_explorer.html), tcur=onset+test-1
    ref_band, test_band = (CP - APP_REF_N, CP), (CP, CP + APP_TEST_N)

    fig, axs = plt.subplots(
        4,
        1,
        figsize=COMPANION_FIGSIZE,  # shared canvas → same on-slide scale as the scores fig
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.6, 1.0]},
    )
    # split levels: one panel per tank, gray plant + brown toDMDc prediction
    for i in range(states.shape[0]):
        axs[i].plot(x, states[i], color=APP_PLANT, lw=1.6)
        axs[i].plot(x, pred[i], color=APP_COLOR["toDMDc"], lw=1.3, ls="--")
        axs[i].set_ylabel(f"$h_{i + 1}$ (m)")
    from matplotlib.patches import Patch

    axs[0].plot([], [], color=APP_PLANT, lw=1.6, label="plant")
    axs[0].plot([], [], color=APP_COLOR["toDMDc"], lw=1.3, ls="--", label="toDMDc")
    axs[0].legend(
        handles=[
            *axs[0].get_legend_handles_labels()[0],
            Patch(facecolor=APP_REF, alpha=0.35, label="reference window"),
            Patch(facecolor=APP_TEST, alpha=0.40, label="test window"),
        ],
        loc="upper right",
        ncol=4,
        frameon=False,
        fontsize=13,
    )

    axs[2].plot(x, q, color=APP_PLANT, lw=1.4)
    axs[2].set_ylabel("input $q$ (m³/s)")

    axs[3].plot(x, s, color=APP_COLOR["toDMDc"], lw=1.7)
    axs[3].axhline(APP_THR, color=APP_THRESH, ls="--", lw=1.2)
    axs[3].set_ylabel("score")
    axs[3].set_xlabel("sample")

    # reference (blue) and test (green) window undercolors, app colours/opacities
    for ax in axs:
        ax.axvspan(*ref_band, color=APP_REF, alpha=0.13)
        ax.axvspan(*test_band, color=APP_TEST, alpha=0.16)
        ax.axvline(CP, color=APP_PLANT, ls="--", lw=1.2)
        ax.set_xlim(LEARN_W, n)
    axs[0].text(
        CP,
        1.04,
        "fouling starts",
        color="#555555",
        ha="center",
        fontsize=13,
        transform=axs[0].get_xaxis_transform(),
    )
    fig.align_ylabels(axs)
    # bottom=0.11 keeps the "sample" xlabel inside the fixed canvas (tight=False)
    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, top=0.96, bottom=0.11, hspace=0.30)
    _save(fig, "fig_twotank_companion", tight=False)


def fig_twotank_companion_scores() -> None:
    """All app identifiers on the fouling plant: the change score per method (in
    window_explorer.html colours) over the app's metrics table. Reuses the same
    _companion_series block as the single-method figure."""
    from examples.plant_residual_server import CP, LEARN_W

    sers = [_companion_series(m) for m in APP_METHODS]
    n = len(sers[0]["score"])
    x = np.arange(n)
    ref_band, test_band = (CP - APP_REF_N, CP), (CP, CP + APP_TEST_N)

    fig, (ax, axt) = plt.subplots(
        2,
        1,
        figsize=COMPANION_FIGSIZE,
        gridspec_kw={"height_ratios": [1.7, 1.0]},
    )
    for ser in sers:
        emph = ser["key"] == "toDMDc"
        ax.plot(
            x,
            ser["score"],
            color=APP_COLOR[ser["key"]],
            lw=2.6 if emph else 1.6,
            label=_app_lbl(ser["key"]),
            zorder=3 if emph else 2,
        )
    ax.axvspan(*ref_band, color=APP_REF, alpha=0.13)
    ax.axvspan(*test_band, color=APP_TEST, alpha=0.16)
    ax.axvline(CP, color=APP_PLANT, ls="--", lw=1.2)
    ax.text(
        CP,
        1.10,
        "fouling starts",
        color="#555555",
        ha="center",
        fontsize=13,
        transform=ax.get_xaxis_transform(),
    )
    ax.axhline(APP_THR, color=APP_THRESH, ls="--", lw=1.2)
    ax.set_xlim(LEARN_W, n)
    ax.set_ylabel("change score")
    ax.set_xlabel("sample")
    ax.xaxis.set_label_coords(0.5, -0.13)  # keep the label clear of the table
    # legend in a strip above the axes so no trace ever crosses it
    ax.legend(
        ncol=len(APP_METHODS),
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.06),
    )

    # app metrics table: model | MAR/e | delay | peak | FAR | AUC (renderTable())
    axt.axis("off")
    cols = ["model", "MAR/e", "delay", "peak", "FAR", "AUC"]

    def _cell(v: int | None) -> str:
        return "–" if v is None else str(v)

    cells = []
    for ser in sers:
        m = ser["metrics"]
        cells.append(
            [
                _app_lbl(ser["key"]),
                "✓" if m["detected"] else "✗",
                _cell(m["delayCross"]),
                _cell(m["delayPeak"]),
                f"{m['far'] * 100:.1f}%",
                "–" if not np.isfinite(m["auc"]) else f"{m['auc']:.3f}",
            ],
        )
    tab = axt.table(
        cellText=cells,
        colLabels=cols,
        loc="center",
        cellLoc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(13)
    tab.scale(1.0, 1.6)
    # best per column (app highlights best): min delay/peak/FAR, max AUC
    metrics = [s["metrics"] for s in sers]
    best = {
        2: min((m["delayCross"] for m in metrics if m["delayCross"] is not None), default=None),
        3: min((m["delayPeak"] for m in metrics if m["delayPeak"] is not None), default=None),
        4: min(m["far"] for m in metrics),
        5: max((m["auc"] for m in metrics if np.isfinite(m["auc"])), default=None),
    }
    for i, ser in enumerate(sers):
        tab[(i + 1, 0)].get_text().set_color(APP_COLOR[ser["key"]])
        tab[(i + 1, 0)].get_text().set_fontweight("bold")
        m = ser["metrics"]
        vals = {2: m["delayCross"], 3: m["delayPeak"], 4: m["far"], 5: m["auc"]}
        for c, b in best.items():
            if b is not None and vals[c] == b:
                tab[(i + 1, c)].get_text().set_fontweight("bold")
    # top=0.90 leaves room for the legend strip above the axes
    fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, top=0.90, bottom=0.04, hspace=0.42)
    _save(fig, "fig_twotank_companion_scores", tight=False)


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
        "A battery module heats up.\nFailing cooling — or the controller doing its job?",
        fontsize=16,
    )
    fig.tight_layout()
    _save(fig, "fig_cold_open")


if __name__ == "__main__":
    print("Rendering talk figures...")
    fig_cold_open()
    fig_bess("hx10", crop=True)
    fig_bess("hx10", crop=False)
    fig_bess("hx20", crop=True)
    fig_twotank()
    fig_twotank_companion()
    fig_twotank_companion_scores()
    print("Done.")
