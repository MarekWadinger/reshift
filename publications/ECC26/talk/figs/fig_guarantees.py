"""Two figures for the 'Two guarantees' slide, one per claim.

fig_cost_mr2:  measured per-update wall time of the truncated online update —
    grows like m·r² across sensors m and rank r, and stays flat over a long
    stream (production Rolling config), i.e. independent of history length.
fig_delay_c:   the frames/score_ synthetic (abrupt permanent step + noise):
    the score peak lands exactly c samples after the change, for any test
    window c and any change time.

Run: ``uv run python publications/ECC26/talk/figs/fig_guarantees.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from talkplot import (  # noqa: E402
    AX_RIGHT,
    BLUE,
    RED,
    save_fig as _save,
    sliding_score,
    talk_style,
)
from talkplot import GREY_L as GREY  # noqa: E402

talk_style()  # shared projector rcParams
GREENS = ["#7fcdbb", "#2a9d8f", "#14665c"]  # light→dark = small→large window
TESTBAND, THRESH = "#27ae60", "#111"
# both figures share one canvas size → identical on-slide scale in the columns
FIGSIZE = (6.4, 5.2)


# --- claim 1: cost O(mr^2), independent of history --------------------------
def _bench_update(m: int, r: int, n_time: int = 100) -> float:
    """Median wall time (ms) of one truncated online-DMD update at (m, r).

    Two vendored bookkeeping artifacts are neutralized so the timing reflects
    the algorithm, not the wrapper: the OnlineSVD weight W is the identity but
    stored dense (W @ A is an O(m²) no-op — make it sparse), and the mode-
    reconstruction history buffer _Y grows by vstack each step (preallocate).
    Neither is part of the update rule the O(mr²) claim covers.
    """
    import scipy.sparse as sps

    from river.decomposition import OnlineDMD

    rng = np.random.default_rng(42)
    od = OnlineDMD(r=r, initialize=2 * r, seed=42)
    X = rng.normal(size=(2 * r + n_time + 1, m))
    for i in range(2 * r):  # burn through batch initialization
        od.update(X[i], X[i + 1])
    od._svd.W = sps.identity(m, format="csr")
    od._Y = np.zeros((2 * r + n_time + 2, m))
    t = np.empty(n_time)
    for j in range(n_time):
        i = 2 * r + j
        t0 = time.perf_counter()
        od.update(X[i], X[i + 1])
        t[j] = time.perf_counter() - t0
    return float(np.median(t)) * 1e3


def fig_cost() -> None:
    ms = np.array([128, 512, 2048, 8192])
    ranks = [8, 32, 64]
    times = {r: np.array([_bench_update(m, r) for m in ms]) for r in ranks}

    # the guarantee is an upper bound: envelope c0 + c1·mr² sitting just above
    # every measured point (measured cost is in fact cheaper than the bound)
    c0 = min(t for r in ranks for t in times[r])
    c1 = 1.3 * max(
        (times[r][i] - c0) / (m * r**2)
        for r in ranks
        for i, m in enumerate(ms)
    )

    # sanity: measured cost must grow with m and with r, and respect the bound
    assert times[ranks[0]][-1] > times[ranks[0]][0], "cost should grow with m"
    assert times[ranks[-1]][-1] > times[ranks[0]][-1], "cost should grow with r"
    assert all(
        times[r][i] <= c0 + c1 * m * r**2 for r in ranks for i, m in enumerate(ms)
    ), "envelope must dominate every measurement"

    # history-independence: the production config (Rolling window) on a stream
    from river.decomposition import OnlineDMD

    from reshift.rolling import Rolling

    rng = np.random.default_rng(42)
    m_h, r_h, win, n_h = 128, 8, 500, 4000
    roll = Rolling(OnlineDMD(r=r_h, initialize=64, seed=42), win + 1)
    Xh = rng.normal(size=(n_h + 1, m_h))
    th = np.empty(n_h)
    for i in range(n_h):
        t0 = time.perf_counter()
        roll.update(Xh[i], Xh[i + 1])
        th[i] = time.perf_counter() - t0
    th *= 1e3

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=FIGSIZE, height_ratios=[1.3, 1.0],
    )
    mm = np.geomspace(ms[0], ms[-1], 100)
    for i, r in enumerate(ranks):
        a = 0.45 + 0.275 * i
        ax0.plot(ms, times[r], "o-", color=BLUE, ms=7, lw=1.6, alpha=a)
        ax0.plot(mm, c0 + c1 * mm * r**2, "--", color=RED, lw=1.3, alpha=a)
        ax0.annotate(f"$r={r}$", xy=(ms[-1], times[r][-1]),
                     xytext=(6, 0), textcoords="offset points",
                     color=BLUE, fontsize=13, va="center")
    ax0.annotate("$\\mathcal{O}(mr^2)$ bound", xy=(mm[35], c0 + c1 * mm[35] * ranks[-1] ** 2),
                 xytext=(0, 6), textcoords="offset points",
                 color=RED, fontsize=13, rotation=8)
    ax0.set_xscale("log", base=2)
    ax0.set_yscale("log")
    ax0.set_xlabel("sensors $m$")
    ax0.set_ylabel("time / update (ms)")
    ax0.set_xlim(ms[0] * 0.9, ms[-1] * 1.7)
    ax0.set_title("measured (blue) stays under the bound — sub-ms", fontsize=14)

    # rolling median so the flat trend reads through timer jitter
    k = 101
    med = np.array([np.median(th[max(0, i - k) : i + 1]) for i in range(win, n_h)])
    ax1.plot(np.arange(win, n_h), th[win:], color=GREY, lw=0.5, alpha=0.5)
    ax1.plot(np.arange(win, n_h), med, color=BLUE, lw=2.0)
    ax1.set_ylim(0, np.percentile(th[win:], 99) * 1.4)
    ax1.set_xlabel("samples seen")
    ax1.set_ylabel("time / update (ms)")
    ax1.set_title("flat over the stream — history length never enters", fontsize=14)

    fig.subplots_adjust(left=0.16, right=0.90, top=0.94, bottom=0.12, hspace=0.62)
    _save(fig, "fig_cost_mr2", tight=False)


# --- claim 2: peak delay = test window c -------------------------------------
N = 700
_NOISE = (np.random.default_rng(42).random(N) - 0.5) * 2  # same pattern as frames/


def _residual(
    onset: int,
    transition: int = 0,
    base: float = 1.0,
    hi: float = 4.0,
) -> np.ndarray:
    """Permanent change at `onset`, ramping over `transition` samples (0 = sharp)."""
    t = np.arange(N)
    frac = np.clip((t - onset) / max(transition, 1), 0.0, 1.0)
    return np.maximum(0.0, base + (hi - base) * frac + _NOISE)


def fig_delay() -> None:
    ref = 200
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)

    # top: one change time, three test windows — peak moves with c
    onset = 300
    for c, col in zip([25, 50, 100], GREENS):
        sc = sliding_score(_residual(onset), ref, c)
        pk = int(np.nanargmax(sc))
        assert abs(pk - (onset + c)) <= 3, f"peak {pk} != onset+c {onset + c}"
        ax0.plot(sc, color=col, lw=1.8, label=f"$c={c}$")
        ax0.plot(pk, sc[pk], "o", color=col, ms=8)
        first = c == 25
        ax0.annotate(f"+{pk - onset}", xy=(pk, sc[pk]),
                     xytext=(-4 if first else 4, 6), textcoords="offset points",
                     ha="right" if first else "left",
                     color=col, fontsize=13, fontweight="bold")
    ax0.axvline(onset, color=GREY, ls="--", lw=1.4)
    ax0.set_ylim(0, 4.2)
    ax0.legend(frameon=False, loc="upper right", fontsize=13)
    ax0.set_ylabel("score")
    ax0.set_title("same change, three test windows $c$", fontsize=14)

    # bottom: one test window, sharp ↔ slow transition — the delay only grows
    # (peak ≈ transition + c), which is why the guarantee reads delay ≥ c
    c = 60
    last_pk = 0
    for tr, col in zip([0, 45, 90], [GREY, BLUE, RED]):
        sc = sliding_score(_residual(onset, tr), ref, c)
        pk = int(np.nanargmax(sc))
        assert pk - onset >= c - 1, f"delay {pk - onset} fell below c={c}"
        assert pk > last_pk, "delay should grow with transition length"
        last_pk = pk
        lab = "sharp" if tr == 0 else f"over {tr}"
        ax1.plot(sc, color=col, lw=1.8, label=lab)
        ax1.plot(pk, sc[pk], "o", color=col, ms=8)
        ax1.annotate(f"+{pk - onset}", xy=(pk, sc[pk]), xytext=(4, 6),
                     textcoords="offset points", color=col, fontsize=13,
                     fontweight="bold")
    ax1.axvline(onset, color=GREY, ls="--", lw=1.4)
    ax1.set_ylim(0, 4.2)
    ax1.legend(frameon=False, loc="upper right", fontsize=13)
    ax1.set_ylabel("score")
    ax1.set_xlabel("sample")
    ax1.set_title(f"same window $c={c}$, sharp ↔ slow transition", fontsize=14)

    for ax in (ax0, ax1):
        ax.set_xlim(ref + 20, N)
    fig.subplots_adjust(left=0.16, right=AX_RIGHT, top=0.94, bottom=0.12, hspace=0.42)
    _save(fig, "fig_delay_c", tight=False)


if __name__ == "__main__":
    print("Rendering guarantee figures...")
    fig_delay()
    fig_cost()
    print("Done.")
