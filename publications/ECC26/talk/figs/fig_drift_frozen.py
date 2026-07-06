"""Slide 2 figure: static model vs adaptive model on a slowly clogging valve.

Reuses the companion two-tank plant and the static batch identifier from
``examples/plant_residual_server.py`` verbatim; the adaptive side is the same
DMDc model wrapped in the paper's ``Rolling`` window (full rank -- same model,
static vs adaptive is the ONLY difference). Only the outflow-coefficient
profile is swapped for a slow clog (aging) plus one real transient fault late
in the record.

Both detectors are commissioned identically (dashed vertical line): alarm when
the 30-sample mean residual exceeds mean + 4 sigma of its own nominal value.
The statistic is plotted normalised by that threshold, so the dashed line at 1
is the alarm line in both panels.

Static: alarms from sample ~460 onward (pure aging -> false alarms climb) and
the real fault at 700 arrives on an already-saturated statistic (hides).
Adaptive: quiet through the aging; its only threshold crossing is the fault.

Run: ``uv run python publications/ECC26/talk/figs/fig_drift_frozen.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import examples.plant_residual_server as prs
from reshift.rolling import Rolling

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from talkplot import AX_LEFT, AX_RIGHT, BLUE, RED, save_fig, talk_style  # noqa: E402
from talkplot import GREY_L as GREY  # noqa: E402

# --- talk style (same as make_talk_figs.py, smaller legend) -----------------
talk_style(**{"legend.fontsize": 14})

# --- aging profile: slow clog + one real transient fault --------------------
NOISE = 0.005
DRIFT_ON = 350  # aging starts (samples); LEARN_W=180 ends well before
CLOG_END = 0.75  # valve passes 25% less by end of record ("years" compressed)
FAULT_T, FAULT_W, FAULT_D = 700, 60, 0.15  # real transient fault on top (subtle)
MEAN_W = 30  # detector: rolling-mean window on the residual
N_SIGMA = 4  # commissioning threshold: nominal mean + 4 sigma


def _kmult_aging(t: np.ndarray, change: str, width: int | float) -> np.ndarray:  # noqa: ARG001
    """Outflow multiplier: linear clog to CLOG_END + a raised-cosine fault dip."""
    clog = 1.0 + (CLOG_END - 1.0) * np.clip(
        (t - DRIFT_ON * prs.DT) / (prs.T - DRIFT_ON * prs.DT), 0, 1,
    )
    u = np.clip((t - FAULT_T * prs.DT) / FAULT_W, 0.0, 1.0)
    return clog - FAULT_D * np.sin(np.pi * u) ** 2


def _residual_rolling(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Adaptive counterpart of prs._residual_batch: the SAME full-rank DMDc,
    re-identified on a rolling LEARN_W window (predict, then learn -- causal)."""
    from river.decomposition import OnlineDMDwC

    lw = prs.LEARN_W
    core = OnlineDMDwC(p=X.shape[1], q=U.shape[1], initialize=lw - 1, w=1.0, eig_rtol=None)
    mdl = Rolling(core, lw)
    cols = [f"x{i}" for i in range(X.shape[1])]
    r = np.full(len(X), np.nan)
    for t in range(1, len(X)):
        xk = {c: float(v) for c, v in zip(cols, X[t - 1], strict=True)}
        yk = {c: float(v) for c, v in zip(cols, X[t], strict=True)}
        try:
            A, B = core._reconstruct_AB()  # noqa: SLF001
            p = np.real(A) @ X[t - 1] + np.real(B) @ U[t - 1]
            r[t] = np.linalg.norm(X[t] - p)
        except (AttributeError, ValueError, TypeError):
            pass  # operator not available yet during warmup
        mdl.learn_one(xk, yk, {"u": float(U[t - 1, 0])})
    finite = np.isfinite(r)
    r[~finite] = r[finite][0]
    return r


def _rolling_mean(r: np.ndarray, w: int = MEAN_W) -> np.ndarray:
    out = np.full(len(r), np.nan)
    c = np.cumsum(np.insert(r, 0, 0.0))
    out[w - 1 :] = (c[w:] - c[:-w]) / w
    return out


def main() -> None:
    # ponytail: patch the private profile hook instead of duplicating the plant
    prs._kmult = _kmult_aging  # noqa: SLF001
    X, U, kprof = prs._simulate("permanent", 6.0, NOISE)
    r_frozen, _ = prs._residual_batch(X, U, control=True, rank=0)
    r_adapt = _residual_rolling(X, U)

    n, lw = len(X), prs.LEARN_W
    x = np.arange(n)
    # per-model commissioning: mean + N_SIGMA sigma of the day-one nominal
    # statistic; plot statistic / threshold so the alarm line is 1 in both panels
    scores = {}
    for key, r in (("frozen", r_frozen), ("adapt", r_adapt)):
        m = _rolling_mean(r)
        nom = m[lw + MEAN_W : DRIFT_ON]
        scores[key] = m / (nom.mean() + N_SIGMA * nom.std())
    s_f, s_a = scores["frozen"], scores["adapt"]

    fig, axs = plt.subplots(
        4,
        1,
        figsize=(12, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [0.55, 0.8, 1.0, 1.0]},
    )
    ax_k, ax_h, ax_f, ax_a = axs

    ax_k.plot(x, kprof, color=GREY, lw=1.8)
    ax_k.set_ylabel("valve\ncoeff.")
    ax_k.annotate(
        "real fault",
        xy=(FAULT_T + FAULT_W // 2, kprof[FAULT_T + FAULT_W // 2]),
        xytext=(FAULT_T + 115, kprof[FAULT_T + FAULT_W // 2] - 0.08),
        color=RED,
        fontsize=13,
        fontweight="bold",
        va="center",
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.4},
    )

    ax_h.plot(x, X[:, 0], color=GREY, lw=1.2, label="$h_1$")
    ax_h.plot(x, X[:, 1], color=GREY, lw=1.2, alpha=0.55, label="$h_2$")
    ax_h.set_ylabel("levels (m)")
    ax_h.legend(loc="upper left", ncol=2, frameon=False)

    for ax, s, col, lab in (
        (ax_f, s_f, RED, "static detector"),
        (ax_a, s_a, BLUE, "adaptive detector"),
    ):
        ax.plot(x, s, color=col, lw=1.5)
        ax.axhline(1.0, color="black", ls=":", lw=1.3)
        ax.set_ylabel("alarm\nstatistic")
        ax.text(
            0.008, 0.93, lab, transform=ax.transAxes, ha="left", va="top",
            color=col, fontsize=14, fontweight="bold",
        )
        ax.fill_between(x, 1.0, np.fmax(s, 1.0), color=col, alpha=0.25)
    vis = slice(lw + MEAN_W, n)  # plotted range; excludes warmup artifacts
    fmax, amax = np.nanmax(s_f[vis]), np.nanmax(s_a[vis])
    ax_f.set_ylim(0, fmax * 1.08)
    ax_a.set_ylim(0, max(amax * 1.35, 2.2))
    ax_f.text(lw + MEAN_W + 10, 1.0 + 0.11 * ax_f.get_ylim()[1],
              "alarm threshold (set at commissioning)",
              fontsize=11, color="black", ha="left")
    ax_a.text(lw + MEAN_W + 10, 1.24, "alarm threshold (set at commissioning)",
              fontsize=11, color="black", ha="left")

    first_fa = int(np.nanargmax(s_f[vis] > 1.0)) + lw + MEAN_W
    ax_f.annotate(
        "false alarms appear\n(no fault yet)",
        xy=(first_fa + 30, 1.25),
        xytext=(first_fa - 90, fmax * 0.48),
        color=RED,
        fontsize=14,
        fontweight="bold",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.6},
    )
    ax_f.annotate(
        "real fault in\nthe alarm flood",
        xy=(FAULT_T + FAULT_W // 2 + 25, np.nanmax(s_f[FAULT_T : FAULT_T + FAULT_W + 40])),
        xytext=(FAULT_T - 130, fmax * 0.86),
        color=RED,
        fontsize=14,
        fontweight="bold",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.6},
    )
    pk = FAULT_T + int(np.nanargmax(s_a[FAULT_T : FAULT_T + FAULT_W + 40]))
    ax_a.annotate(
        "real fault\nraises alarm",
        xy=(pk, s_a[pk]),
        xytext=(pk - 60, amax * 1.42),
        color=BLUE,
        fontsize=14,
        fontweight="bold",
        ha="center",
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.6},
    )
    ax_a.text(
        450, 0.28, "adapts — no false alarms",
        color=BLUE, fontsize=13, ha="center",
    )

    for ax in axs:
        ax.axvline(DRIFT_ON, color=GREY, ls="--", lw=1.2)
        ax.set_xlim(lw + MEAN_W, n)
    ax_a.set_xlabel("sample")
    # commissioning marker labelled under the x axis, clear of the panels
    ax_a.text(
        DRIFT_ON, -0.42, "commissioning day",
        transform=ax_a.get_xaxis_transform(), ha="center", va="top",
        color=GREY, fontsize=13,
    )
    fig.align_ylabels(axs)
    fig.subplots_adjust(
        left=AX_LEFT, right=AX_RIGHT, top=0.985, bottom=0.135, hspace=0.30,
    )
    save_fig(fig, "fig_drift_frozen", tight=False)

    # step-1 slide: same layout, adaptive panel dropped entirely; the shared
    # x-axis (ticks, xlabel, commissioning label) moves up to the static panel
    ax_a.set_visible(False)
    ax_f.tick_params(labelbottom=True)
    ax_f.set_xlabel("sample")
    ax_f.text(
        DRIFT_ON, -0.42, "commissioning day",
        transform=ax_f.get_xaxis_transform(), ha="center", va="top",
        color=GREY, fontsize=13,
    )
    save_fig(fig, "fig_drift_frozen_static", tight=False)

    # self-check: the story must hold numerically
    drift = slice(450, FAULT_T - 10)
    assert (s_f[drift] > 1).mean() > 0.8, "frozen must false-alarm through the drift"
    assert first_fa < 500, "frozen false alarms must start well before the fault"
    assert not (s_a[lw + MEAN_W : FAULT_T] > 1).any(), "adaptive must be quiet pre-fault"
    assert (s_a[FAULT_T : FAULT_T + FAULT_W + 40] > 1).any(), "adaptive must catch the fault"
    assert not (s_a[FAULT_T + FAULT_W + 60 :] > 1).any(), "adaptive must re-settle"
    print(
        f"ok: frozen FAR(drift)={(s_f[drift] > 1).mean():.0%}, first FA at {first_fa}; "
        f"adaptive fires only at the fault (peak {amax:.2f})",
    )


if __name__ == "__main__":
    main()
