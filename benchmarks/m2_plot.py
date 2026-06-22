"""Diagnostic plots for the M2 benchmark on the real synthetic files.

Captures toDMDc and OSVD-CPD score traces on a published realization and plots
signal + overlaid scores + calibrated thresholds + detections + change-points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import benchmarks.m2_cpd_metrics as m


def capture_synthetic(idx: int) -> dict:
    """Run both detectors on real y{idx}.txt, returning signal + score traces."""
    x, cps = m.load_real_synthetic(idx)
    n = x.shape[0]
    p_dmd, _s_dmd, p_svd, _s_svd = m.build_synthetic_pipelines()
    sc_dmd = np.zeros(n)
    sc_svd = np.zeros(n)
    for i, xi in enumerate(pd.DataFrame(x).to_dict(orient="records")):
        sc_dmd[i] = p_dmd.score_one(xi)
        p_dmd.learn_one(xi)
        sc_svd[i] = p_svd.score_one(xi)
        p_svd.learn_one(xi)
    thr_dmd = m.calibrated_threshold(sc_dmd, m.SYN_GRACE, int(cps.min()))
    thr_svd = m.calibrated_threshold(sc_svd, m.SYN_GRACE, int(cps.min()))
    return {
        "x": x.ravel(),
        "cps": cps,
        "sc_dmd": sc_dmd,
        "sc_svd": sc_svd,
        "thr_dmd": thr_dmd,
        "thr_svd": thr_svd,
    }


def plot_one(idx: int, out: Path) -> None:
    """Two-panel signal+scores plot for one realization."""
    d = capture_synthetic(idx)
    t = np.arange(len(d["x"]))
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        height_ratios=[1, 1.4],
    )
    ax0.plot(t, d["x"], lw=0.5, color="0.3")
    ax0.set_ylabel(f"signal y{idx}")
    for cp in d["cps"]:
        for ax in (ax0, ax1):
            ax.axvline(cp, color="tab:red", ls="--", lw=0.8, alpha=0.7)
    ax1.plot(t, d["sc_dmd"], lw=0.7, color="tab:blue", label="toDMDc score")
    ax1.plot(
        t,
        d["sc_svd"],
        lw=0.7,
        color="tab:orange",
        ls=":",
        label="OSVD-CPD score",
    )
    ax1.axhline(
        d["thr_dmd"],
        color="tab:blue",
        ls="-.",
        lw=0.8,
        alpha=0.6,
        label=f"toDMDc thr={d['thr_dmd']:.3f}",
    )
    ax1.set_ylabel("change score")
    ax1.set_xlabel("sample")
    ax1.set_ylim(-0.05, max(1.0, float(np.nanpercentile(d["sc_dmd"], 99.5))))
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax0.set_title(
        f"Real synthetic y{idx}: toDMDc vs OSVD-CPD "
        "(red dashed = true change-points)",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    outdir = Path(__file__).resolve().parent / "m2_plots"
    outdir.mkdir(exist_ok=True)
    for i in range(m.SYN_N_FILES):
        plot_one(i, outdir / f"synthetic_y{i}.png")
