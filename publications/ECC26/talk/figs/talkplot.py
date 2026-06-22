"""One reusable stacked-panel plot for every talk figure. Stop rewriting plots.

    panels([ {"label":"x", "y":X}, {"label":"score", "y":s, "annot":[(i,"text")]} ],
           phases=[(0,4000,"#2a9d8f","control"), (4000,6000,"#d1495b","fault")],
           title="...", save="figs/foo")

Each panel: dict with y (1d or 2d array), label (y-axis), optional color,
optional annot=list of (x, text) arrows. phases shade vertical bands across all
panels. save writes .png+.pdf. That's it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    },
)
BLUE, RED, GREY, GREEN, AMBER = (
    "#1f6fb2",
    "#d1495b",
    "#6b7280",
    "#2a9d8f",
    "#e9c46a",
)


def panels(panel_list, phases=None, title=None, save=None, figsize=None):
    k = len(panel_list)
    fig, axs = plt.subplots(
        k,
        1,
        figsize=figsize or (13, 1.8 * k + 1),
        sharex=True,
        squeeze=False,
    )
    axs = axs[:, 0]
    for a0, a1, col, _lab in phases or []:
        for ax in axs:
            ax.axvspan(a0, a1, color=col, alpha=0.08)
    for ax, p in zip(axs, panel_list, strict=True):
        y = np.asarray(p["y"])
        ax.plot(
            y,
            lw=p.get("lw", 1.2),
            color=p.get("color", BLUE) if y.ndim == 1 else None,
        )
        ax.set_ylabel(p["label"], fontsize=12)
        if p.get("hline") is not None:
            ax.axhline(p["hline"], color=GREY, ls=":", lw=1.2)
        for xi, txt in p.get("annot", []):
            yi = (
                float(np.asarray(y)[int(xi)])
                if y.ndim == 1
                else ax.get_ylim()[1] * 0.7
            )
            ax.annotate(
                txt,
                xy=(xi, yi),
                xytext=(xi, yi + abs(yi) * 0.4 + 0.5),
                color=RED,
                fontsize=11,
                ha="center",
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.5},
            )
    # phase labels on the top panel
    if phases:
        top = axs[0]
        yt = top.get_ylim()[1] * 0.86
        for a0, a1, col, lab in phases:
            if lab:
                top.text(
                    (a0 + a1) / 2,
                    yt,
                    lab,
                    ha="center",
                    color=col,
                    fontsize=11,
                    fontweight="bold",
                )
    if title:
        axs[0].set_title(title)
    axs[-1].set_xlabel("sample")
    fig.align_ylabels(axs)
    fig.tight_layout()
    if save:
        for ext in ("png", "pdf"):
            fig.savefig(f"{save}.{ext}", bbox_inches="tight")
        print(f"wrote {save}.png/.pdf")
    return fig, axs


if __name__ == "__main__":
    x = np.cumsum(np.random.default_rng(0).normal(0, 1, (400, 2)), 0)
    s = np.abs(np.random.default_rng(1).normal(0, 1, 400))
    s[200:210] += 5
    panels(
        [
            {"label": "signal", "y": x},
            {
                "label": "score $Q_k$",
                "y": s,
                "hline": 3,
                "annot": [(205, "change")],
            },
        ],
        phases=[(0, 200, GREEN, "normal"), (200, 400, RED, "fault")],
        title="talkplot self-check",
        save="/tmp/talkplot_selfcheck",
    )
    assert (np.asarray(s)[200:210] > 3).any(), "annot panel broken"
    print("ok")
