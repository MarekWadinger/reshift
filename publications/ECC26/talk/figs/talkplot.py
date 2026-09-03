"""THE shared toolkit for every ECC26 talk figure. Stop rewriting plots.

Import from here instead of re-declaring helpers in a new fig_*.py:

    panels(...)        stacked-panel plotter (see below)
    save_fig(fig, "fig_foo")            png+pdf save loop; tight=False keeps
                                        the full fixed canvas (equal on-slide
                                        scale across figures)
    talk_style()       projector rcParams (font.size 17 family); pass
                       overrides, e.g. talk_style(**{"legend.fontsize": 14})
    sliding_score(r, ref, test)         max(mean(test)/mean(ref) - 1, 0) over
                                        sliding windows (window_explorer.html)
    zoom_inset(ax, rect)                white-framed inset for sub-percent zooms
    BLUE RED GREY GREY_L GREEN AMBER    fixed palette; TWO greys on purpose:
                                        GREY dark (#6b7280), GREY_L light
                                        (#9aa0a6, make_talk_figs family) —
                                        never unify, it shifts pixels
    AX_LEFT, AX_RIGHT                   shared plot extent so x-axes line up
                                        across slides

panels() usage:

    panels([ {"label":"x", "y":X}, {"label":"score", "y":s, "annot":[(i,"text")]} ],
           phases=[(0,4000,"#2a9d8f","control"), (4000,6000,"#d1495b","fault")],
           title="...", save="figs/foo")

Each panel: dict with y (1d or 2d array), label (y-axis), optional color,
optional annot=list of (x, text) arrows. phases shade vertical bands across all
panels. save writes .png+.pdf. That's it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

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
GREY_L = "#9aa0a6"  # light grey of the make_talk_figs family; distinct from GREY
# shared plot-area left/right (figure fraction) so every data plot spans the
# SAME horizontal extent on its slide — only thing that makes the x-axes line up
AX_LEFT, AX_RIGHT = 0.11, 0.965


def talk_style(**overrides: object) -> None:
    """Projector rcParams (big, clean, high-contrast). Call after imports."""
    style: dict[str, object] = {
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
    }
    style.update(overrides)
    # matplotlib types rcParams keys as a closed Literal set, which a
    # str-keyed dict cannot satisfy; runtime validation happens in RcParams.
    plt.rcParams.update(style)  # ty: ignore[no-matching-overload]


def save_fig(
    fig: plt.Figure,
    name: str,
    *,
    tight: bool = True,
    outdir: Path = HERE,
) -> None:
    # tight=False keeps the full fixed canvas (so two figures of equal figsize
    # render at the same scale on a slide — identical font sizes).
    kw = {"bbox_inches": "tight"} if tight else {}
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def sliding_score(
    r: np.ndarray,
    ref: int,
    test: int,
    *,
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


def zoom_inset(
    ax: plt.Axes, rect: tuple[float, float, float, float]
) -> plt.Axes:
    """White framed inset for zooming on sub-percent differences."""
    axz = ax.inset_axes(rect)
    axz.set_xticks([])
    axz.set_yticks([])
    axz.grid(False)
    axz.set_facecolor("white")
    for sp in axz.spines.values():
        sp.set_visible(True)
        sp.set_color(GREY)
    return axz


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
