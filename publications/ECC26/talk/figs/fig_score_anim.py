"""Frame sequence for the 'model turns change into a number' slide.

Permanent abrupt change. Three back-to-back parameter sweeps, each starting from
where the previous stopped (held params carry over), mirroring the knobs in
examples/window_explorer.html:

  1. noise   0 -> 1      (windows tiny: score is swamped by noise)
  2. ref     5 -> 200    (a longer baseline cancels the noise)
  3. test    5 -> 200    (a longer test window steadies the peak)

Each frame: residual (with the true change marked) over the score, plus a readout
of the current knob values. Embed with \\animategraphics (see slides.tex).

    uv run python fig_score_anim.py    # writes frames/score_0000.png ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the talk's shared rcParams (font sizes) and plot margins so every frame
# matches the static figures' width and fonts on the slide.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from talkplot import AX_LEFT, AX_RIGHT, sliding_score, talk_style  # noqa: E402

talk_style()

N, BASE, CENTER, HI = 600, 1.0, 300, 4.0
THR, LAG = 0.25, 0
SCORE_CEIL = 3.5  # display ceiling: above the real peak, below the noise spikes
# Colours straight from examples/window_explorer.html: one per series for BOTH
# residual and score (synthetic = blue), the window band shades, threshold, plant.
SYN, GREY, THRESH = "#1f77b4", "#999", "#111"
REFBAND, LAGBAND, TESTBAND = "#2f86d6", "#94a3b8", "#27ae60"

# One fixed noise pattern, scaled by amplitude per frame so the picture animates
# smoothly instead of flickering with a fresh draw each step.
_NOISE = (np.random.default_rng(42).random(N) - 0.5) * 2
_CLEAN = np.where(np.arange(N) >= CENTER, HI, BASE).astype(float)


def residual(noise: float) -> np.ndarray:
    return np.maximum(0.0, _CLEAN + _NOISE * noise)


def scores(r: np.ndarray, ref: int, test: int) -> np.ndarray:
    """score[t] = max(mean(test)/mean(ref) - 1, 0), windows right-aligned at t."""
    return sliding_score(r, ref, test, lag=LAG)


def frames() -> list[tuple[float, int, int]]:
    """(noise, ref, test) per frame across the sweeps; each holds where the last
    stopped, then the windows and noise wind back down toward the clean case."""
    seq: list[tuple[float, int, int]] = []
    for noise in np.linspace(0, 1, 40):  # 1: noise up      (ref=test=5)
        seq.append((float(noise), 5, 5))
    for ref in np.linspace(5, 200, 45):  # 2: ref up         (noise=1, test=5)
        seq.append((1.0, int(round(ref)), 5))
    for test in np.linspace(5, 100, 45):  # 3: test up        (noise=1, ref=200)
        seq.append((1.0, 200, int(round(test))))
    for ref in np.linspace(200, 5, 45):  # 4: ref back down  (noise=1, test=100)
        seq.append((1.0, int(round(ref)), 100))
    for noise in np.linspace(1, 0, 40):  # 5: noise back down(ref=5, test=100)
        seq.append((float(noise), 5, 100))
    return seq


def main() -> None:
    out = Path(__file__).parent / "frames"
    out.mkdir(exist_ok=True)
    seq = frames()
    for i, (noise, ref, test) in enumerate(seq):
        r = residual(noise)
        sc = scores(r, ref, test)
        fig, (ax_r, ax_s) = plt.subplots(
            2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[2, 1],
        )
        # Window bands anchored at the change like the HTML default (test window's
        # left edge sits on the onset; reference + lag immediately before it).
        for ax in (ax_r, ax_s):
            ax.axvspan(CENTER - LAG - ref, CENTER - LAG, color=REFBAND, alpha=0.20)
            if LAG:
                ax.axvspan(CENTER - LAG, CENTER, color=LAGBAND, alpha=0.22)
            ax.axvspan(CENTER, CENTER + test, color=TESTBAND, alpha=0.22)
            ax.axvline(CENTER, color=GREY, ls="--", lw=1.2)  # true change
        # residual and score share the series colour (synthetic = blue)
        ax_r.plot(r, lw=1.0, color=SYN)
        ax_r.set_ylabel("residual")
        ax_r.set_ylim(-0.2, HI + 1.3)
        # Readout doubles as the legend: ref label in the blue band colour, test
        # in the green, so the shaded windows are unambiguous.
        tkw = {"transform": ax_r.transAxes, "va": "top", "fontsize": 15,
               "family": "monospace", "fontweight": "bold",
               "bbox": {"boxstyle": "round,pad=0.2", "fc": "white",
                        "ec": "none", "alpha": 0.7}}
        ax_r.text(0.015, 0.94, f"noise {noise:.2f}", color=GREY, **tkw)
        ax_r.text(0.27, 0.94, f"reference {ref}", color=REFBAND, **tkw)
        ax_r.text(0.55, 0.94, f"test {test}", color=TESTBAND, **tkw)
        # Clip the displayed score to a ceiling just above the real step peak
        # (~3.24): tiny-window noise spikes (D_ref near zero) would otherwise shoot
        # to ~18 and ram the top edge. Flat-topping them keeps the line on screen.
        ax_s.plot(np.minimum(sc, SCORE_CEIL), lw=1.4, color=SYN)
        ax_s.axhline(THR, color=THRESH, ls="--", lw=1.0)  # threshold
        ax_s.set_ylim(0, SCORE_CEIL + 0.2)
        ax_s.set_ylabel("score")
        ax_s.set_xlabel("sample")
        for ax in (ax_r, ax_s):
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        # shared left/right margins → frames span the same plot extent as the
        # static figures (full canvas, no tight crop, so width is exactly figsize).
        fig.subplots_adjust(left=AX_LEFT, right=AX_RIGHT, top=0.985, bottom=0.11, hspace=0.12)
        # Unpadded index: \animategraphics builds names as score_0, score_1, ...
        fig.savefig(out / f"score_{i}.png", dpi=130)
        plt.close(fig)

    print(f"wrote {len(seq)} frames to {out}/score_*.png")
    # larger windows must tame the noise the tiny-window frame could not, over a
    # pre-change stretch both window sizes can actually score (sample 250..300).
    seg = slice(250, CENTER)
    noisy = np.nanstd(scores(residual(1.0), 5, 5)[seg])
    calm = np.nanstd(scores(residual(1.0), 120, 120)[seg])
    assert calm < noisy, "growing the windows should reduce pre-change score jitter"


if __name__ == "__main__":
    main()
