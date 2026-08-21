"""What is DMD? — 30-second bridge slide for the control audience.

Bridges "Getting a solid model" (we need an interpretable linear model) to
"What is truncation?" (which SVD directions to keep). The point is the
MECHANISM — how DMD fits the model ON the directions:

  left  — steps 1+2: the streaming trajectory, the SVD directions of the
          data (direction 1 BLUE, direction 2 GREEN — the SAME colours the
          truncation slide uses for its kept directions), and each sample
          projected onto them.
  right — step 3: the projected coordinates z1, z2 over time. DMD is
          least-squares on THESE — fit z_k = A~ z_{k-1}. The model lives on
          the directions, so dropping a direction (truncation, next slide)
          just deletes a row and column of A~.

Naming per deck convention: step is k-1 -> k. Colors fixed: GREY = data,
BLUE/GREEN = SVD directions 1/2 (matches fig_truncation_terms).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from talkplot import BLUE, GREEN, GREY, save_fig, talk_style  # noqa: E402

DECAY, THETA = 0.965, np.deg2rad(11.0)  # |lambda|, rotation per step
N, SIGMA, SEED = 95, 0.07, 7  # noise big enough to SEE — it is the villain


def true_A() -> np.ndarray:
    R = np.array(
        [
            [np.cos(THETA), -np.sin(THETA)],
            [np.sin(THETA), np.cos(THETA)],
        ]
    )
    S = np.diag([1.0, 0.45])  # squash -> the SVD directions are distinct
    return S @ (DECAY * R) @ np.linalg.inv(S)


def simulate() -> np.ndarray:
    A = true_A()
    x = np.empty((N, 2))
    x[0] = [2.1, 0.15]
    for k in range(1, N):
        x[k] = A @ x[k - 1]
    return x + np.random.default_rng(SEED).normal(0, SIGMA, x.shape)


def main() -> None:
    talk_style(**{"axes.grid": False, "font.size": 15})
    X = simulate()
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T  # coordinates along the SVD directions

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(12.5, 5.2), width_ratios=[1, 1.25]
    )

    # ---- left: the data and its SVD directions ----------------------------
    axl.set_aspect("equal")
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_xlabel("$x_1$")
    axl.set_ylabel("$x_2$")
    lim = 1.12 * np.abs(Xc).max()
    axl.set_xlim(-lim, lim)
    axl.set_ylim(-lim, lim)

    axl.plot(*Xc.T, color=GREY, lw=1.0, alpha=0.6, zorder=1)
    axl.scatter(*Xc.T, s=22, color=GREY, alpha=0.55, zorder=2)
    for j, col in [(0, BLUE), (1, GREEN)]:
        d = 2.0 * (s[j] / np.sqrt(N)) * Vt[j]
        for sign in (1, -1):
            axl.annotate(
                "",
                xy=sign * d,
                xytext=(0, 0),
                arrowprops={"arrowstyle": "-|>", "color": col, "lw": 3.5},
                zorder=5,
            )
    axl.annotate(
        "direction 1",
        xy=(-1.65, -0.62),
        color=BLUE,
        fontsize=14,
        fontweight="bold",
    )
    axl.annotate(
        "direction 2",
        xy=(0.25, 0.72),
        color=GREEN,
        fontsize=14,
        fontweight="bold",
    )
    axl.set_title(
        "① SVD: the directions the data moves in\n"
        "② project each sample onto them",
        color=BLUE,
        fontweight="bold",
    )

    # ---- right: project on the directions, fit the step there -------------
    axr.set_xlabel("sample $k$")
    axr.set_ylabel("coordinate along direction")
    axr.plot(Z[:, 0], color=BLUE, lw=2.2, label="$z_1$ (direction 1)")
    axr.plot(Z[:, 1], color=GREEN, lw=2.2, label="$z_2$ (direction 2)")
    axr.legend(loc="upper right", frameon=False)

    # the fit: least squares on the projected coordinates
    A_hat, *_ = np.linalg.lstsq(Z[:-1], Z[1:], rcond=None)
    A_hat = A_hat.T
    # show the one-step map on the coordinates: from (k-1) predict (k)
    for k in (18, 36, 54):
        zpred = A_hat @ Z[k - 1]
        for j, col in [(0, BLUE), (1, GREEN)]:
            axr.annotate(
                "",
                xy=(k, zpred[j]),
                xytext=(k - 1, Z[k - 1, j]),
                arrowprops={"arrowstyle": "-|>", "color": GREY, "lw": 2.5},
                zorder=4,
            )
        axr.plot(
            [k] * 2, zpred, "o", ms=6, color=GREY, zorder=4, mfc="white"
        )
    axr.annotate(
        "least squares on these:\n$z_k = \\tilde{A}\\, z_{k-1}$",
        xy=(36, Z[35, 0]),
        xytext=(40, -1.75),
        color=GREY,
        fontsize=16,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.4},
    )
    axr.set_title(
        "③ DMD: fit the one-step model\non those coordinates",
        color=BLUE,
        fontweight="bold",
    )
    # the seed for the truncation slides: once the dynamics decay, what is
    # left on every direction is pure sensor noise
    axr.annotate(
        "dynamics gone —\nonly noise left",
        xy=(84, Z[84, 1]),
        xytext=(70, -1.05),
        color=GREY,
        fontsize=14,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": GREY, "lw": 1.4},
    )

    fig.tight_layout()
    save_fig(fig, "fig_what_is_dmd")

    # self-check: the reduced fit recovers the decaying rotation
    lam = np.linalg.eigvals(A_hat)[0]
    lam_true = np.linalg.eigvals(true_A())[0]
    assert abs(abs(lam) - abs(lam_true)) < 0.05, "|lambda| off"
    assert abs(abs(np.angle(lam)) - THETA) < np.deg2rad(2.5), "angle off"
    print(
        f"ok  |lambda|={abs(lam):.3f}  "
        f"angle={np.degrees(abs(np.angle(lam))):.1f} deg"
    )


if __name__ == "__main__":
    main()
