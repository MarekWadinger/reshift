"""oDMDc inverts the noise floor: coefficient blow-up on the two-tank.

Companion to the "The fix: online truncation" slide. Setup: the nonlinear
two-tank with a DUPLICATED tank-2 level sensor -- measurements [h1, h2, h2']
where h2' = h2 + noise(sigma). The duplicated channel adds no information,
so the data is nearly rank-deficient and the exact update inverts its tiny
noise floor:

  oDMDc   OnlineDMDwC(p=0, q=0)  ||[A|B]|| ~ 1/sigma (garbage coefficients);
                                  at sigma = 0 it refuses outright ("Failed
                                  rank(X) >= n_modes": singular X'X)
  toDMDc  OnlineDMDwC(p=2, q=1)  truncates before inverting: O(10)

Same streaming code (Rolling window), only the truncation flag differs.

Run: uv run python publications/ECC26/talk/figs/fig_truncation_blowup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fig_rotation_ellipse import K0, inflow  # noqa: E402
from talkplot import BLUE, GREY, RED, save_fig  # noqa: E402

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
    },
)

N = 2600  # stream length
W = 800  # rolling learning window
SIGMA = 1e-7  # noise of the duplicated sensor
P_TRUNC, Q_TRUNC = 2, 1  # toDMDc ranks; oDMDc uses p = q = 0


def simulate(sigma: float = SIGMA, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Two-tank with a duplicated tank-2 sensor: X = [h1, h2, h2']."""
    t = np.arange(0.0, float(N), 1.0)

    def f(tt: float, s: np.ndarray) -> list[float]:
        h1, h2 = np.clip(s, 0.0, 12.0)
        q = float(inflow(tt))
        return [q - K0 * np.sqrt(h1), K0 * np.sqrt(h1) - K0 * np.sqrt(h2)]

    sol = solve_ivp(
        f, (t[0], t[-1]), [4.8, 4.8], t_eval=t, rtol=1e-9, atol=1e-9
    )
    h = sol.y.T
    rng = np.random.default_rng(seed)
    # channels 2 and 3 measure the SAME level; they differ only by noise
    x = np.column_stack([h[:, 0], h[:, 1], h[:, 1]])
    x += rng.normal(0, sigma, x.shape)
    q = np.asarray(inflow(t)) + rng.normal(0, sigma / 2, len(t))
    return x, q


def stream(
    p: int, q_: int, X: np.ndarray, u: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Stream through Rolling(OnlineDMDwC); return per-step ||[A|B]|| trace.

    Trace is NaN until the window is full; the bool reports whether the
    update hard-failed (singular inversion on rank-deficient data).
    """
    from river.decomposition import OnlineDMDwC

    from reshift.rolling import Rolling

    core = OnlineDMDwC(p=p, q=q_, initialize=W - 1, w=1.0, seed=42)
    model = Rolling(core, W)
    cols = ["h1", "h2", "h2r"]
    norms = np.full(len(X), np.nan)
    for i in range(len(X)):
        try:
            model.update(dict(zip(cols, X[i])), u={"q": float(u[i])})
        except ValueError:  # singular X'X: the literal divide-by-zero
            return norms, True
        if i >= W:  # operator in the original sensor space
            A, B = core._reconstruct_AB()
            norms[i] = np.linalg.norm(np.column_stack([A, B]))
    return norms, False


def fig_terms() -> None:
    """Left: X = sum of rank-1 terms sigma_i u_i v_i^T on the duplicated-sensor
    two-tank -- terms 1-2 are the tank dynamics, term 3 the noise floor.
    Right: the model A drawn as a square whose parts carry the terms' colors:
    the blue block is built from the dynamics terms, the red row/column from
    the noise term. Truncation keeps the blue r x r block."""
    X, _ = simulate()
    Xw = X[:W].T  # the m x W window exactly as identification sees it
    U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
    flip = np.sign(U.sum(axis=0))  # SVD sign is arbitrary; make traces upright
    flip[flip == 0] = 1.0
    U, Vt = U * flip, Vt * flip[:, None]
    assert S[2] < 1e-4 * S[1], "third term should be the noise floor"

    t = np.arange(W)
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.7, 1.0], wspace=0.30, hspace=0.45)
    axs = [fig.add_subplot(gs[0, 0])]
    axs += [fig.add_subplot(gs[i, 0], sharex=axs[0]) for i in range(1, 4)]
    axs[0].plot(t, Xw.T, lw=0.9, color=GREY)
    axs[0].set_ylabel("data\n$[h_1, h_2, h_2']$")
    tags = ["dynamics", "dynamics", "noise"]
    term_colors = [BLUE, "#2a9d8f", RED]  # one colour per term
    for i in range(3):
        c = term_colors[i]
        # one trace per term: its time course sigma_i * v_i
        axs[i + 1].plot(t, S[i] * Vt[i], lw=1.1, color=c)
        axs[i + 1].set_ylabel(f"direction {i + 1}")
        axs[i + 1].annotate(
            tags[i],
            xy=(0.99, 0.78),
            xycoords="axes fraction",
            ha="right",
            color=c,
            fontsize=13,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    axs[3].set_xlabel("time step $k$")
    for ax in axs[:3]:
        ax.tick_params(labelbottom=False)
    fig.align_ylabels(axs)

    # right: A as a square. In the direction basis, column i of the operator
    # is built from term i (A U = X' V Sigma^{-1}, column-wise) — so each
    # column carries its term's colour. Truncation keeps the top-left r x r.
    axb = fig.add_subplot(gs[1:3, 1])
    axb.set_aspect("equal")
    axb.axis("off")
    for i in range(3):  # row (top = direction 1)
        for j in range(3):  # column j is built from term j
            axb.add_patch(
                plt.Rectangle(
                    (j, 2 - i),
                    0.94,
                    0.94,
                    facecolor=term_colors[j],
                    alpha=0.75,
                    edgecolor="white",
                    lw=1.5,
                ),
            )
    for k in range(3):  # term numbers along the top, coloured like the terms
        axb.text(k + 0.47, 3.12, str(k + 1), ha="center", fontsize=14,
                 fontweight="bold", color=term_colors[k])
    axb.add_patch(
        plt.Rectangle((0.02, 1.08), 1.9, 1.84, fill=False, ls="--", lw=2.2, edgecolor="black"),
    )
    axb.set_title(r"the model $\bar A$", fontsize=15)
    axb.text(1.47, -0.35, "column $i$ is built from direction $i$\ndashed: kept — rank $r$",
             ha="center", va="top", fontsize=12)
    axb.set_xlim(-0.4, 3.3)
    axb.set_ylim(-1.6, 3.6)

    # the sigmas as a bar plot below the square, matching colours (log scale)
    axh = fig.add_subplot(gs[3, 1])
    axh.bar([1, 2, 3], S, bottom=1e-8, color=term_colors, width=0.6)
    axh.set_yscale("log")
    axh.set_ylim(1e-8, 1e4)
    axh.set_yticks([1e-6, 1e-1, 1e4])
    axh.set_xticks([1, 2, 3])
    axh.set_ylabel(r"$\sigma_i$")
    axh.set_xlabel("direction $i$")
    fig.tight_layout()
    save_fig(fig, "fig_truncation_terms")


def main() -> None:
    X, q = simulate()
    tr_e, _ = stream(0, 0, X, q)  # oDMDc: no truncation
    tr_t, _ = stream(P_TRUNC, Q_TRUNC, X, q)  # toDMDc: r = 2

    # sigma = 0 (exactly duplicated sensor): the update refuses outright
    X0, q0 = simulate(sigma=0.0)
    crashed = stream(0, 0, X0, q0)[1]
    print(f"sigma=0: {'CRASH (singular X.T X)' if crashed else '?!'}")
    print(f"oDMDc  median {np.nanmedian(tr_e):.3g}")
    print(f"toDMDc median {np.nanmedian(tr_t):.3g}")

    # --- self-checks: the figure's claims hold on this run -------------------
    assert crashed, "oDMDc should refuse on exactly duplicated channels"
    assert np.nanmedian(tr_e) > 10 * np.nanmedian(tr_t), "oDMDc should blow up"
    assert np.nanmax(tr_t) < 100, "toDMDc should stay O(10)"

    # --- figure: one axes, two curves ------------------------------------------
    t = np.arange(N)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.semilogy(t, tr_e, color=RED, lw=1.5, label="oDMDc — no truncation")
    ax.semilogy(t, tr_t, color=BLUE, lw=1.8,
                label=f"toDMDc — rank $r={P_TRUNC}$ (ours)")
    ax.axhline(np.nanmedian(tr_e), color=RED, ls=":", lw=1.0, alpha=0.5)
    ax.axhline(np.nanmedian(tr_t), color=BLUE, ls=":", lw=1.0, alpha=0.5)
    ax.annotate(
        rf"$\times${np.nanmedian(tr_e) / np.nanmedian(tr_t):,.0f}",
        xy=(W + 40, np.sqrt(np.nanmedian(tr_e) * np.nanmedian(tr_t))),
        fontsize=15, color=GREY,
    )
    ax.set_xlim(W - 50, N)
    ax.set_xlabel("time step $k$")
    ax.set_ylabel(r"identified $\Vert[\tilde A_k\,|\,\tilde B_k]\Vert$")
    ax.set_title(
        "two-tank with a duplicated tank-2 sensor "
        r"($h_2' = h_2 + $ noise, $\sigma = 10^{-7}$)",
        fontsize=14,
    )
    ax.legend(loc="center right", frameon=False)
    ax.grid(True, which="both", alpha=0.2)

    save_fig(fig, "fig_truncation_blowup")


if __name__ == "__main__":
    fig_terms()
    main()
