"""How the basis rotation K = U_k^T U_{k-1} propagates into A and B (DMDc).

Nonlinear two-tank plant (example 13) driven by a two-tone inflow q (two
frequencies so B is identifiable); valve k1 clogs by 1 % at t = T/2.
Control-aware fit on Hankel-lifted, per-window-centered data at rank 2:

    z_{t+s} = A z_t + B q_t,   z = W @ U  (reduced coordinates)

Naming, consistent everywhere (GREY = k-1/before, RED = k/after,
BLUE = realigned):
    data before (k-1) / data after (k)
    U_{k-1} / U_k                        the two frames
    A_{k-1}, B_{k-1}                     fit on before-data, U_{k-1} frame
    Kpp A_{k-1} Kpp^T, Kpp B_{k-1}       rotated into the U_k frame
                                         (odmd.py: A = _UU @ A @ _UU.T;
                                          B Kqq with scalar q -> Kqq = 1)
    A_k, B_k                             fit on after-data, U_k frame

Panels: (1) levels; (2) eigenvalues of A -- similarity transform keeps them,
the clog moves them (it sits in A: tank-1 pole tau = 2 sqrt(h)/k1);
(3) data cloud with both frames; (4) input map B -- the rotation visibly
realigns B's direction, the leftover ~1 % magnitude gap is the physics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from reshift.preprocessing import hankel  # noqa: E402

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# talkplot's import applies the shared size-13 rcParams this figure uses
from talkplot import BLUE, GREY, RED, save_fig, zoom_inset  # noqa: E402

T, DT = 4000.0, 1.0
K0 = 0.05  # healthy outflow coefficient
CLOG = 0.99  # valve k1 clogs by 1 %
U0, UA = 0.11, 0.025  # inflow mean / modulation amplitude
W1, W2 = 0.10, 0.023  # two-tone inflow frequencies (B identifiable)
SIGMA = 0.01  # state measurement noise std
Q_SIGMA = 0.005  # inflow reading noise std
STRIDE = 20  # s-step operator (1-step A ~ I shows nothing)
WINDOW = 800


def inflow(t: np.ndarray | float) -> np.ndarray:
    """Exogenous inflow q(t), two-tone so the input is persistently exciting."""
    return U0 + UA * np.sin(W1 * np.asarray(t)) + 0.5 * UA * np.sin(
        W2 * np.asarray(t)
    )


def simulate(
    seed: int = 42, total: float = T
) -> tuple[np.ndarray, np.ndarray, int]:
    """Integrate the two-tank; k1 clogs to CLOG*K0 at cp. Returns (X, q, cp)."""
    t = np.arange(0.0, total, DT)
    cp = len(t) // 2

    def rhs(k1: float, k2: float):  # noqa: ANN202
        def f(tt: float, s: np.ndarray) -> list[float]:
            h1, h2 = np.clip(s, 0.0, 12.0)
            q = float(inflow(tt))
            return [q - k1 * np.sqrt(h1), k1 * np.sqrt(h1) - k2 * np.sqrt(h2)]

        return f

    kw = {"rtol": 1e-9, "atol": 1e-9}
    pre = solve_ivp(rhs(K0, K0), (t[0], t[cp]), [4.8, 4.8], t_eval=t[:cp], **kw)
    post = solve_ivp(
        rhs(CLOG * K0, K0), (t[cp], t[-1]), pre.y[:, -1], t_eval=t[cp:], **kw
    )
    x = np.hstack([pre.y, post.y]).T
    rng = np.random.default_rng(seed)
    x += rng.normal(0, SIGMA, x.shape)
    q = inflow(t) + rng.normal(0, Q_SIGMA, len(t))
    return x, q, cp


def fit_basis(W: np.ndarray) -> np.ndarray:
    """Rank-2 left singular basis of a centered snapshot window."""
    return np.linalg.svd(W.T, full_matrices=False)[0][:, :2]


def fit_AB(
    Z: np.ndarray, u: np.ndarray, s: int = STRIDE
) -> tuple[np.ndarray, np.ndarray]:
    """DMDc: z_{t+s} = A z_t + B u_t on reduced snapshots Z (n, 2)."""
    R = np.column_stack([Z[:-s], u[:-s]])
    Th, *_ = np.linalg.lstsq(R, Z[s:], rcond=None)
    return Th[:2].T, Th[2]


def circle(A: np.ndarray, n: int = 400) -> np.ndarray:
    """Image of the unit circle under A."""
    t = np.linspace(0, 2 * np.pi, n)
    return A @ np.vstack([np.cos(t), np.sin(t)])


def main() -> None:
    X, q, cp = simulate()
    Xh = np.asarray(hankel(X, 10, 1))  # embed: m = 20
    qh = q[: len(Xh)]

    # two data windows (each centered on its own mean)
    a_old, b_old = cp - WINDOW, cp  # before (k-1)
    a_new, b_new = cp + 150, cp + 150 + WINDOW  # after (k)
    W_old = Xh[a_old:b_old] - Xh[a_old:b_old].mean(0)
    W_new = Xh[a_new:b_new] - Xh[a_new:b_new].mean(0)
    q_old = qh[a_old:b_old] - qh[a_old:b_old].mean()
    q_new = qh[a_new:b_new] - qh[a_new:b_new].mean()

    # two frames: U_old = U_{k-1} (from before-data), U_new = U_k (after-data)
    U_old, U_new = fit_basis(W_old), fit_basis(W_new)
    for j in range(2):  # sign-fix (SVD sign ambiguity)
        if U_new[:, j] @ U_old[:, j] < 0:
            U_new[:, j] *= -1

    # K: rotation U_{k-1} frame -> U_k frame (the slide's K_pp;
    # odmd.py: _UU = _U.T @ U_prev)
    K = U_new.T @ U_old

    # operators (z_{t+s} = A z_t + B q_t):
    # A_old, B_old = A_{k-1}, B_{k-1}: before-dynamics, U_{k-1} frame
    # A_re,  B_re : the same before-dynamics rotated into the U_k frame
    #               (A = Kpp A Kpp^T; B = Kpp B Kqq, scalar q -> Kqq = 1)
    # A_new, B_new = A_k, B_k: after-dynamics, U_k frame
    A_old, B_old = fit_AB(W_old @ U_old, q_old)
    A_re, B_re = K @ A_old @ K.T, K @ B_old
    A_new, B_new = fit_AB(W_new @ U_new, q_new)
    theta = np.degrees(np.arctan2(K[1, 0], K[0, 0]))

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2)
    # panel 1 is split: levels x on top, input u below, shared x axis
    sub = gs[0, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.10)
    ax0 = fig.add_subplot(sub[0])
    ax0u = fig.add_subplot(sub[1], sharex=ax0)
    axe = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    # keep the boxes the same size; "datalim" adapts the limits instead
    # of shrinking the box when the aspect is locked to equal
    # (axe/ax2 exempt: |B| ~ 40 vs differences ~ 1 would squash them flat)
    ax1.set_aspect("equal", adjustable="datalim")

    # --- (1) levels x (top) and inflow u (bottom) on a shared time axis -----
    t0, t1 = a_old - 400, b_new + 400
    for seg, col in [(slice(t0, cp), GREY), (slice(cp, t1), RED)]:
        ax0.plot(range(seg.start, seg.stop), X[seg, 0], color=col, lw=0.9)
        ax0.plot(
            range(seg.start, seg.stop), X[seg, 1], color=col, lw=0.9, ls="--"
        )
        ax0u.plot(range(seg.start, seg.stop), q[seg], color=col, lw=0.8)
    for ax in (ax0, ax0u):
        ax.axvspan(a_old, b_old, color=GREY, alpha=0.15)
        ax.axvspan(a_new, b_new, color=RED, alpha=0.10)
        ax.axvline(cp, color=RED, ls=":", lw=1.5)
    ax0.plot([], [], color="k", lw=0.9, label="$h_1$")
    ax0.plot([], [], color="k", lw=0.9, ls="--", label="$h_2$")
    ax0.set_title("two-tank levels: valve $k_1$ clogs by 1 %", fontsize=12)
    ax0.set_ylabel("level")
    ax0.tick_params(labelbottom=False)
    ax0.legend(loc="lower left", fontsize=10)
    ax0u.set_ylabel("$q$")
    ax0u.set_xlabel("sample")

    # --- (2) augmented operator A-bar = [A|B]: the slide's object -----------
    # M maps the unit sphere of (z1, z2, q) to an ellipse in the U_k frame:
    # M_old ([A|B]_{k-1}: U_{k-1}-frame matrix plotted unchanged),
    # M_re (realigned: [Kpp A Kpp^T | Kpp B]), M_new ([A|B]_k, U_k frame)
    M_old = np.column_stack([A_old, B_old])
    M_re = np.column_stack([A_re, B_re])
    M_new = np.column_stack([A_new, B_new])

    def ell(M: np.ndarray) -> np.ndarray:
        """Boundary of the unit-sphere image of a 2x3 map."""
        u, s, _ = np.linalg.svd(M, full_matrices=False)
        return (u * s) @ circle(np.eye(2))

    E_old, E_re, E_new = ell(M_old), ell(M_re), ell(M_new)
    axe.plot(
        *E_old, color=GREY, lw=2, ls="--",
        label=r"$\bar A_{k-1}$ (before)",
    )
    axe.plot(*E_re, color=BLUE, lw=2.5, label=r"$\bar A_{k-1}$ realigned")
    axe.plot(*E_new, color=RED, lw=2, ls="-.", label=r"$\bar A_k$ (after)")
    axe.set_title(
        r"augmented $\bar A=[\tilde A\,|\,\tilde B]$: realigned tracks after",
        fontsize=12,
    )
    axe.set_xlabel("mode 1")
    axe.set_ylabel("mode 2")
    axe.legend(loc="lower left", fontsize=9)
    # zoom on the long-axis tip, where the tracking is obvious
    tip_new = np.linalg.svd(M_new, full_matrices=False)[0][:, 0]
    sv_new = np.linalg.svd(M_new, compute_uv=False)[0]
    if tip_new[0] > 0:  # pick the left tip consistently
        tip_new = -tip_new
    tips = []
    for E in (E_old, E_re, E_new):
        j = np.argmin(np.linalg.norm(E - (sv_new * tip_new)[:, None], axis=0))
        tips.append(E[:, j])
    tips = np.stack(tips)
    cxa, cya = tips.mean(0)
    rza = 1.4 * np.abs(tips - tips.mean(0)).max() + 1e-3
    axz = zoom_inset(axe, (0.66, 0.66, 0.32, 0.32))
    axz.plot(*E_old, color=GREY, lw=2, ls="--")
    axz.plot(*E_re, color=BLUE, lw=2.5)
    axz.plot(*E_new, color=RED, lw=2, ls="-.")
    axz.set_xlim(cxa - rza, cxa + rza)
    axz.set_ylim(cya - rza, cya + rza)
    axe.indicate_inset_zoom(axz, edgecolor=GREY)

    # --- (3) data cloud with both frames ------------------------------------
    # everything drawn in the U_k frame: both clouds projected on U_new;
    # the U_k axes are (1,0),(0,1); the U_{k-1} axes are the columns of K
    Z_old, Z_new = W_old @ U_new, W_new @ U_new
    ax1.scatter(
        *Z_old.T, s=4, color=GREY, alpha=0.25, label="data before ($k{-}1$)"
    )
    ax1.scatter(
        *Z_new.T, s=4, color=RED, alpha=0.25, label="data after ($k$)"
    )
    L = 0.9 * np.abs(Z_old).max()
    ax1.plot([0, L], [0, 0], color=BLUE, lw=2, label="$U_k$")
    ax1.plot([0, 0], [0, L], color=BLUE, lw=2)
    K2 = L * K
    ax1.plot(
        [0, K2[0, 0]], [0, K2[1, 0]], color=GREY, ls="--", lw=2,
        label="$U_{k-1}$",
    )
    ax1.plot([0, K2[0, 1]], [0, K2[1, 1]], color=GREY, ls="--", lw=2)
    ax1.set_title(
        rf"$K_{{pp}}=U_k^\top U_{{k-1}}$: basis rotates by "
        rf"$\theta\approx{theta:.2f}^\circ$",
        fontsize=12,
    )
    ax1.set_xlabel("mode 1")
    ax1.set_ylabel("mode 2")
    ax1.legend(loc="upper left", fontsize=9)
    ax1z = zoom_inset(ax1, (0.66, 0.66, 0.32, 0.32))
    ax1z.plot([0, L], [0, 0], color=BLUE, lw=2)
    ax1z.plot([0, K2[0, 0]], [0, K2[1, 0]], color=GREY, ls="--", lw=2)
    rz1 = 2.5 * abs(K2[1, 0]) + 0.01
    ax1z.set_xlim(L - rz1, L + rz1)
    ax1z.set_ylim(-rz1, rz1)
    ax1.indicate_inset_zoom(ax1z, edgecolor=GREY)

    # --- (4) input map B: rotation realigns its direction -------------------
    # B is a 2-vector (scalar input); all three drawn in U_k frame coords:
    # B_old (U_{k-1}-frame vector plotted unchanged -> misaligned),
    # B_re = Kpp B_old (rotated into U_k frame), B_new = B_k (fit after)
    for B, c, ls, lab in [
        (B_old, GREY, "--", r"$\tilde B_{k-1}$ (before)"),
        (B_re, BLUE, "-", r"$K_{pp}\tilde B_{k-1}K_{qq}$"),
        (B_new, RED, "-.", r"$\tilde B_k$ (after)"),
    ]:
        ax2.plot([0, B[0]], [0, B[1]], color=c, ls=ls, lw=2, label=lab)
    ax2.set_title(
        "input map $\\tilde B$: rotation realigns its direction",
        fontsize=12,
    )
    ax2.set_xlabel("mode 1")
    ax2.set_ylabel("mode 2")
    ax2.legend(loc="upper left", fontsize=9)
    # zoom on the arrow tips, centered on their centroid
    tips = np.stack([B_old, B_re, B_new])
    cx, cy = tips.mean(0)
    rz2 = 1.4 * np.abs(tips - tips.mean(0)).max() + 1e-3
    ax2z = zoom_inset(ax2, (0.66, 0.66, 0.32, 0.32))
    for B, c, ls in [(B_old, GREY, "--"), (B_re, BLUE, "-"), (B_new, RED, "-.")]:
        ax2z.plot([0, B[0]], [0, B[1]], color=c, ls=ls, lw=2)
        ax2z.plot(*B, marker="o", ms=5, color=c)
    ax2z.set_xlim(cx - rz2, cx + rz2)
    ax2z.set_ylim(cy - rz2, cy + rz2)
    ax2.indicate_inset_zoom(ax2z, edgecolor=GREY)

    fig.tight_layout()
    save_fig(fig, "fig_rotation_ellipse")

    def ang(u: np.ndarray, v: np.ndarray) -> float:
        return float(
            np.degrees(
                np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
            )
        )

    g_before = np.linalg.norm(M_old - M_new)
    g_re = np.linalg.norm(M_re - M_new)
    print(
        f"theta={theta:.2f} deg  "
        f"Abar gap: before={g_before:.3f}, realigned={g_re:.3f}  "
        f"B angle to B_k: before={ang(B_old, B_new):.2f} deg, "
        f"realigned={ang(B_re, B_new):.2f} deg"
    )
    # self-checks: rotation preserves eig(A); realignment tracks the refit
    assert np.allclose(
        sorted(np.linalg.eigvals(A_re)), sorted(np.linalg.eigvals(A_old))
    ), "similarity transform must preserve the spectrum"
    assert g_re < 0.7 * g_before, "realigned Abar should track Abar_k"
    assert ang(B_re, B_new) < 0.5 * ang(B_old, B_new), (
        "Kpp B should realign toward B_k"
    )


if __name__ == "__main__":
    main()
