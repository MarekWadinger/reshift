# %% [markdown]
# # A nonlinear, time-varying benchmark: the two-tank system with a drifting orifice
#
# Example `00` (LTV control) drives **linear** dynamics whose coefficient varies
# linearly in time, $\omega(t) = 1 + \epsilon t$, and shows that online DMD
# *tracks* that drift. Example `12` (forced Van der Pol) is **nonlinear** with an
# *abrupt* parameter step. This example sits at the intersection — the
# **nonlinear** two-tank system,
#
# $$\dot h_1 = \frac{q}{F_1} - \frac{k_1(t)}{F_1}\sqrt{h_1}, \qquad
#   \dot h_2 = \frac{k_1(t)}{F_2}\sqrt{h_1} - \frac{k_2(t)}{F_2}\sqrt{h_2},$$
#
# with an **exogenous inflow** $q(t)=q_0+a\sin\Omega t$ (a genuine control,
# independent of the state) and an **LTV-style time-varying term** on the outflow
# coefficients,
#
# $$k_i(t) = k_0\,(1 + \epsilon t),$$
#
# the same linear-in-time drift as example `00` — physically, an orifice slowly
# *fouling* (scaling/clogging) as the plant runs. On its own the two-tank model
# is nonlinear but *time-invariant* (the RHS has no explicit $t$); the drift term
# is what makes it time-varying.
#
# **Why this example needs a control-aware score.** The default `SubID` score is
# an input-blind subspace-projection residual $\lVert x-\Phi\Phi^\top x\rVert$: it
# never uses the input, so a known sinusoidal forcing leaks straight into the
# statistic as a ripple at the forcing frequency (one false bump per inflow
# cycle). Here we use `SubIDChangeDetector(..., control_aware=True)`, which scores
# the **one-step prediction residual** $\lVert x_{k+1}-(A x_k + B u_k)\rVert$ with
# the control matrix $B$ that `OnlineDMDwC` already learns. The forced response is
# now *predicted* and cancels out, the baseline is flat, and — with the input
# accounted for — the raw two-state model needs no Hankel lift.
#
# We contrast the two ways such a plant degrades:
#
# 1. **Abrupt** — a gentle drift runs throughout (silently tracked out), then the
#    base coefficient *steps* at $t=T/2$. The held model mismatches the new
#    regime and the score **peaks** sharply against the flat baseline.
# 2. **Gradual** — the plant is time-invariant until $t=T/2$, then a slow linear
#    drift *switches on* (incipient fouling). The rolling model **tracks** it,
#    exactly as in example `00`, so the score **does not** peak.

# %%
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river.decomposition import OnlineDMDwC
from scipy.integrate import solve_ivp

sys.path.append("../")
from reshift.chdsubid import SubIDChangeDetector
from reshift.rolling import Rolling

if TYPE_CHECKING:
    from collections.abc import Callable

# %%
# Fixed configuration
T = 4000.0  # total simulated time (tanks are slow: timescale ~1/k)
DT = 1.0  # sampling step -> 4000 snapshots
F1 = 1.0  # tank-1 area
F2 = 1.0  # tank-2 area
K0 = 0.05  # base outflow coefficient
EPS = 1.5e-4  # LTV drift rate of k_i(t) = K0*(1 + EPS*t)  (cf. example 00)
STEP = 5.0  # abrupt multiplicative jump of K0 at the change point
EPS_FAST = 2e-4  # drift rate switched on at the change in the gradual case

U0 = 0.11  # mean exogenous inflow (steady level ~ (U0/K0)^2)
UA = 0.025  # inflow modulation amplitude (keeps levels off the empty boundary)
OMEGA_U = 0.10  # inflow modulation frequency (~5 cycles per detection window)
SIGMA = 0.01  # state measurement noise std
U_SIGMA = 0.005  # control-signal (inflow reading) noise std

LEARN_W = 600  # rolling OnlineDMDwC window
P = 2  # state DMD rank (= full raw state; control-aware needs no Hankel lift)
Q = 1  # input rank (scalar inflow)
DET_WIN = 300  # detection window (ref_size = test_size)

DETECT_RATIO = 2.0  # post/pre score ratio above which we call a detection
TRACK_RATIO = 1.5  # ratio below which the drift counts as tracked-out


def inflow(t: float | np.ndarray) -> np.ndarray:
    """Exogenous inflow q(t): a positive, state-independent control signal."""
    return U0 + UA * np.sin(OMEGA_U * np.asarray(t))


def twotank(
    k1: Callable[[float], float],
    k2: Callable[[float], float],
) -> Callable[[float, np.ndarray], list[float]]:
    """Two-tank RHS f(t, [h1, h2]) with time-varying outflow coeffs k1(t), k2(t)."""

    def f(t: float, s: np.ndarray) -> list[float]:
        h1 = np.clip(s[0], 0.0, 10.0)  # levels in the two tanks
        h2 = np.clip(s[1], 0.0, 10.0)
        q = inflow(t)
        dh1 = q / F1 - k1(t) / F1 * np.sqrt(h1)
        dh2 = k1(t) / F2 * np.sqrt(h1) - k2(t) / F2 * np.sqrt(h2)
        return [dh1, dh2]

    return f


def simulate(
    mode: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Integrate the time-varying two-tank with a change at t=T/2.

    mode="abrupt": gentle LTV drift throughout, base coeff steps at the change.
    mode="gradual": time-invariant before the change, drift switches on after.

    Returns (states, inputs, cp index).
    """
    t = np.arange(0.0, T, DT)
    cp = len(t) // 2
    tcp = t[cp]
    if mode == "abrupt":

        def k_pre(tt: float) -> float:
            return K0 * (1 + EPS * tt)

        def k_post(tt: float) -> float:
            return K0 * STEP * (1 + EPS * tt)
    elif mode == "gradual":

        def k_pre(_tt: float) -> float:  # time-invariant regime
            return K0

        def k_post(tt: float) -> float:
            return K0 * (1 + EPS_FAST * (tt - tcp))
    else:
        msg = f"unknown mode {mode!r}"
        raise ValueError(msg)

    kw = {"rtol": 1e-9, "atol": 1e-9}
    pre = solve_ivp(
        twotank(k_pre, k_pre),
        (t[0], t[cp]),
        [1.0, 1.0],
        t_eval=t[:cp],
        **kw,
    )
    post = solve_ivp(
        twotank(k_post, k_post),
        (t[cp], t[-1]),
        pre.y[:, -1],
        t_eval=t[cp:],
        **kw,
    )
    x = np.hstack([pre.y, post.y]).T
    rng = np.random.default_rng(seed)
    x += rng.normal(0, SIGMA, x.shape)
    states = pd.DataFrame(x, columns=pd.Index(["$h_1$", "$h_2$"]))
    # The plant is driven by the clean inflow (above); the model only sees a
    # noisy reading of it, like the state measurements.
    q = inflow(t) + rng.normal(0, U_SIGMA, len(t))
    inputs = pd.DataFrame({"$q$": q})  # exogenous control
    return states, inputs, cp


def score_over_time(states: pd.DataFrame, inputs: pd.DataFrame) -> np.ndarray:
    """Run control-aware ODMD-CPD over the stream; return the per-sample score trace.

    ``control_aware=True`` scores the one-step prediction residual
    ``||x_{k+1} - (A x_k + B u_k)||`` using the learned control matrix B, so the
    known sinusoidal forcing is regressed out of the score instead of leaking in
    as a per-cycle ripple (which the input-blind subspace-projection score does).
    With the input handled, the raw two-state model needs no Hankel lift.
    """
    odmd = Rolling(
        OnlineDMDwC(
            p=P,
            q=Q,
            initialize=LEARN_W - 1,
            w=1.0,
            exponential_weighting=False,
            eig_rtol=None,
        ),
        LEARN_W,
    )
    det = SubIDChangeDetector(
        odmd,
        ref_size=DET_WIN,
        test_size=DET_WIN,
        grace_period=LEARN_W + DET_WIN + 1,
        start_soon=True,
        control_aware=True,
    )
    s = np.zeros(len(states))
    rows = zip(
        states.to_dict(orient="records"),
        inputs.to_dict(orient="records"),
        strict=True,
    )
    for i, (xi, ui) in enumerate(rows):
        try:
            s[i] = det.score_one(xi)
        except ZeroDivisionError:  # degenerate reference (numerical fragility)
            s[i] = s[i - 1] if i else 0.0
        det.learn_one(xi, u=ui)
    return np.nan_to_num(s)


def plot_case(mode: str, title: str, *, detect: bool) -> None:
    """Simulate one degradation mode, score it, plot signal/input/score, self-check.

    detect=True  -> assert the score peaks after the change (clean detection).
    detect=False -> assert the score stays flat (the drift is tracked out).
    """
    states, inputs, cp = simulate(mode)
    scores = score_over_time(states, inputs)

    fig, (ax_sig, ax_u, ax_sc) = plt.subplots(
        3,
        1,
        figsize=(11, 6),
        sharex=True,
        height_ratios=[1, 0.6, 1.4],
    )
    ax_sig.plot(states["$h_1$"].to_numpy(), lw=0.8, color="0.3", label="$h_1$")
    ax_sig.plot(states["$h_2$"].to_numpy(), lw=0.8, color="C0", label="$h_2$")
    ax_sig.axvline(cp, color="C1", ls="--", label="true change")
    ax_sig.set(ylabel="level", title=title)
    ax_sig.legend(loc="upper left", ncol=3)

    ax_u.plot(inputs["$q$"].to_numpy(), lw=0.8, color="C0")
    ax_u.axvline(cp, color="C1", ls="--")
    ax_u.set(ylabel="$q$ (inflow)")

    ax_sc.plot(scores, lw=1.0, color="C2")
    ax_sc.axvline(cp, color="C1", ls="--")
    ax_sc.set(xlabel="sample", ylabel="ODMD-CPD score")
    fig.tight_layout()
    plt.show()

    base_max = scores[LEARN_W + DET_WIN + 1 : cp].max()
    post = scores[cp:]
    ratio = post.max() / base_max
    if detect:
        # The held pre-change model must clearly mismatch the new regime.
        assert ratio > DETECT_RATIO, (
            f"[{mode}] expected a peak, got {ratio:.1f}x"
        )
        verdict = f"detected — peak at sample {post.argmax() + cp}"
    else:
        # The rolling model should absorb the slow drift: no peak above baseline.
        assert ratio < TRACK_RATIO, (
            f"[{mode}] expected tracking, got {ratio:.1f}x"
        )
        verdict = "tracked out — no peak (as in example 00)"
    print(f"ok [{mode}]: {ratio:.1f}x baseline — {verdict}")


# %% [markdown]
# ## Abrupt change: orifice partially blocks at the midpoint
#
# A gentle LTV drift (the time-varying term) runs the whole time; the rolling
# model tracks it, and — because the score is control-aware — the sinusoidal
# forcing is regressed out, so the baseline is **flat and ripple-free**. At
# $t=T/2$ the base outflow coefficient steps up: the held model mismatches the
# new regime and the score peaks sharply once the post-change window fills. The
# peak lags the change by about one detection window — the expected latency.

# %%
plot_case(
    "abrupt",
    rf"Time-varying two-tank — abrupt $k_0$ step ($\times${STEP})",
    detect=True,
)

# %% [markdown]
# ## Gradual change: fouling drift switches on at the midpoint
#
# Before the change the plant is genuinely time-invariant. At $t=T/2$ a slow
# linear drift $k_i(t)=k_0(1+\epsilon t)$ switches on (incipient fouling). Because
# the rolling `OnlineDMDwC` re-fits every step, it simply *follows* the slow
# drift — exactly the tracking behaviour example `00` demonstrates for the LTV
# system — and the score never peaks. The limitation worth stating plainly: CPD
# flags abrupt regime breaks, but a sufficiently slow LTV drift is absorbed by
# the adaptive model rather than flagged.

# %%
plot_case(
    "gradual",
    "Time-varying two-tank — gradual fouling drift onset",
    detect=False,
)
