"""Probe: controlled LTV system, run ALL methods, report per-phase so we can
pick the sharpest counter-example (a control-blind baseline that fires on the
control steps while toDMDc stays quiet).

  Phase A  control steps, map fixed   -> ours LOW (no fire); blind baselines may FIRE
  Phase B  abrupt map change          -> all should SPIKE (detect)
  Phase C  slow drift of the map       -> ours LOW (adapt)
  Phase D  abrupt change during drift  -> ours SPIKE (still detect)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))


def simulate(seed: int = 0, noise: float = 0.05):
    r"""Nonlinear two-tank with input delay (paper model):

        dh1/dt = q(t-tau) - (k1/F1) sqrt(h1)
        dh2/dt = (k1/F2) sqrt(h1) - (k2/F2) sqrt(h2)

    Control = inflow q (stepped). Fault = valve-constant change (the *map*).
      A  control steps, valves fixed         -> no fire
      B  abrupt k2 jump (valve fault)         -> detect
      C  slow k2 drift (gradual fouling)      -> adapt
      D  abrupt k1 jump during drift          -> still detect
    """
    rng = np.random.default_rng(seed)
    n, dt, tau = 11000, 1.0, 25  # delay = 25 samples
    F1 = F2 = 1.0
    k1 = k2 = 0.20  # nominal valve constants (fast response)
    pb, drift0, drift1 = 6000, 6500, 9000
    h = np.zeros((n, 2))
    q = np.zeros(n)
    h[0] = [4.0, 4.0]
    qk = 0.4
    for k in range(n - 1):
        if k % 120 == 0:  # LARGE control steps throughout
            qk = float(rng.uniform(0.15, 0.75))
        q[k] = qk
        # --- inject faults into the MAP (valve constants), not the signal ---
        k2k = k2
        k1k = k1
        if pb <= k < drift0:
            k2k = k2 * 1.7  # B: abrupt valve fault
        elif drift0 <= k < drift1:
            a = (k - drift0) / (drift1 - drift0)
            k2k = k2 * (1.7 - 0.7 * a)  # C: drift back toward nominal
        if k >= drift1:
            k1k = k1 * 1.8  # D: abrupt second fault
        qd = q[k - tau] if k >= tau else 0.4
        h1, h2 = max(h[k, 0], 0.0), max(h[k, 1], 0.0)
        dh1 = qd - (k1k / F1) * np.sqrt(h1)
        dh2 = (k1k / F2) * np.sqrt(h1) - (k2k / F2) * np.sqrt(h2)
        w = rng.normal(0, 0.01, 2)  # process noise: never freezes
        h[k + 1, 0] = max(h[k, 0] + dt * dh1 + w[0], 0.0)
        h[k + 1, 1] = max(h[k, 1] + dt * dh2 + w[1], 0.0)
    x = h + rng.normal(0, noise, h.shape)  # measurement noise
    u = q.reshape(-1, 1)
    return x, u, {"change1": pb, "drift": (drift0, drift1), "change2": drift1}


W = 1500  # learning / rolling window
REF = TEST = 200
HN = 8  # time-delay order (poly+hankel give residual -> no /0)


def _subid(core, x, u=None, *, poly=True):
    from river.feature_extraction import PolynomialExtender
    from river.preprocessing import Hankelizer

    from reshift.chdsubid import SubIDChangeDetector

    det = SubIDChangeDetector(
        core,
        ref_size=REF,
        test_size=TEST,
        grace_period=W,
        start_soon=True,
    )
    pipe = Hankelizer(HN) | det
    if poly:
        pipe = PolynomialExtender(2) | pipe
    xs = pd.DataFrame(x).to_dict(orient="records")
    us = pd.DataFrame(u).to_dict(orient="records") if u is not None else None
    s = np.zeros(len(x))
    fails = 0
    for i in range(len(x)):
        try:
            s[i] = pipe.score_one(xs[i])
        except ZeroDivisionError:  # degenerate reference (numerical fragility)
            s[i] = s[i - 1] if i else 0.0
            fails += 1
        if us is not None:
            pipe.learn_one(xs[i], u=us[i])
        else:
            pipe.learn_one(xs[i])
    if fails:
        print(f"    [{fails} numerical failures carried forward]")
    return np.nan_to_num(s)


def run_all(x, u):
    from river.decomposition import OnlineDMD, OnlineDMDwC, OnlineSVD

    from reshift.rolling import Rolling

    out = {}
    # 1) ours: control-aware DMDc
    out["toDMDc (ours, control-aware)"] = _subid(
        Rolling(
            OnlineDMDwC(
                p=4,
                q=1,
                initialize=W - 1,
                w=1.0,
                exponential_weighting=False,
                eig_rtol=None,
            ),
            W,
        ),
        x,
        u,
    )
    # 2) control-blind online DMD (no u)
    out["online DMD (control-blind)"] = _subid(
        Rolling(
            OnlineDMD(
                r=4,
                initialize=W,
                w=1.0,
                exponential_weighting=False,
                seed=42,
            ),
            W + 1,
        ),
        x,
    )
    # 3) control-blind SVD reference
    out["SVD reference (control-blind)"] = _subid(
        Rolling(OnlineSVD(n_components=4, initialize=W, seed=42), W + 1),
        x,
    )
    # 4) CUSUM on the signal energy (classic, control-blind)
    e = np.linalg.norm(x, axis=1)
    mu, sd = e[:W].mean(), e[:W].std()
    z = (e - mu) / (sd + 1e-9)
    cs = np.zeros_like(z)
    for k in range(1, len(z)):
        cs[k] = max(0.0, cs[k - 1] + abs(z[k]) - 0.5)
    out["CUSUM (signal-based)"] = cs
    return out


if __name__ == "__main__":
    x, u, ev = simulate()
    res = run_all(x, u)
    c1 = ev["change1"]
    d0, d1 = ev["drift"]
    c2 = ev["change2"]
    phases = {
        "A control(1k:4k)": (1000, 4000),
        "B change1": (c1, c1 + 400),
        "C drift": (d0 + 300, d1 - 200),
        "D change2": (c2, c2 + 400),
    }
    print(f"{'method':32s} " + " ".join(f"{p:>16s}" for p in phases))
    for name, s in res.items():
        # normalise each method to its own control-phase p95 so we compare SNR
        floor = np.percentile(s[1000:4000], 95) + 1e-9
        cells = []
        for a, b in phases.values():
            cells.append(f"{s[a:b].max() / floor:6.1f}x")
        print(f"{name:32s} " + " ".join(f"{c:>16s}" for c in cells))
    print(
        "\n(values = peak/own-control-floor; ~1x in phase A = no fire on control)",
    )
    np.savez(
        Path(__file__).parent / "_claims_data.npz",
        x=x,
        u=u,
        c1=c1,
        d0=d0,
        d1=d1,
        c2=c2,
        **{f"s{i}": s for i, s in enumerate(res.values())},
        names=np.array(list(res.keys())),
    )
