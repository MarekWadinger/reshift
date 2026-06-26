"""Backend for the window-explorer upper screen.

Turn a real two-tank plant change into a model-mismatch residual, using a chosen
DMD-family identifier, and serve it to the sibling ``window_explorer.html``.

One JSON endpoint::

    GET /residual?method=onlineDMDc&change=permanent&width=6&noise=0.08

returns ``{"r", "clean", "onset", "states", "pred", "inputs"}`` where ``r`` is the
per-sample one-step prediction error ``||x_k - (A x_{k-1} + B u_{k-1})||`` and
``clean`` is the outflow-coefficient change profile (ground truth for the lower
screen's FAR/MAR). The plant is the time-varying two-tank of example 13; the
change is applied to the outflow coefficient as a permanent smoothstep or a
transient Gaussian bump of tunable width.

Run::  python examples/plant_residual_server.py        # then open http://localhost:8000
       python examples/plant_residual_server.py --selftest

Models are preconfigured (fixed rank/window) per method -- no per-method knobs.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import numpy as np
from river.decomposition import OnlineDMD, OnlineDMDwC
from scipy.integrate import solve_ivp

sys.path.append(str(Path(__file__).resolve().parent.parent))
from reshift.dmd import DMD, DMDwC

if TYPE_CHECKING:
    from collections.abc import Hashable

# --- plant (two-tank, cf. example 13), shrunk for interactive response --------
T, DT = 900.0, 1.0  # 900 snapshots
CP = 450  # change point (sample index)
F1 = F2 = 1.0
K0, STEP = 0.05, 5.0  # base outflow coeff; multiplicative jump at change
U0, UA, OMEGA_U = 0.11, 0.025, 0.10
LEARN_W = 180  # identification / warmup window


def _kmult(t: np.ndarray, change: str, width: float) -> np.ndarray:
    """Outflow-coefficient multiplier over time: 1 -> STEP (permanent) or a bump."""
    tcp = CP * DT
    # both changes START at tcp and last exactly `width`
    u = np.clip((t - tcp) / max(width, 1e-9), 0.0, 1.0)
    if change == "permanent":
        s = u * u * (3 - 2 * u)  # smoothstep up to the new level
    else:  # transient: raised-cosine pulse back to baseline over [tcp, tcp+width]
        s = np.sin(np.pi * u) ** 2
    return 1.0 + (STEP - 1.0) * s


def _simulate(
    change: str,
    width: float,
    noise: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the two-tank with a time-varying outflow coeff; return X, U, clean."""
    t = np.arange(0.0, T, DT)
    kprofile = _kmult(t, change, width)
    k_at = lambda tt: K0 * np.interp(tt, t, kprofile)  # noqa: E731

    def f(tt: float, s: np.ndarray) -> list[float]:
        h1, h2 = np.clip(s[0], 0.0, 10.0), np.clip(s[1], 0.0, 10.0)
        q = U0 + UA * np.sin(OMEGA_U * tt)
        k = k_at(tt)
        return [
            q / F1 - k / F1 * np.sqrt(h1),
            k / F2 * np.sqrt(h1) - k / F2 * np.sqrt(h2),
        ]

    sol = solve_ivp(
        f,
        (t[0], t[-1]),
        [4.0, 4.0],
        t_eval=t,
        rtol=1e-8,
        atol=1e-8,
    )
    rng = np.random.default_rng(seed)
    X = sol.y.T + rng.normal(0, noise, sol.y.T.shape)
    U = (U0 + UA * np.sin(OMEGA_U * t) + rng.normal(0, noise / 2, len(t)))[
        :,
        None,
    ]
    return X, U, kprofile


def _residual_batch(
    X: np.ndarray,
    U: np.ndarray,
    *,
    control: bool,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen identification on an early nominal window, one-step residual over all t.

    The fit window is the first ``LEARN_W`` samples -- well before the change and
    before any ref/test detection window -- so the residual over the whole record
    (and in particular both detection windows) is out of sample. Fitting on the
    pre-change window adjacent to the change would put the ref window in sample and
    optimistically deflate D_train.
    """
    win = slice(0, LEARN_W)
    if control:
        mdl = DMDwC(r=rank)
        mdl.fit(X[win], U=U[win])
        assert mdl.A is not None  # set by fit
        assert mdl.B is not None
        A, B = np.real(mdl.A), np.real(mdl.B)
        pred = X[:-1] @ A.T + U[:-1] @ B.T
    else:
        mdl = DMD(r=rank)
        mdl.fit(X[win])
        assert mdl.A is not None  # set by fit
        pred = X[:-1] @ np.real(mdl.A).T
    xhat = X.copy()
    xhat[1:] = pred
    r = np.zeros(len(X))
    r[1:] = np.linalg.norm(X[1:] - pred, axis=1)
    r[0] = r[1]
    return r, xhat


def _residual_online(
    X: np.ndarray,
    U: np.ndarray,
    *,
    control: bool,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling identification; causal one-step residual (predict, then learn)."""
    cols = [f"x{i}" for i in range(X.shape[1])]
    # Separate typed handles (not a union) so the control/no-control paths
    # type-check against their own learn_one / operator signatures.
    mdlc = (
        OnlineDMDwC(p=rank, q=1, initialize=LEARN_W - 1, w=1.0, eig_rtol=None)
        if control
        else None
    )
    mdln = (
        None
        if control
        else OnlineDMD(r=rank, initialize=LEARN_W - 1, w=1.0, eig_rtol=None)
    )
    r = np.full(len(X), np.nan)
    xhat = X.copy().astype(float)
    for t in range(1, len(X)):
        xk: dict[Hashable, float] = {
            c: float(v) for c, v in zip(cols, X[t - 1], strict=True)
        }
        yk: dict[Hashable, float] = {
            c: float(v) for c, v in zip(cols, X[t], strict=True)
        }
        try:
            if mdlc is not None:
                A, B = mdlc._reconstruct_AB()  # noqa: SLF001
                p = np.real(A) @ X[t - 1] + np.real(B) @ U[t - 1]
            else:
                assert mdln is not None
                p = np.real(mdln.A) @ X[t - 1]
            xhat[t] = p
            r[t] = np.linalg.norm(X[t] - p)
        except AttributeError, ValueError, TypeError:
            pass  # operator not available yet during warmup
        if mdlc is not None:
            mdlc.learn_one(xk, yk, {"u": float(U[t - 1, 0])})
        else:
            assert mdln is not None
            mdln.learn_one(xk, yk)
    # forward/backward fill warmup NaNs with the first finite residual
    finite = np.isfinite(r)
    if finite.any():
        r[~finite] = r[finite][0]
    else:
        r[:] = 0.0
    return r, xhat


# method key -> (online?, control?, rank)
METHODS = {
    "DMD": (False, False, 0),  # batch, full rank
    "tDMD": (False, False, 1),  # batch, truncated
    "DMDc": (False, True, 0),  # batch, control, full
    "onlineDMD": (True, False, 2),  # rolling, full
    "onlineDMDc": (True, True, 2),  # rolling, control, full
    "toDMDc": (True, True, 1),  # rolling, control, truncated
}


@lru_cache(maxsize=128)
def compute_residual(
    method: str,
    change: str,
    width: float,
    noise: float,
) -> dict:
    """Run plant + chosen identifier; return residual, change profile, onset."""
    online, control, rank = METHODS[method]
    X, U, clean = _simulate(change, width, noise)
    fn = _residual_online if online else _residual_batch
    r, xhat = fn(X, U, control=control, rank=rank)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    xhat = np.nan_to_num(xhat, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "r": r.round(5).tolist(),
        "clean": clean.round(5).tolist(),
        "onset": CP,
        "warmup": LEARN_W,  # samples before the model is identified; don't score these
        "states": X.round(5).T.tolist(),  # [h1, h2] plant (measured)
        "pred": xhat.round(5).T.tolist(),  # [h1, h2] model one-step prediction
        "inputs": U[:, 0].round(5).tolist(),  # exogenous inflow q
    }


HTML = Path(__file__).with_name("window_explorer.html")
PLOTLY = Path(__file__).with_name("plotly.min.js")
# Served live for the "Read the paper" link during local preview. On GitHub Pages
# the CI compiles this LaTeX source into paper.pdf instead (see pages.yml).
PAPER = (
    Path(__file__).resolve().parents[1] / "publications" / "ECC26" / "root.pdf"
)


class Handler(BaseHTTPRequestHandler):
    """Serve the explorer page and the /residual JSON endpoint."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default request logging."""

    def do_GET(self) -> None:
        """Route GET requests to the page or the residual endpoint."""
        u = urlparse(self.path)
        if u.path in ("/", "/window_explorer.html"):
            self._send(200, "text/html", HTML.read_bytes())
        elif u.path == "/plotly.min.js" and PLOTLY.exists():
            self._send(200, "application/javascript", PLOTLY.read_bytes())
        elif u.path == "/paper.pdf" and PAPER.exists():
            self._send(200, "application/pdf", PAPER.read_bytes())
        elif u.path == "/residual":
            q = parse_qs(u.query)
            try:
                out = compute_residual(
                    q.get("method", ["toDMDc"])[0],
                    q.get("change", ["permanent"])[0],
                    float(q.get("width", ["6"])[0]),
                    float(q.get("noise", ["0.08"])[0]),
                )
                self._send(200, "application/json", json.dumps(out).encode())
            except (KeyError, ValueError) as e:
                self._send(400, "text/plain", str(e).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _selftest() -> None:
    for m in METHODS:
        for change in ("permanent", "transient"):
            out = compute_residual(m, change, 6.0, 0.08)
            r = np.array(out["r"])
            base = r[LEARN_W:CP].mean()
            post = r[CP : CP + 200].max()
            assert np.isfinite(r).all()
            assert (r >= 0).all()
            ratio = post / base
            msg = f"{m:11s} {change:9s} baseline={base:7.3f}  peak/base={ratio:5.1f}x"
            print(msg)  # noqa: T201
    print("ok: all methods produce finite, non-negative residuals")  # noqa: T201


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        # threaded so the browser's concurrent per-model requests run in parallel
        ThreadingHTTPServer(("localhost", 8000), Handler).serve_forever()
