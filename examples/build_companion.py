"""Build a self-contained static companion for GitHub Pages.

Precomputes the plant/DMD residuals over a grid of (method x change x width x
noise), writes one JSON per combo plus a manifest, and copies the explorer page
and vendored Plotly next to them. The result is a folder that runs fully
client-side (no backend) -- host it on GitHub Pages and link a QR to it.

Run::  python examples/build_companion.py [out_dir]   # default: examples/companion
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import plant_residual_server as srv

# Grid the static page snaps the width/noise sliders to. Keep it small -- every
# combo is one JSON file (~45 kB). methods x changes x widths x noises files.
WIDTHS = [6, 30, 90]  # transition width: sharp / medium / slow
NOISES = [0.0, 0.08, 0.2]  # measurement noise
CHANGES = ["permanent", "transient"]

HERE = Path(__file__).resolve().parent


def key(method: str, change: str, width: int, noise: float) -> str:
    """Stable filename stem shared with the page's fetch path."""
    return f"{method}_{change}_w{width}_n{noise:.2f}"


def build(out: Path) -> None:
    """Generate <out>/data/*.json, manifest.json, index.html, plotly.min.js."""
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    count = 0
    for method in srv.METHODS:
        for change in CHANGES:
            for w in WIDTHS:
                for n in NOISES:
                    d = srv.compute_residual(
                        method,
                        change,
                        float(w),
                        float(n),
                    )
                    (data / f"{key(method, change, w, n)}.json").write_text(
                        json.dumps(d),
                    )
                    count += 1
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "methods": list(srv.METHODS),
                "changes": CHANGES,
                "widths": WIDTHS,
                "noises": NOISES,
                "warmup": srv.LEARN_W,
                "onset": srv.CP,
            },
        ),
    )
    shutil.copy(HERE / "plotly.min.js", out / "plotly.min.js")
    shutil.copy(HERE / "window_explorer.html", out / "index.html")
    # Include the paper if it's already built locally; in CI it's compiled from
    # LaTeX and staged separately (see .github/workflows/pages.yml).
    paper = HERE.parent / "publications" / "ECC26" / "root.pdf"
    if paper.exists():
        shutil.copy(paper, out / "paper.pdf")
    total = sum(f.stat().st_size for f in out.rglob("*"))
    print(f"built {out}: {count} residual files, {total / 1e6:.1f} MB total")  # noqa: T201


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "companion"
    build(target)
