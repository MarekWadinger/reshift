"""Claims figure: all methods on the two-tank scenario, via talkplot.panels."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

from _claims_probe import run_all, simulate
from talkplot import AMBER, GREEN, GREY, RED, panels

x, u, ev = simulate()
res = run_all(x, u)
c1, (d0, d1), c2 = ev["change1"], ev["drift"], ev["change2"]
n = len(x)
phases = [
    (0, c1, GREEN, "A control"),
    (c1, d0, RED, "B change"),
    (d0, d1, AMBER, "C drift"),
    (d1, n, RED, "D change"),
]

panel_list = [
    {"label": "tank levels", "y": x, "lw": 0.6},
    {"label": "control $u$", "y": u, "lw": 0.7, "color": GREY},
]
panel_list += [
    {"label": name.split("(")[0].strip(), "y": s} for name, s in res.items()
]

panels(
    panel_list,
    phases=phases,
    title="Two-tank: control throughout; faults in the MAP (valve constants)",
    save=str(HERE / "fig_claims_compare"),
)
