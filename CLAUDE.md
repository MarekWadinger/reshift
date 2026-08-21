# odmd-subid-cpd

## Plotting — use the existing toolkits, do not reinvent

- **Talk figures** (`publications/ECC26/talk/figs/`): shared helpers live in
  `figs/talkplot.py` — `panels`, `save_fig`, `talk_style`, `sliding_score`,
  `zoom_inset`, colors (`BLUE/RED/GREY/GREY_L/GREEN/AMBER`), `AX_LEFT/AX_RIGHT`.
  Import these; do NOT re-declare rcParams blocks, color constants, or a
  png+pdf save loop in a new `fig_*.py`. Two greys are intentional:
  `GREY = #6b7280` (dark, talkplot/rotation figs), `GREY_L = #9aa0a6` (light,
  make_talk_figs family) — never "unify" them, it shifts pixels.
- **Paper/thesis figures**: use the public API in `reshift/plot.py`
  (`set_size`, `plot_chd`) — a separate styling regime (usetex, small fonts).
  Never mix it with the talk style.
- Figure regeneration must be behavior-preserving: verify by comparing the
  regenerated PNGs (PDFs embed timestamps; diff the PNGs).
