# ECC26 talk — artifact index

Everything generated lives here. Nothing is "lost"; if it's not in this list it doesn't exist.

## Build

- `make_talk_figs.py` → all `fig_*.pdf`                  ·  `uv run python figs/make_talk_figs.py`
- `slides.tex`         → `slides.pdf`                    ·  `pdflatex slides.tex` (run from this dir)
- `talkplot.py`        → shared plotting toolkit (see Figure conventions)

## Figure conventions

`figs/talkplot.py` is the single source for shared plotting helpers: `panels`,
`save_fig`, `talk_style` (projector rcParams), `sliding_score`, `zoom_inset`,
colors, and `AX_LEFT/AX_RIGHT` (shared plot extent across slides). New
`fig_*.py` scripts import from it — no re-declared rcParams blocks, colors, or
save loops. Two greys exist on purpose: `GREY = #6b7280` (dark) and
`GREY_L = #9aa0a6` (light, the make_talk_figs family); do not unify them.

## Documents

| file | what |
|---|---|
| `outline.md` | speaker outline, slide-by-slide + timings + Q&A |
| `outline.orig.md` | original outline (backup, pre-rewrite) |
| `slides.tex` / `slides.pdf` | the deck (14 pages incl. backup) |

## Figures (figs/)

| pdf | slide | claim shown | honesty status |
|---|---|---|---|
| `fig_cold_open` | 1 | battery temp rising | real data |
| `fig_twotank_proof` | 9 | detects bias / gain-change / drift | real sim; NO "flat through control" claim |
| `fig_bess_hx10_crop` | 10 | fault detected ≈ c after onset | real data; hx10 (Q≈1.9) |
| `fig_bess_hx20_crop` | alt 10 | same, matches committed paper fig | real data; hx20 (Q≈0.9) |
| `fig_bess_hx10_full` | backup | full run, late peaks labelled uncertain | real data |
| `fig_stability_variance` | (unused) | variance vs SVD-ref | claim did NOT reproduce — do not use |

## Open decisions (unresolved)

1. **Spine**: "control → no fire" is NOT supported by the detector (change score is ~control-blind; verified 3 ways). Reframe pending.
2. **BESS run**: paper commits hx20 (Q=0.88) but text says "Q≈2" (=hx10). Pick one; fix paper.
3. **SOTA numbers table**: not generated. Machinery = `examples/cpdmd_compare.py:metrics_table()`.

## Tables in the paper

- capability matrix (✓/✗): slides only (slide 5/11); paper has it as prose (`root.tex` Discussion).
- experiment summary: `root.tex` Table `tab:experiments`.
