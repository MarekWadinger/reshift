# ECC26 talk — artifact index

Everything generated lives here. Nothing is "lost"; if it's not in this list it doesn't exist.

## Build
- `make_talk_figs.py` → all `fig_*.pdf` except claims  ·  `uv run python figs/make_talk_figs.py`
- `fig_claims.py`      → `fig_claims_compare.pdf`        ·  `uv run python figs/fig_claims.py`
- `slides.tex`         → `slides.pdf`                    ·  `pdflatex slides.tex` (run from this dir)
- `talkplot.py`        → reusable `panels()` plotter (every new figure goes through it)
- `_claims_probe.py`   → two-tank sim + `run_all()` (all methods); imported by `fig_claims.py`

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
| `fig_trap` | 4 | identical input → different output | synthetic, TRUE by construction |
| `fig_twotank_proof` | 9 | detects bias / gain-change / drift | real sim; NO "flat through control" claim |
| `fig_bess_hx10_crop` | 10 | fault detected ≈ c after onset | real data; hx10 (Q≈1.9) |
| `fig_bess_hx20_crop` | alt 10 | same, matches committed paper fig | real data; hx20 (Q≈0.9) |
| `fig_bess_hx10_full` | backup | full run, late peaks labelled uncertain | real data |
| `fig_claims_compare` | (analysis) | all methods, 4 phases | shows toDMDc≈blind, fires on control |
| `fig_stability_variance` | (unused) | variance vs SVD-ref | claim did NOT reproduce — do not use |

## Open decisions (unresolved)
1. **Spine**: "control → no fire" is NOT supported by the detector (change score is ~control-blind; verified 3 ways). Reframe pending.
2. **BESS run**: paper commits hx20 (Q=0.88) but text says "Q≈2" (=hx10). Pick one; fix paper.
3. **SOTA numbers table**: not generated. Machinery = `examples/cpdmd_compare.py:metrics_table()`.

## Tables in the paper
- capability matrix (✓/✗): slides only (slide 5/11); paper has it as prose (`root.tex` Discussion).
- experiment summary: `root.tex` Table `tab:experiments`.
