## v5.1.0 (2026-06-22)

### Features

- **examples**: compare ODMD-CPD against CPDMD baseline (#28)

### Docs

- drop stale codecov badge token after repo rename (#24)

## v5.0.0 (2026-06-19)

### Features

- rename package to reshift (#23)

### Docs

- modernize README + package metadata (logo, pain-first header, keywords) (#22)

## v4.0.1 (2026-06-19)

### Fixes

- **ci**: bump/release with built-in GITHUB_TOKEN, not stale PAT (#21)
- **ci**: make version bump actually run; robust docs trigger (#20)
- **docs**: deploy gh-pages with built-in GITHUB_TOKEN; add workflow_dispatch (#19)
- **ci**: don't fail CI on codecov upload for Dependabot/fork PRs (#18)
- **ci**: correct workflow_run names + harden the workflow bundle (#16)
- **lint**: resolve real violations; drop from ruff ignore
- **examples**: annotate trusted pickle loads with noqa S301
- **datasets**: add timeout to requests.get calls (S113)

### Refactor

- **metrics**: extract distance variants; clear commented code (ERA001); per-file-ignore notebooks
- **lint**: extract magic-value constants and add package inits (PLR2004/INP001); drop from ignore
- **api**: make boolean params keyword-only (FBT001/2/3); drop from ignore
- **types**: scope SLF001 to research notebooks (SLF001)
- **types**: replace blanket type-ignores with fixes/specific codes (PGH003)
- **pandas**: use .to_numpy() over .values (PD011)
- **sim**: use context managers and contextlib.suppress (SIM105/SIM115)
- **perf**: replace append-loops with comprehensions (PERF401)
- **imports**: move type-only imports into TYPE_CHECKING (TC002/TC003)
- **pathlib**: convert os.path to pathlib (PTH)
- **logging**: replace prints with logging in functions/; enable T201
- remove unused bayes_opt_parallel.py

### Docs

- allow the deploy job to run on workflow_dispatch too (not only
workflow_run), so manual deploys work and the trigger is explicit.
- **lint**: fix N817/D105/D107/D401/D404/D417/W505; drop from ignore
- **docstrings**: add Google-style docstrings to public API (D100-D104)
- **todo**: add author and linked issues to TODOs (TD002/TD003)
- **ruff**: document NPY002 as deliberately preserved

## v4.0.0 (2026-04-15)

### Fixes

- enhance LaTeX availability check in plot.py

### Docs

- **ECC26**: update author acknowledgment in LaTeX document
- **examples**: refresh notebooks with updated results and figures
- **ECC26**: document figure convention and tighten discussion
- add ROADMAP.md with cross-referenced planned work
- **review**: catalog logical issues found in decomposition module

## v3.0.0 (2026-04-10)

### Fixes

- suppress type error for Timedelta division in metrics
- suppress method override type check error in bayes_opt_parallel
- wrap DataFrame columns and index with pd.Index() in results dataframe
- improve type hints for River Rolling support and DataFrame pandas typing
- add DataFrame support to plot_chd function and fix notebook column naming
- remove Python object tag from mkdocs.yml YAML
- suppress type checking for dynamic module import in run-benchmarks.py
- remove duplicate column renaming code from 08_nonlinear_delay_control.ipynb
- remove Python object tag from mkdocs.yml YAML

### Docs

- add cover letter and ORCID information for manuscript submission
- improve docstrings and refactor scoring logic in change detection

## v2.0.0 (2025-11-14)

### Fixes

- **functions/**: standardize docstring formatting in metrics and rolling modules

### Docs

- revise paper materials for submission
- enhance the theoretical foundation and practical application of ODMD-CPD
- update author information
- fix cite command usage and move section headers to section files
- update document structure for IEEE journal submission
- **README**: Revise installation instructions in README to clarify local setup options with Rust, VSCode devcontainer, and Docker.

## v1.0.0 (2024-11-27)

Initial development of the library and experiments (pre-conventional-commit
history, curated).

### Features

- **OnlineDMD**: streaming DMD with control, exponential weighting, and windowing
- **SubIDDriftDetector**: subspace-identification change/anomaly detector with configurable detection delay
- **OnlineSVD** / **OnlinePCA**: incremental SVD and online PCA (Eftekhari et al., 2019)
- **Hankelizer**: time-delay embedding as a `river.Transformer`, with mini-batch and rolling variants
- multivariate input support and `transform_one`/`transform_many` interfaces aligned with the `river` API
- parallelized Bayesian hyperparameter optimization
- worked examples and datasets: CATS, SKAB, NPRS, BESS, USP, and nonlinear delayed-control systems
- Streamlit web application for interactive exploration

### Fixes

- known-`B` control handling, numerical precision, and subspace-distance computation
- `transform_many` with rolling windows and `Rolling` alignment with its `river` parent
- Python 3.9 compatibility

### Performance

- faster eigendecomposition and optimized `SubIDDriftDetector` internals

### Refactor

- dropped the `cvxpy` dependency; renamed `DMDC`→`DMDwC`, `train_size`→`ref_size`, and standardized `(n, m)` input shapes
- migrated decomposition methods onto the `river` API

### Docs

- project README with benchmark results and a mkdocs/readthedocs documentation site
- manuscript foundations: method, notation, results, introduction, conclusion, and bibliography
