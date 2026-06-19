# Roadmap

Planned work for ODMD-SubID-CPD. Items are grouped by type, inspired by
[Conventional Commits](https://www.conventionalcommits.org/).

> **Legend** &mdash; items are roughly ordered by priority within each section.
> `[branch]` = work-in-progress on a named branch.

---

## feat: New Features & Methods

- [ ] Parallel hyperparameter tuning & evaluation scripts
      `[origin/parallel-tune-eval]`
- [ ] Comparison with [TIRE](https://github.com/deryckt/TIRE/tree/master)
      (Time-Invariant Representation)
      ([De Ryck et al., 2021](https://doi.org/10.1109/TSP.2021.3087031))
- [ ] Compare with additional CPD methods:
  - [ ] Pruned Exact Linear Time
        ([PELT; Killick et al., 2012](https://doi.org/10.1080/01621459.2012.737745))
  - [ ] Bayesian Estimator of Abrupt change, Seasonality, and Trend
        ([BEAST; Zhao et al., 2019](https://doi.org/10.1016/j.rse.2019.04.034))
        &mdash; [Rbeast](https://github.com/zhaokg/Rbeast)
  - [ ] Relative unconstrained Least-Squares Importance Fitting
        ([RuLSIF; Liu et al., 2013](https://doi.org/10.1016/j.neunet.2013.01.012))
  - [ ] GBDT-RuLSIF
        ([Hushchyn & Ustyuzhanin, 2021](https://doi.org/10.1016/j.jocs.2021.101385))
  - [ ] QDA-RuLSIF
        ([Hushchyn & Ustyuzhanin, 2021](https://doi.org/10.1016/j.jocs.2021.101385))
  - [ ] Kernel Learning CPD
        ([KL-CPD; Chang et al., 2019](https://arxiv.org/abs/1901.06077))
  - [ ] ChangeFinder
        ([Takeuchi & Yamanishi, 2006](https://doi.org/10.1109/TKDE.2006.1599387))
  - [ ] Online Neural Network
        ([OnlineNN; Hushchyn et al., 2020](https://arxiv.org/abs/2010.01388))
  - [ ] NEWMA
        ([Keriven et al., 2020](https://doi.org/10.1109/TSP.2020.2990597))
  - [ ] TS-CP2
        ([Deldari et al., 2021](https://doi.org/10.1145/3442381.3449903))
- [ ] Explore scoring options and weighting of individual terms
      <!-- functions/chdsubid.py:233-234 -->
      ([`functions/chdsubid.py:233-234`](functions/chdsubid.py#L233-L234))
- [ ] Explore strategy of updating OnlineDMD only when change is anticipated
      (anti-windup / switching behaviour)
- [ ] Implement OnlineDMD based on
      [Nedzhibov, 2023](https://doi.org/10.56082/annalsarscimath.2023.1-2.229)
- [ ] Enable on-the-fly Hankelization of control inputs
      ([`app/app.py:366`](app/app.py#L366))
- [ ] Improve grace period handling &mdash; detection should start once
      Transformer is fitted, not after a fixed grace period
      ([`functions/chdsubid.py:165`](functions/chdsubid.py#L165))
- [ ] Explore proper utilisation of imaginary part of score
      ([`functions/chdsubid.py:236`](functions/chdsubid.py#L236))
- [ ] Add support for 2D arrays in `hankel()`
      ([`functions/preprocessing.py:94`](functions/preprocessing.py#L94))
- [ ] Investigate weather-sensitive short-term load forecasting using DMDwC
      ([Mansouri et al., 2023](https://doi.org/10.1016/j.epsr.2023.109387))
- [ ] Find out why DMDwC needs SVD for the output space and validate
      whether omitting it is justified
      ([`functions/dmd.py:80`](functions/dmd.py#L80))

## fix: Bug Fixes & Correctness

- [ ] Fix values in matrix A changing sign
      (replacing `scipy.sparse.linalg.svds` with `numpy.linalg.svd` helped;
      needs a principled solution)
- [ ] Sharp peak when initializing change point detection &mdash; spike
      connected to increasing error between base and test statistics at
      early time steps
- [ ] Fix eco-pack example not converging &mdash; caused by long periods of
      constant values
      ([`examples/07_eco_pack.ipynb`](examples/07_eco_pack.ipynb))
- [ ] Address correlated / binary data causing SVD convergence issues
- [ ] Fix Zhang SVD batch update for truncated DMD with multiple rows
      (`np.block` dimension mismatch)
      ([L3](publications/IEEECSS/reviews/logical_issues_odmd.md#L3)
      &mdash; [`vendor/river/river/decomposition/osvd.py:919`](vendor/river/river/decomposition/osvd.py#L919),
      triggered via [`odmd.py:430`](vendor/river/river/decomposition/odmd.py#L430))
- [ ] Reconcile code norm (column-wise L1) with paper theory
      (Frobenius norm squared, Eq. 1)
      ([L7](publications/IEEECSS/reviews/logical_issues_odmd.md#L7)
      &mdash; [`functions/chdsubid.py:300`](functions/chdsubid.py#L300))
- [ ] DMD modes not orthonormal &mdash; reconstruction via
      Phi Phi^T is not a proper projector for non-normal
      systems; decide on approach (projected modes / QR / pseudoinverse /
      SVD-based projection)
      ([L1](publications/IEEECSS/reviews/logical_issues_odmd.md#L1)
      &mdash; [`vendor/river/river/decomposition/odmd.py:344-368`](vendor/river/river/decomposition/odmd.py#L344-L368))
- [ ] `xi` property has unbounded memory &mdash; Vandermonde matrix grows
      with `n_seen`; cap window or use closed-form formula
      ([L2](publications/IEEECSS/reviews/logical_issues_odmd.md#L2)
      &mdash; [`vendor/river/river/decomposition/odmd.py:370-392`](vendor/river/river/decomposition/odmd.py#L370-L392))
- [ ] Rolling DMD off-by-one in unsupervised window size
      (`_x_first` vs `_x_last`)
      ([L5](publications/IEEECSS/reviews/logical_issues_odmd.md#L5)
      &mdash; [`vendor/river/river/decomposition/odmd.py:471-510`](vendor/river/river/decomposition/odmd.py#L471-L510))
- [ ] Window width not aligned correctly with index in Streamlit app
      ([`app/app.py:136`](app/app.py#L136))

## benchmark: Evaluation & Datasets

- [ ] Evaluate on [TCPD](https://github.com/alan-turing-institute/TCPD)
      multivariate datasets
      ([Van den Burg & Williams, 2020](https://arxiv.org/abs/2003.06222)):
  - [ ] bee_waggle_6
  - [ ] occupancy
  - [ ] quality control (univariate)
  - [ ] scan line (univariate)
  - [ ] well log (univariate)
- [ ] Run against
      [TCPDBench](https://github.com/alan-turing-institute/TCPDBench)
      benchmark suite
- [ ] Integrate / compare with
      [ruptures](https://github.com/deepcharles/ruptures)
- [ ] Integrate / compare with
      [Rbeast](https://github.com/zhaokg/Rbeast)
      ([Zhao et al., 2019](https://doi.org/10.1016/j.rse.2019.04.034))
- [ ] Investigate [TimeEval](https://github.com/TimeEval/TimeEval)
      benchmark framework
- [ ] Evaluate on battery cycle-life dataset
      ([Severson et al., 2019](https://doi.org/10.1038/s41560-019-0356-8)
      &mdash; [data](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation))

## perf: Performance & Numerical

- [ ] Find fastest eigenvector computation path
      (`eig`, `eigh`, `schur`, banded, tridiagonal)
- [ ] Use sparse SVD only when the number of snapshots to update is large
      (switch heuristic)

## docs: Documentation & Paper

- [ ] Write about evaluation criteria &mdash; define TP (detected CP within
      margin of ground-truth), FN (outside margin or missed). Justify
      Precision & Recall over Accuracy for CPD (class imbalance). Add MAE
      as supplementary metric measuring closeness of detected CPs to true
      CPs. Specify margin = 10 used in experiments.
      References:
      [Aminikhanghahi & Cook, 2017](https://doi.org/10.1007/s10115-016-0987-z);
      [Van den Burg & Williams, 2020](https://arxiv.org/abs/2003.06222);
      [Tatbul et al., 2018](https://arxiv.org/abs/1803.03639);
      [Prabuchandran et al., 2022](https://doi.org/10.1007/s10489-021-02321-6).
- [ ] Paper revision &mdash; statistical methodology: add non-parametric
      significance testing, confidence intervals, and hypothesis tests for
      performance differences across multiple runs
- [ ] Paper revision &mdash; performance metrics section:
  - Detection accuracy (TPR and detection delay)
  - False alarm characterisation (FPR and significance)
  - Computational efficiency (runtime and memory)
  - Theoretical validation (convergence and stability)
- [ ] ECC26 revision &mdash; regenerate figures with corrected axis labels
      and legends
      ([response.md TODOs](publications/ECC26/reviews/response.md) &mdash;
      Reviewers 3, 7, 13)
- [ ] Resolve notation consistency across paper sections
      (e.g. ODMD-CPD_diff variant)
- [ ] Fill in `pyproject.toml` project description (currently placeholder)
