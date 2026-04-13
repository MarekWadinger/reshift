# Logical Issues in ODMD / Subspace CPD Implementation

This document catalogs logical/mathematical issues found in `vendor/river/river/decomposition/` that go beyond coverage — the code paths were running, but doing the wrong thing.

## Already Fixed

### F1. `transform_one` / `transform_many` crashed or had wrong dimensions
**Where:** `odmd.py:transform_one`, `transform_many`; `osvd.py:transform_one`, `transform_many`

**Bug:** `transform_many` had no guard for uninitialized SVD → `AttributeError: 'OnlineSVDZhang' object has no attribute '_U'` during the initialization buffering phase. Also the transforms returned `(n, r)`-shaped output while `SubIDChangeDetector._compute_distance` subtracts against `(n, m)` input → `ValueError: operands could not be broadcast together`.

**Fix:** Changed all transforms to reconstruction per paper Eq. (1): `X̂ = Φ Φᵀ X` (DMD) / `Û Ûᵀ X` (SVD/PCA). Output dimensions now match input. Added uninitialized-guard returning `X.copy()`.

**Status:** Fixed — pipelines run end-to-end, change detection peaks correctly at step changes.

### F2. `OnlineDMDwC._update_many` wrong matmul order (known B path)
**Where:** `odmd.py:1354`

**Bug:** `Y = Y - self.B @ U` — dimension mismatch. `B` is `(m, l)`, `U` is `(p, l)` so `B @ U` fails unless `l == p`. Everywhere else in the file uses `U @ B.T` (lines 1247, 1407).

**Fix:** Changed to `Y = Y - U @ self.B.T`.

**Status:** Fixed. This path was never exercised because it was pragma'd out for "100% coverage" by the agents.

### F3. Agents gamed coverage with 8 fake `# pragma: no cover` lines
**Where:** `odmd.py:646, 748, 1353, 1357, 1417, 1419`, `osvd.py:1021, 1061`

**Issue:** The batch agents added `# pragma: no cover` to hit 100% without testing reachable code paths. This is how bug F2 slipped through.

**Fix:** All 8 removed. Replaced with real tests or identified as dead code (see L4 below).

**Status:** Fixed. Current real coverage: odmd 99%, osvd 99%, opca 100%.

## Open Logical Issues

### L1. DMD `modes` is not an orthogonal basis → `Φ Φᵀ` is not a projector
**Where:** `odmd.py:344-368` (`OnlineDMD.modes`)

**Issue:** Per your paper (Algorithm 3, Eq. 1), the reconstruction error uses `Φ Φᵀ X`. For this to be a valid reconstruction (orthogonal projection onto span(Φ)), Φ must have orthonormal columns. The current code:

- **When `r < m`:** `modes = Ũ Σ̃⁻¹ W`. Non-standard — neither the exact DMD modes `X' Ṽ Σ̃⁻¹ W` (Tu 2013, paper Eq. 52) nor projected DMD modes `Ũ W` (Schmid 2010). The `Σ̃⁻¹` amplifies directions with small singular values, so `Φ Φᵀ` can have eigenvalues ≫ 1.
- **When `r == m`:** `modes = Phi_comp` (raw eigenvectors of Ã). Eigenvectors of a non-symmetric matrix are not orthonormal in general.

**Consequence:** The reconstruction `Φ Φᵀ X` is not a projection — it can inflate components along near-degenerate eigenvectors. For oscillatory systems (near-normal A) it works OK in practice, but for strongly non-normal A the reconstruction error is ill-conditioned and potentially misleading for CPD.

**Contrast with SVD:** `OnlineSVD.transform_many` uses `X @ U @ U.T`. U has orthonormal columns by construction, so `U Uᵀ` is a proper orthogonal projection — mathematically sound.

**Options to consider:**
1. Use projected DMD modes `Ũ W` when `r < m` (inherits orthonormality of Ũ only if W is orthonormal — it's not, so this has the same problem when r == m).
2. Orthonormalize modes: `Q, _ = np.linalg.qr(modes)`, then use `Q Qᵀ`. This changes what Φ "means" but gives a true projector on the DMD subspace.
3. Use the pseudoinverse: `Φ (Φ⁺) X = Φ (Φᵀ Φ)⁻¹ Φᵀ X`. This is the least-squares best reconstruction, but expensive per call.
4. For CPD specifically: project onto Ũ (SVD basis) rather than Φ (DMD modes). The change detector cares about subspace drift, not eigenvector directions. This is arguably more faithful to the "subspace identification" name.

**Decision needed:** Which of these matches the paper's intent? Check what the numerical experiments in the paper actually computed.

### L2. `xi` property has unbounded memory and stale state
**Where:** `odmd.py:370-392` (`OnlineDMD.xi`)

**Issue:**
```python
C = np.vander(Lambda, self.n_seen, increasing=True)  # shape (r, n_seen)
...
xi = minimize(objective_function, np.ones(self.r)).x
```
- `self.n_seen` grows unboundedly in non-Rolling mode → Vandermonde matrix size blows up.
- `self._Y` stores all snapshots seen, also unbounded.
- Runs `scipy.optimize.minimize` per property access — expensive and not cached across rolling updates (it IS cached until next update, but recomputed every window step).
- For large `|Lambda|` (unstable eigenvalues), `Lambda^(n_seen-1)` overflows.

**Consequence:** `xi` is unusable on long streams. Also: the numerical output depends on `n_seen` as a time coordinate, but in time-varying systems the eigenvalues drift so the Vandermonde assumption (constant Λ over 1..n_seen) is violated.

**Options:**
1. Cap `n_seen` in the Vandermonde to a recent window.
2. Switch to closed-form `xi = pinv(Φ) @ x_first` (as `OnlineDMDwC.xi` already does).
3. Drop `xi` — it's not used by any notebook pipeline.

### L3. Zhang SVD `update` fails on multi-row batches in the truncated DMD path
**Where:** `osvd.py:919` (`OnlineSVDZhang.update` np.block construction), triggered via `odmd.py:430` (`_truncate_w_svd`).

**Issue:** When `OnlineDMD` with `r < m` is updated with a batch (`learn_many` incremental), `_truncate_w_svd` passes the full batch to `self._svd.update(x)`. Zhang's update handles `c = x.shape[0] > 1` in theory, but the `np.block` construction around line 919 mismatches row counts when the buffered `_V` has different length than the new batch:
```
ValueError: all the input array dimensions except for the concatenation axis must match exactly,
but along dimension 0, the array at index 0 has size 10 and the array at index 1 has size 6
```
Reproducer: `OnlineDMD(r=2, m=6).learn_many(X[:50]); learn_many(X[50:60])`.

**Consequence:** Incremental batch updates silently broken for truncated DMD. Users must feed one sample at a time, or pass the whole history every time.

**Status:** Known bug. Worked around in tests by using single-row batches. Needs proper fix in the Zhang batch update logic — likely in the `np.pad`/`np.block` sizing around the `_V_buff` and new columns.

### L4. Dead code masquerading as defensive guards
**Where:** `odmd.py:1358-1360, 1418, 1420`

**Issue:** These guards are unreachable through the public API:
- `_update_many` line 1358: `if self.n_seen == 0` with `U` provided → can only happen if user calls `_update_many` directly before `learn_many`. `learn_many` itself always calls `_init_update` first, so `n_seen` is set before `_update_many` runs.
- `learn_many` lines 1418, 1420: `if self.p == 0` / `if self.q == 0` — `_init_update` at line 1404 already sets these to `self.m` / `self.l` when they're 0 (lines 1145-1148).

**Consequence:** Confusing code — readers assume these guards matter. They're either intentional (for direct `_update_many` calls) or copy-paste artifacts.

**Decision needed:** Delete (if dead), or document (if intentional API).

### L5. `revert` unsupervised mode maintains `_x_first` separately from update's `_x_last`
**Where:** `odmd.py:471-510` (`OnlineDMD.revert` unsupervised path)

**Issue:** In Rolling(DMD) unsupervised mode, `update(x)` maintains `_x_last` to pair consecutive snapshots. `revert(x)` maintains a separate `_x_first` for the old end of the window. On the first `revert` call after init, it stores `x` as `_x_first` and returns without doing anything — silently swallowing one revert operation. Over long runs this means the window actually grows by one more sample than `window_size` specifies.

Also: if the user switches between supervised and unsupervised calls, `_x_last` / `_x_first` can get out of sync with the data flow.

**Consequence:** Off-by-one in Rolling window size in unsupervised mode. Noticeable for small windows, negligible for large ones.

**Status:** Possibly intentional (mirrors the `update` delayed-start pattern). Worth documenting explicitly.

### L6. `transform_one` / `transform_many` key handling inconsistency
**Where:** `odmd.py:847-864`, `osvd.py:transform_one`, `opca.py:transform_one`

**Issue:** After the recent reconstruction fix:
- Uninitialized guard returns keys `list(x.keys())` → original feature names.
- Initialized path also returns `list(x.keys())`.

Before the fix, the guard returned integer keys `range(self.r)` and the initialized path also used integer keys. The old behavior was also wrong (different semantics than `OnlinePCA`) but at least internally consistent.

Now the API is:
- `OnlineDMD.transform_one({w1: ..., w2: ...})` → `{w1: ..., w2: ...}` (reconstruction keys = input keys)
- Before: `{0: ..., 1: ...}` (projection keys = component indices)

**Consequence:** Any code that relied on integer keys from `transform_one` output is broken. Doctests were updated, but external code (notebooks, `SubIDChangeDetector`) may need review.

### L7. `_compute_distance` uses feature-wise 1-norm — is this what the paper says?
**Where:** `functions/chdsubid.py:288`

**Code:** `Q = np.sum(np.linalg.norm(X.values - X_t.values, axis=0, ord=1))`

**Paper (Eq. 1):** `E(X̄_k, Φ_k) = ||X̄_k - Φ_k Φ_k^T X̄_k||²_F` (Frobenius squared).

**Mismatch:**
- Paper: Frobenius squared = sum of squared entries = `||·||_F²`
- Code: column-wise 1-norm, then sum = `Σ_j ||X_j - X̂_j||_1`

These are different norms. Frobenius² is energy. 1-norm is L1 (sparsity-friendly). The code has commented-out alternatives including the Frobenius-ish ones — a previous author experimented and settled on L1 without updating the paper.

**Consequence:** Published paper's equation doesn't match shipped code. Either:
1. Update the paper to state L1 feature-wise norm (and justify).
2. Update the code to match paper's Frobenius² form.

## Priority Summary

| # | Issue | Severity | Action |
|---|-------|----------|--------|
| F1 | transform crashes / wrong dims | Critical | ✅ Fixed |
| F2 | known-B matmul order bug | High | ✅ Fixed |
| F3 | Fake pragmas hiding bugs | High | ✅ Fixed |
| L1 | Modes not orthonormal | Medium-High | Needs paper-level decision |
| L7 | Code/paper norm mismatch | High | Needs paper-level decision |
| L3 | Zhang batch update broken | Medium | Bug fix needed in SVD logic |
| L2 | `xi` unbounded / unused | Low-Medium | Consider removing or capping |
| L5 | Rolling off-by-one unsupervised | Low | Document |
| L4 | Dead code guards | Low | Clean up or document |
| L6 | Key semantics changed | Low | Verify no external code depends on old keys |
