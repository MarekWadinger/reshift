# Review: ECC26 Revision Audit -- Truncated Online DMD with Control for Industrial Change Detection

## Summary

This audit cross-checks the authors' response to five reviewers (response.md) against the actual revised manuscript (root.tex) and the included figures. The goal is to verify that items marked DONE are genuinely implemented, that remaining TODOs are acknowledged, and that no gap exists between promises and reality that would jeopardize acceptance of the camera-ready version.

---

## 1. Quality of Conceptual Clarifications

### 1.1 Contributions (R7 concern) -- VERIFIED

The Contributions subsection (root.tex lines 78--86) now explicitly states: "While the individual components -- online DMDc [9] and online SVD [10] -- exist separately, our contribution is their integration with basis-alignment updates and a CPD framework." This directly addresses R7's concern that the technical contribution was "subtle." The delineation is clear and honest.

**Verdict: Adequately addressed.**

### 1.2 Window Temporal Relationships (R7 concern) -- VERIFIED WITH CAVEAT

Line 170 states "ordered chronologically as learning, base, and test" and the bullet list (lines 171--175) gives the chronological ordering. However, the separation parameter `b` is only described textually as the gap "separated from the test window by b samples" inside the base window bullet. There is no diagram or timeline figure. For a 6-page conference paper this is acceptable, but the description still requires careful reading -- a reviewer might ask: does `b` separate the *end* of base from the *start* of test, or the *start* of base from the *start* of test? The current wording ("separated from the test window by b samples") is slightly ambiguous. Given that `b` appears in the parameter list of Algorithm 1 but never in an equation, this should be clarified in one sentence.

**Verdict: Partially addressed. The ordering is clear; the precise definition of `b` remains slightly ambiguous.**

### 1.3 Notation Changes -- sigma/omega vs a/b (R7 concern) -- VERIFIED

Line 107: "lambda_i = sigma_i + i*omega_i, where sigma_i indicates growth/decay and omega_i the oscillation frequency." This replaces the old `a_i + i*b_i` notation and avoids the clash with window parameters `a, b`. Confirmed in both the prose and the eigendecomposition passage.

**Verdict: Fully addressed.**

### 1.4 Eigendecomposition Phi vs Lambda (R3 concern) -- VERIFIED

Lines 102--105 now write the eigendecomposition explicitly as `A_tilde W = W Lambda` with `Lambda = diag(lambda_1, ..., lambda_r)`, then separately define Phi = X' V Sigma^{-1} W. The distinction is unambiguous.

**Verdict: Fully addressed.**

---

## 2. Theoretical Claims -- Softening and Justification

### 2.1 Detection Delay (R2, R3 concerns) -- VERIFIED

Line 232: "The peak of Q_k occurs *approximately* c samples after the true change-point" (softened from "exactly"). Same line continues: "This provides a deterministic lower bound on detection delay, with the actual delay depending on noise level and change magnitude." The word "lower bound" replaces the erroneous "upper bound."

However, there is an inconsistency: line 82 in the Contributions still says "bounded detection delays *equal to* the test window size." This is stronger than "approximately c" and contradicts the softening. This should be harmonized -- either say "bounded detection delays on the order of the test window size" or "approximately equal to."

**Verdict: Mostly addressed but an inconsistency at line 82 undermines the softening.**

### 2.2 Convergence and Stability (R2, R3, R7, R10) -- VERIFIED

The standalone "Convergence and Stability Analysis" subsection with formal Theorem 1 has been removed. What remains is a concise paragraph (lines 234--235) citing [9] for convergence of the online updates. This is appropriate for a conference paper and follows R3's own suggestion.

**Verdict: Fully addressed.**

### 2.3 Interpretability Calibration (R2 concern) -- VERIFIED

Lines 236--237: "While Q_k does not have an absolute scale (its magnitude depends on the operating regime), it provides a consistent ranking of change severity. In practice, calibration via historical Q_k values during normal operation is recommended for threshold selection." This directly addresses R2's complaint that Q_k is not interpretable without calibration.

**Verdict: Fully addressed.**

### 2.4 Detection Guarantees Removal (R2 concern) -- VERIFIED

The formal detection guarantee equation with unspecified delta, tau, alpha has been removed. The convergence claim is now a prose statement citing [9]. This is a significant improvement -- the prior version made formal claims without providing the machinery to back them up.

**Verdict: Fully addressed.**

---

## 3. Figures -- CRITICAL OUTSTANDING ISSUES

This is the most significant gap between the response and the actual manuscript. The response.md lists four figure-related items as TODO (Phase 3), and inspection of the actual PDF figures confirms none have been resolved:

### 3.1 Y-axis Labels Still Say "score" Not "statistic"

All three figures use y-axis labels "ODMD-CPD score [-]", "ODMD-CPD_diff score [-]", and "OSVD-CPD score [-]". The response (item #3, #24) claims all instances of "score" were replaced with "statistic." The manuscript text is consistent (line 194: "detection statistic", line 280: "detection statistic"), but the **figures themselves have not been regenerated**. This is a direct contradiction: the text says "statistic" while the plots say "score."

**Severity: HIGH. The figures contradict the text and the response claim.**

### 3.2 Fig. 2 Legend Shows "X" for Both Tank Levels

In the two-tank figure, the top subplot legend shows two entries both labeled "X" (one blue, one orange). The response acknowledges this (TODO: "Fix Fig. 2 legend") but it remains unfixed. R3 explicitly asked for "a proper legend distinguishing h_1 and h_2" and R7 noted "the two colours in the first subplot are both labelled as X."

**Severity: HIGH. Two reviewers flagged this; it remains broken.**

### 3.3 Red/Dashed Lines Not Explained in Captions

All three figures contain solid red and dashed red vertical lines. These presumably mark true change-points and detected change-points (or change-point boundaries), but neither the figures nor the captions explain them. The Fig. 1 caption (line 287) says "peaks close to 100 samples after each true change-point" but never states what the red lines represent. Fig. 2 and Fig. 3 captions are similarly silent.

**Severity: MEDIUM-HIGH. This was flagged by R7 and remains unresolved.**

### 3.4 No X-axis Labels with Real Time Context

R13 asked for x-axis labels relating samples to real time. The synthetic and two-tank figures show only sample indices (0--10000, 0--12000). The BESS figure also shows only sample indices (0--~21000) with no time reference. The BESS case text mentions "30-second intervals" but the figure does not reflect this.

**Severity: MEDIUM. R13 flagged this; it remains a TODO.**

### 3.5 Fig. 1 Has Four Subplots But Caption Describes Three

The synthetic figure (Fig. 1) actually has four subplots: (1) signal, (2) ODMD-CPD score, (3) ODMD-CPD_diff score, (4) OSVD-CPD score. The caption at line 287 says "middle two plots" which is correct but does not explain what the third subplot (ODMD-CPD_diff) represents. The text never defines or mentions this "diff" variant anywhere in the manuscript. This is confusing to a reader.

**Severity: MEDIUM. An unexplained subplot is worse than no subplot.**

### 3.6 Synthetic Figure Caption Claim Mismatch

Line 287 says the statistic "peaks close to 100 samples after each true change-point" (revised from "peaks exactly 100"). However, looking at the figure, the first change-point around sample 1000 produces a peak that visually appears well before sample 1100. The claim of "close to 100" seems reasonable for most change-points but could be tightened -- the text at line 282 gives the more precise "102 +/- 3 samples (mean +/- std)" for change-points 2--9, which is better.

**Verdict: Acceptable for the text; the caption is slightly vague but not wrong.**

---

## 4. Remaining Text Issues

### 4.1 "Score" vs "Statistic" in Discussion

Line 327 uses "interpretable statistics" (correct). Line 335 uses "detection statistic" (correct). All in-text occurrences appear consistent. The only remaining "score" instances are in the figures (see 3.1 above).

**Verdict: Text is clean; figures are not.**

### 4.2 Undefined Abbreviations

- "POD" replaced with "left singular vectors" -- confirmed, not found in text.
- "HVAC" replaced with "cooling system" -- confirmed at line 312.
- "BESS" defined in section title at line 310 -- confirmed.

**Verdict: Fully addressed in text.**

### 4.3 DMDc Restructuring (R2 concern) -- VERIFIED

Section II-B (lines 109--121) now presents the unknown-B augmented formulation as primary, with a single sentence for the known-B case: "When B is known, one can instead compensate..." This is a clean restructuring.

**Verdict: Fully addressed.**

### 4.4 Basis-Alignment Motivation (R3 concern) -- VERIFIED

Line 152: "Since the online SVD update changes the basis U_k, the existing reduced-order matrices must be re-expressed in the new coordinates." This provides the missing motivation.

**Verdict: Fully addressed.**

### 4.5 Singularity Remark for K^{U'U} (R3 concern) -- VERIFIED

Lines 157--158: "K^{U'U}_k is well-conditioned when the truncation rank r <= min(m, l), as it is a submatrix of a unitary rotation; near-singularity would indicate an abrupt subspace change, which is itself a detectable event."

**Verdict: Fully addressed.**

### 4.6 Gamma Subscript Fix (R3 concern) -- VERIFIED

Lines 133--136 now use Gamma_{k:k+c} consistently. The old Gamma_{k+c} has been corrected.

**Verdict: Fully addressed.**

### 4.7 Algorithm 1 Negative Weights Clarification (R3 concern) -- VERIFIED

Line 223: "Revert old snapshots via (8)--(9) with -C". This replaces the vague "negative weights" phrasing.

**Verdict: Fully addressed.**

### 4.8 Section Renamed to Numerical Validation (R7 concern) -- VERIFIED

Line 250: "\section{Numerical Validation}". Opening sentence (line 252): "two simulation studies and one real-data case study."

**Verdict: Fully addressed.**

### 4.9 Citation [10] Updated (R3 concern) -- VERIFIED

The bib entry for Zhang2022 (main.bib line 241--249) now includes archiveprefix = {arXiv}, eprint = {2204.05398}.

**Verdict: Fully addressed.**

### 4.10 Typo Fixes -- VERIFIED

- "sysntem" is not present in root.tex. Fixed.
- "roost cause" is not present. Fixed.

**Verdict: Fully addressed.**

---

## 5. Structural Tightening (R10 concern)

### 5.1 Page Count

The manuscript appears to fit within 6 pages (typical ECC limit). The removed sections (Convergence & Stability subsection, Applicability subsection, known-B equation) and compressed Discussion achieve the space savings claimed.

**Verdict: Likely addressed, but a compiled PDF should be checked.**

### 5.2 Applicability Subsection Removed -- VERIFIED

No "Applicability" subsection exists in the current manuscript. The Discussion section has only two subsections: "Comparison with State-of-the-Art" and "Practical Deployment Considerations."

**Verdict: Fully addressed.**

---

## 6. Items Marked TODO in response.md That Remain Unresolved

| # | TODO Item | Reviewer | Severity |
|---|-----------|----------|----------|
| 1 | "Consider adding a brief remark on limitations of the method: what magnitude of changes can be detected depending on rank and window sizes" | R2 | MEDIUM -- this is a substantive gap; even one sentence would help |
| 2 | "Consider strengthening the stochastic perspective in theoretical properties" | R2 | LOW-MEDIUM -- deferred to journal is defensible for ECC |
| 3 | "Consider adding 1-2 sentences of CPD motivation in the introduction" | R2 | LOW -- the intro already motivates CPD adequately at lines 53--57 |
| 4 | "Consider a brief mention of time-delay embedding in Preliminaries" | R2 | LOW -- it appears in Section II-D implicitly; a forward reference would help |
| 5 | "Consider adding a notation summary or clarifying inline" | R2 | LOW -- sigma/omega rename largely addresses the confusion |
| 6 | Fix all figure issues (x-axis labels, legends, red line explanations) | R3, R7, R13 | **HIGH** |

---

## 7. Gap Analysis: Promises vs. Reality

### Claims in response.md NOT reflected in root.tex

1. **"ODMD-CPD score" replaced with "ODMD-CPD statistic" (3 occurrences)** -- True in text, FALSE in figures. The y-axis labels in all three figures still read "score."

2. **"peaks exactly 100" changed to "peaks close to 100" in Fig. 1 caption** -- True in caption text (line 287). But the figure itself was not regenerated, so the visual presentation is unchanged.

3. **Line 82 inconsistency**: Contributions say "bounded detection delays equal to the test window size" while Theoretical Properties (line 232) says "approximately c samples." These need to be reconciled.

### Items genuinely implemented and verified (20 of 24 completed items)

All items in the "Completed" table of response.md are verified in root.tex EXCEPT:
- Item #3 and #24 ("score" -> "statistic"): only in text, not in figures
- Item #6 ("peaks close to 100"): text only; figure not regenerated

---

## 8. Additional Issues Found During Review

### 8.1 Abstract Still Says "three case studies"

Line 45: "three case studies: synthetic step changes, nonlinear two-tank system with input delays, and industrial battery energy storage system." The Contributions (line 84) says "two case studies." These are contradictory. The abstract includes the synthetic benchmark as a case study; the Contributions section excludes it. This should be made consistent -- probably "three" is correct since Section IV has three subsections.

### 8.2 ODMD-CPD_diff Not Defined Anywhere

Figures 1 and 3 contain subplots labeled "ODMD-CPD_diff score" but this variant is never defined in the text. The BESS caption (line 319) calls it "Error difference formulation (bottom) shows bidirectional detection" but the formulation is not given. Either remove these subplots or add a one-line definition (e.g., Q_diff = E_T/c - E_B/a).

### 8.3 Table Still Says "experimental validation"

Line 257: "Summary of experimental validation case studies" -- should be "numerical validation" to match the section title change (item #23 in response).

### 8.4 Number of Case Studies in Contributions vs. Section IV

Contributions (line 84) list only two case studies: "(i) a nonlinear two-tank system" and "(ii) an industrial battery energy storage system." But Section IV has three subsections including the synthetic benchmark. Either add the synthetic case to the contributions list or explain why it is excluded.

### 8.5 Missing Time-Delay Embedding in Preliminaries

Time-delay embedding (Hankelization) is used in Algorithm 1 (line 208) and in the parameter guidelines (line 244) but never formally defined. The Preliminaries section jumps from online DMD to truncated online DMD without covering this. R2 flagged this ("Time delay embeddings seem fairly important in later sections, but they are not mentioned at all here"). It remains a TODO.

---

## Recommendation

**MINOR REVISION** -- but the figure issues must be resolved before camera-ready submission.

The text-level revisions are thorough and genuinely address the major reviewer concerns: contribution clarity, notation conflicts, theoretical softening, structural tightening, and terminology consistency. The authors have done substantial and honest work on the manuscript body.

However, the figures are a blocking issue:
1. All y-axis labels still say "score" instead of "statistic" -- directly contradicting the text
2. Fig. 2 legend still shows "X"/"X" instead of h_1/h_2
3. Red vertical lines remain unexplained in all captions
4. An unexplained "ODMD-CPD_diff" subplot appears in Figs. 1 and 3 without any textual definition
5. X-axis labels lack real-time context (R13)

These are not cosmetic -- they affect interpretability and self-containedness of the figures, which multiple reviewers explicitly flagged. The figures must be regenerated with corrected labels before the camera-ready deadline.

### Priority Actions

1. **[BLOCKING]** Regenerate all three figures with "statistic" replacing "score" in y-axis labels
2. **[BLOCKING]** Fix Fig. 2 top subplot legend: "X"/"X" -> "h_1"/"h_2"
3. **[BLOCKING]** Add red-line legend or caption explanation to all figures (e.g., "Solid red: true change-point; dashed red: detection window boundaries")
4. **[HIGH]** Either define ODMD-CPD_diff in the text or remove those subplots from Figs. 1 and 3
5. **[HIGH]** Reconcile "equal to" (line 82) with "approximately" (line 232) in detection delay claims
6. **[HIGH]** Fix Table I caption: "experimental" -> "numerical"
7. **[MEDIUM]** Reconcile abstract "three case studies" with Contributions "two case studies"
8. **[MEDIUM]** Add x-axis real-time context per R13
9. **[LOW]** Add a sentence on detection sensitivity vs. change magnitude (R2 limitation remark)
10. **[LOW]** Add a brief forward reference to time-delay embedding in Preliminaries
