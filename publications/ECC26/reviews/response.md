# Response to Reviewers — ECC 2026 Submission 701

We thank the Associate Editor and all reviewers for their constructive feedback. Below we address each comment individually. Changes in the revised manuscript are marked in blue.

---

## Associate Editor

> Five reviews have been obtained for that paper, and they all highlight the quality of the work. There are, however, some suggestions that would greatly improve the paper, and I suggest you take them into account for the final version.

We thank the Associate Editor for the positive assessment. We have carefully addressed all reviewer suggestions as detailed below.

---

## Reviewer 2 (Review 4213)

> My main criticism towards the paper is the lack of a better insight on the limitations of the method: what magnitude of changes can be detected depending on the rank and window sizes.

TODO: Consider adding a brief remark in the Discussion or Parameter Selection section.

> In general, I feel the authors do not address randomness enough: it is included in the examples, and considered in the parameter delta of the detection guarantee, but I think it should have a more central role in a changepoints detection algorithm.

TODO: Consider strengthening the stochastic perspective in the theoretical properties section.

> My second criticism is towards the claim that statistic Q_k obtained from the algorithm is interpretable: for a user, there is no explanation on what values to expect a priori, and interpreting them is then impossible, beyond the knowledge that higher means more likely to be a changepoint.

DONE: Revised the Interpretability property in Section III-C to explicitly acknowledge that Q_k has no absolute scale (magnitude depends on the operating regime) and that calibration via historical Q_k values during normal operation is recommended for threshold selection.

> **I Introduction** — The paper does a good job at introducing first subspace based changepoints detection and then online DMD for change detection. However, there is not that much motivation about detection of changepoints.

TODO: Consider adding 1-2 sentences of CPD motivation in the introduction.

> **II Preliminaries** — When mentioning DMD with control, I feel it is not a good idea to mention both cases of input being known or unknown, as the reader might feel confused on which case applies to the paper.

DONE: Restructured Section II-B to present the augmented (unknown B) formulation as the primary case, with the known-B compensation noted briefly as a special case in one sentence.

> Time delay embeddings seem fairly important in later sections, but they are not mentioned at all here.

TODO: Consider a brief mention of time-delay embedding in Preliminaries.

> Too many hats and tildes, I think the authors could make the meaning of each matrix more clear.

TODO: Consider adding a notation summary or clarifying inline.

> **III Algorithm** — The claim about the detection delay is dubious, specially in the use of "exactly". As mentioned before, the kind of randomness affecting the signal is not really mentioned.

DONE: Changed "exactly" to "approximately" and clarified that `c` is a deterministic lower bound on detection delay, with actual delay depending on noise level and change magnitude (Section III-C).

> The interpretability is also questionable: pretty much every method that relies on residuals can claim this, but without a way of scaling the metric, it is hard to make any interpretation from the numbers.

DONE: Acknowledged in the revised Interpretability property that Q_k provides relative (not absolute) severity ranking and requires calibration.

> The parameter selection guidelines are really appreciated.

Thank you.

> The detection guarantees do not seem very specific without any information about delta, tau and alpha.

DONE: Removed the formal detection guarantee equation (which lacked concrete bounds) and folded the key convergence and stability claims into the Theoretical Properties subsection with a citation to [9]. Detailed proofs are deferred to the journal version.

> **IV Experimental Validation** — The plots, on the other hand, are not that useful. A parameter benchmark could be more informative to understand the effect of the different hyperparameters.

We appreciate the suggestion. A detailed parameter sensitivity study is deferred to the extended journal version due to space constraints. The parameter selection guidelines in Section III-D provide practical rules for tuning.

> **V/VI Discussion and Conclusion** — Typo: sysntem

DONE: Fixed "sysntem" to "system".

> I did not feel the applicability part gave any interesting insights.

DONE: Removed the Applicability subsection entirely to save space. The relevant points are already covered in the Introduction and Discussion.

---

## Reviewer 3 (Review 4215)

> Some argumentation for the transformation (12)-(13) would be appreciated. Why was it chosen, and for what specific purpose?

DONE: Added explanation that the online SVD update changes the basis, requiring re-expression of existing reduced-order matrices in the new coordinates.

> What would happen if m + l > p + q? It seems like the matrix being inverted in (13) could become singular in that case; does this influence the algorithm?

DONE: Added a remark that K^{U'U}\_k is well-conditioned when r <= min(m, l) as it is a submatrix of a unitary rotation, and that near-singularity would itself indicate a detectable abrupt subspace change.

> The theoretical properties stated in Section III-C would benefit from some argumentation. Perhaps they are better saved for a later journal version.

DONE: Following this suggestion, we removed the standalone Convergence and Stability Analysis subsection (including the formal Theorem 1 and Detection Guarantees with unspecified parameters). The key convergence and stability claims are now concisely stated in the Theoretical Properties subsection with a citation to [9]. Detailed proofs are deferred to the journal version.

> The author affiliations on p. 1 should be formatted according to IEEE conference standards.

The current formatting uses `\IEEEauthorrefmark` with `\IEEEauthorblockN`/`\IEEEauthorblockA`, which follows the standard IEEE conference template. We are not aware of a specific deviation; if the reviewer could clarify, we would be happy to adjust.

> In (3), Phi = Lambda?

DONE: Clarified by writing the eigendecomposition explicitly as `A_tilde W = W Lambda` with `Lambda = diag(lambda_1, ..., lambda_r)`, making the distinction between modes Phi and eigenvalues Lambda unambiguous.

> Some k's seem to be missing in (8)-(9).

DONE: Fixed inconsistent subscript — `Gamma_{k+c}` in eqs. (8)-(9) now reads `Gamma_{k:k+c}`, consistent with the definition in the text.

> "Revert old snapshots using negative weights" seems a bit too vague compared to the rest of the presentation.

DONE: Replaced with explicit reference: "Revert old snapshots via (6)-(7) with -C" in Algorithm 1.

> In Section III-B, shouldn't c be a deterministic lower bound?

DONE: Changed "upper bound" to "lower bound" in the detection delay property (Section III-C). Also softened "exactly" to "approximately" per the reviewer's earlier remark.

> In Fig. 2, please provide a proper legend for the states plot.

TODO: Add proper legend distinguishing h_1 and h_2 in Fig. 2.

> Please provide a proper citation for [10].

DONE: Updated citation [10] to include "arXiv preprint arXiv:2204.05398".

---

## Reviewer 7 (Review 8067)

> The technical contribution of the paper is a bit subtle to me. The DMD method is clearly not the contribution of the paper, and the modification of the DMD method in this paper is not clear to me. I can only guess that the paper employs the recursive SVD proposed by [10] for the SVD process of the original DMD method.

DONE: Revised the Contributions subsection to explicitly state that while the individual components (online DMDc [9] and online SVD [10]) exist separately, the contribution is their integration with basis-alignment updates and the CPD framework.

> In the change point detection algorithm, what is the temporal relationship between the base, test and learning phases? Are these three windows attached to the previous one consecutively?

DONE: Revised the window definitions in Section III-A to explicitly state the chronological ordering (learning, base, test) and clarify the separation parameter b between base and test windows.

> The unpresented convergence and stability analysis of the detection framework is also confusing to me. Are these results already established elsewhere, or do you omit the proofs for a reason?

DONE: Removed the standalone Convergence and Stability Analysis subsection. The convergence claim now cites [9] as its source. Detailed proofs for the complete framework are deferred to the journal version.

> The experimental validation part contains two simulations. The authors may want to clarify this because simulations are substantially different from experiments.

DONE: Renamed the section from "Experimental Validation" to "Numerical Validation" and clarified in the opening sentence that it comprises two simulation studies and one real-data case study.

> The plots need to be properly labelled. For example, the red lines in the plot are not explained. In Fig. 2, the two colours in the first subplot are both labelled as "X". In Fig. 3, the lines in different colours are not explained at all.

TODO: Fix plot labels — explain red lines in captions, fix Fig. 2 legend, add legend to Fig. 3.

> The use of the term "score" is confusing. I am not sure if this is the same as the so-called "statistic" as in (15).

DONE: Replaced all instances of "ODMD-CPD score" with "ODMD-CPD statistic" for consistency with eq. (15).

> I suggest that the authors double-check all abbreviations to make sure that they are defined before using them.

DONE: Replaced undefined "POD" with "left singular vectors", replaced undefined "HVAC" with "cooling system", defined "BESS" at first use in section title.

> There are also some reused symbols like "a" and "b", first defined as DMD mode parameters in Section II-A, and then as the hyperparameters for change point detection in Section III-A.

DONE: Renamed DMD eigenvalue components from `a_i + ib_i` to `sigma_i + i*omega_i` to avoid conflict with window size parameters `a, b`.

---

## Reviewer 10 (Review 17285)

> Some parts of the exposition are quite verbose and could really do with "tight" treatment. The theoretical sections are right but dense, and some transitions between derivations assume substantial prior knowledge, which may limit readability.

DONE: Removed the standalone Convergence and Stability Analysis subsection, folding key claims into the Theoretical Properties subsection. Added brief motivating sentences for the basis-alignment transformations.

> Similarly, the experimental section is well constructed but could be more concise without losing impact.

DONE: Renamed section to "Numerical Validation" and tightened the introductory text.

> I also suggest shortening the paper down to the suggested 6-page length of ECC site.

DONE: Removed: Convergence and Stability Analysis subsection (with 2 equations), Applicability subsection, known-B equation. Compressed: Discussion comparisons into a single paragraph, Practical Deployment into a single paragraph. Net savings: approximately 1 page.

---

## Reviewer 13 (Review 17291)

> The approach is compared only to SVD — a generic approach. Possibly it could be compared also to something more "industrialized", higher TRL, as there exist other change-point detection methods.

SVD-based CPD was chosen as the closest subspace method from the same methodological family. A broader benchmark against industrialized methods (e.g., CUSUM, kernel-based) is planned for the extended journal version.

> The charts do not have x-axis labeled — in the text it is said these are samples, but relation to real time span would be good in label or image description.

TODO: Add x-axis labels with sample count and real time context to all figures.

---

## Summary of Changes

### Completed

| #   | Change                                                                                               | Reviewers       |
| --- | ---------------------------------------------------------------------------------------------------- | --------------- |
| 1   | Fixed typo "roost cause" -> "root cause"                                                             | —               |
| 2   | Fixed typo "sysntem" -> "system"                                                                     | R2              |
| 3   | "ODMD-CPD score" -> "ODMD-CPD statistic" (3 occurrences)                                             | R7              |
| 4   | Renamed DMD eigenvalue symbols `a_i, b_i` -> `sigma_i, omega_i`                                      | R7              |
| 5   | Softened "exactly c samples" -> "approximately", changed "upper bound" -> "lower bound"              | R2, R3          |
| 6   | "peaks exactly 100" -> "peaks close to 100" in Fig. 1 caption                                        | R2, R3          |
| 7   | Algorithm 1: vague "negative weights" -> explicit equation reference with `-C`                       | R3              |
| 8   | "POD modes" -> "left singular vectors" (undefined abbreviation)                                      | R7              |
| 9   | "HVAC" -> "cooling system", added "(BESS)" at section title                                          | R7              |
| 10  | Citation [10]: added "arXiv preprint arXiv:2204.05398"                                               | R3              |
| 11  | Explicit eigendecomposition `A_tilde W = W Lambda` in eq. (2)-(3)                                    | R3              |
| 12  | Fixed `Gamma_{k+c}` -> `Gamma_{k:k+c}` in eqs. (8)-(9)                                               | R3              |
| 13  | Restructured DMDc: unknown-B primary, known-B as brief note                                          | R2              |
| 14  | Added basis-alignment motivation for transformations (12)-(13)                                       | R3              |
| 15  | Added singularity remark for K^{U'U}\_k                                                              | R3              |
| 16  | Removed Convergence & Stability subsection; folded into Theoretical Properties                       | R2, R3, R7, R10 |
| 17  | Removed Applicability subsection                                                                     | R2, R10         |
| 18  | Compressed Discussion comparisons into single paragraph                                              | R10             |
| 19  | Compressed Practical Deployment into single paragraph                                                | R10             |
| 20  | Softened interpretability claim; acknowledged Q_k needs calibration                                  | R2              |
| 21  | Clarified window temporal ordering (learning, base, test)                                            | R7              |
| 22  | Clarified contribution delineation (integration vs. components)                                      | R7              |
| 23  | Renamed "Experimental Validation" -> "Numerical Validation"; distinguished simulations vs. real data | R7              |
| 24  | Fixed remaining "detection score" -> "detection statistic"                                           | R7              |

### Remaining (by phase)

**Phase 2 — Notation & Math:** COMPLETED

- [x] Clarify Phi vs Lambda in eq. (3) (R3)
- [x] Add missing k subscripts in eqs. (8)-(9) (R3)
- [x] Simplify known/unknown B presentation (R2)
- [x] Add argumentation for transformations (12)-(13) (R3)

**Phase 3 — Figures:**

- [ ] Add x-axis labels to all figures (R13)
- [ ] Add proper legend to Fig. 2 (R3, R7)
- [ ] Add legend to Fig. 3 (R7)
- [ ] Explain red lines in captions (R7)

**Phase 4 — Structural tightening:** COMPLETED

- [x] Shorten to ~6 pages (R10)
- [x] Remove/condense Applicability subsection (R2, R10)
- [x] Trim Discussion comparisons (R10)
- [x] Trim convergence section, defer proofs to journal (R2, R3, R7)
- [x] Clarify temporal relationship of windows (R7)
- [x] Clarify contribution vs. preliminaries (R7)

**Phase 5 — Minor content:**

- [x] Soften interpretability claims (R2)
- [x] Distinguish "simulations" from "experiments" (R7)
- [x] Author affiliations: already using standard IEEE format; noted in response (R3)

### Not implementing (rebuttal justification)

| Request                                  | Justification                                                        |
| ---------------------------------------- | -------------------------------------------------------------------- |
| Parameter sensitivity benchmark (R2)     | Deferred to journal version; guidelines in III-D                     |
| Additional CPD method comparisons (R13)  | Deferred to journal version; SVD chosen as closest subspace baseline |
| Deep randomness/stochastic analysis (R2) | Ongoing work; formal detection power characterization planned        |
| Full convergence proofs (R3, R7)         | Deferred to journal version per R3's own suggestion                  |
