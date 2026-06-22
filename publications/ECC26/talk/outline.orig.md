# Talk Outline

Why? - industry needs adaptable interpretable change detection relevant throughout the system lifetime. Current methods fail in either one of (interpretable, adaptable, robust, control-aware)
How? - creating stable version of the current SOTA fullfilling params which only suffered from robustness issue
What? - toDMDc, state of the art performance on both simulated and real data

## Motivation (Bridge + the question (control vs. fault)) (5 min)

### Industry

- Setup: Control and monitoring over the process lifetime, but that doesn't work big time because the system changes and the deployed systems not.

### Why do systems change?

" What are the changes? - lasting or transient states, not necessarly bad but need to be in the known."

- reference - ok, known
- input - ok, known
- aging - ok, unknown
- seasonalities - ok, unknown
- faults and transient states - nok, unknown

"Are changes bad? Any change that is not anticipated could be"

### How to react on changes?

- identify it ok/onk
  - ok: keep going
    - do what you did, keep systems acitve and up to date (informed)
  - nok: stop

"Usually system do not remain functional after unintended change is detected"

For anomalies this is resolved [say how simply]

### What is out there

| Method | Interpretability | Adaptability | Control-Awareness | Robustness |
|--------|------------------|--------------|-------------------|------------|
| M1     | ✅               | ❌           | ❌                | ❌         |
| M2     | ✅               | ✅           | ❌                | ❌         |
| oDMDc  | ✅               | ✅           | ✅                | ❌         |

### Caveat of existing SOTA

oDMDc is a good take, but lacks robustness
[plot second best SOTA after us and show where it fails]

### Quick show results

[plot our method best SOTA, same data, problem solved]

## Method

### Our Contribution - truncation of oDMDc

| Method | Interpretability | Adaptability | Control-Awareness | Robustness |
|--------|------------------|--------------|-------------------|------------|
| Ours   | ✅               | ✅           | ✅                | ✅         |

- How we truncate
- Briefly about how it serves change detection (bridge, but not our main contribution)

### Why it works for CPD

### Case study (show only one)

## Conclusion

- results in numbers and implications
- github open source, easy install

## Q&A
