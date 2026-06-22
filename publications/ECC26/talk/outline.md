# ECC26 Talk — toDMDc for Industrial Change Detection

**Session:** WeB9 — Fault Detection and Tolerance II (slot WeB9.4)
**Format:** 15 min talk + 5 min Q&A → 11 slides, ~85 s slack banked (lands ~13:35 on paper)
**Spine (one idea, said 3× verbatim):** *The controller acting is not a fault. The system changing is.*
**One name everywhere:** call it **toDMDc** on every slide. Never switch to "ODMD-CPD" mid-talk.

---

## Delivery principles (read once before rehearsing)

1. **One idea per slide.** If the title needs two sentences, it's two slides.
2. **Spine line, verbatim, 3×** — slide 1 (pose), slide 6 (prove), slide 11 (pay off). Identical eight words each time. Do not paraphrase your own mantra.
3. **Calibrated, not confident.** This room punishes overclaiming and rewards honest bounds. Say "bounded by," "to first order," "suggestive." Never "exactly," "solved," "cross-validated."
4. **Two figures, two jobs:** two-tank = *proves the spine* (control vs fault on a controlled plant); BESS = *real-data win* (early warning). They are not redundant. Synthetic + math = backup.
5. **Avoid stage tongue-twisters:** never say "disambiguate" / "explaining away" live. Say "tell apart" / "account for the input."
6. **Numbers, not adjectives.** Every plot: walk over, point to the one thing, name it. Don't read axes aloud.

---

## SLIDE 1 — Cold open: the problem in one breath (0:00–1:00) · 60 s

**On screen:** A battery temperature trace creeping up — **axis labeled °C, titled "battery module temperature"** so the opening line confirms the image, not explains it. Title + authors small at the bottom. One line:
> *The controller acting is not a fault. The system changing is.*

**Say:**
- "This is a battery module. Its cooling system is starting to fail. The temperature is drifting up."
- "Right now, a standard monitor cannot tell that apart from the controller simply doing its job — a charge cycle looks like a fault, a fault looks like a charge cycle."
- Spine line. **Pause one beat.** "That sentence is the whole talk."

**Delivery:** No session housekeeping, no abstract. Stakes + spine in 60 s. The room must lean in before slide 2.

---

## SLIDE 2 — Why monitors decay (1:00–2:15) · 75 s

**On screen:** Lifetime timeline — model frozen at commissioning (t=0) while the plant drifts away. Gap widens; false-alarm rate rises.

**Say:**
- "We commission a monitor once. The plant then runs for *years* — it ages, re-seasons, gets re-tuned, gets new setpoints."
- "The model is frozen at day one. The gap between the plant today and the model grows. False alarms climb, real faults hide behind them."
- "So the requirement isn't a *better* detector. It's one that *stays current* — without a human re-tuning it."

**Audience hook:** the hydro-generator (.3) and causality (.6) papers both fight real industrial data that won't sit still. You're naming their pain.

---

## SLIDE 3 — Four things move the signal; three are normal (2:15–3:15) · 60 s

**On screen:** Two boxes only — **KNOWN / keep running** vs **UNKNOWN / act**. Inside, short labels (setpoint, **control input** | aging, seasonality, **fault**). No dense table.

**Say:**
- "Four things move a signal. Three are normal."
- "The setpoint and the **control input** — known, expected, not a fault. Aging and seasonality — slow, also not a fault, but the model must *adapt* to them."
- "Only the last one — an unexplained change in how the system responds — is a fault. A signal-watcher can't tell the control input apart from the fault. They look identical. *That* is the gap."

**Delivery:** This sets up the spine. Don't rush the last sentence.

---

## SLIDE 4 — The trap, in one picture (3:15–4:15) · 60 s

**On screen:** One trace. Control step → big swing (label **NORMAL**). Later, *same* input → *different* response (label **FAULT**). A flat threshold fires on both.

**Say:**
- "Here's the trap. The controller commands a step — the output swings hard. Nothing is wrong."
- "Later, the *same* command — and the response is different. *That's* the fault. The output level alone can't separate them."
- "SPC, CUSUM, plain reconstruction error — they watch the signal, so they alarm on both. We need to watch the *map* from input to state."

**Delivery:** Point physically to the two events. This slide earns the method.

---

## SLIDE 5 — One missing box (4:15–5:10) · 55 s

**On screen:** Trimmed matrix — 3 rows, honest labels.

| Method | Interpretable | Adaptive | Control-aware | **Numerically stable** |
|---|:--:|:--:|:--:|:--:|
| Statistical (SPC, CUSUM) | ✅ | ❌ | ❌ | ✅ |
| **Online DMDc** [Zhang2019] | ✅ | ✅ | ✅ | ❌ |
| **toDMDc (ours)** | ✅ | ✅ | ✅ | ✅ |

**Say:**
- "Read the columns as requirements: operators must trust it (interpretable), it must stay current (adaptive), it must account for the input (control-aware), and it must not blow up numerically."
- "Online DMDc had the first three. One box open — it goes numerically unstable on real data, and that single failure is what kept it unreliable."
- "We close that box. That's the contribution — and the rest of this talk."

**Delivery (credibility):** Column says **numerically stable**, *not* "robust" — that is exactly what you prove and nothing more. Row says **Online DMDc [Zhang2019]**, not "subspace" — you are not claiming to beat N4SID. Don't add a deep-learning row; you don't need to dunk on four families.

---

## SLIDE 6 — The idea: watch the map, not the signal (5:10–6:35) · 85 s · **CORE**

**On screen:** The controlled model `x_{k+1} = A x_k + B θ_k`. Beside it, the two-window intuition: a reference error vs. a test error → ratio `Q_k`.

**Say:**
- "We fit a controlled linear model online — next state from current state *and the known input*. Because the input is *inside* the model, a control-driven swing is *predicted* — and a predicted swing barely costs anything."
- "Only an excursion the model can't explain with the known input raises the score `Q_k`. Same dynamics → reference and test errors match → Q ≈ 0. The map moved → test error jumps → Q > 0."
- **Spine line, verbatim:** "We accounted for the input. What's left is the system. *The controller acting is not a fault. The system changing is.*"

**Delivery:** Slowest slide. One equation, one intuition, the spine line. The full three-window `Q_k` algebra lives in backup B2 — do not put it here.

---

## SLIDE 7 — The fix: online truncation + proof of life (6:35–7:55) · 80 s · **CONTRIBUTION**

**On screen:** **Real before/after plot** — online DMDc statistic exploding on small singular values vs. toDMDc staying bounded on the *same* data. One line of method: rank-r truncation, online, orthonormal basis.

**Say:**
- "Why did online DMDc become unreliable? Small singular values. When the data's energy sits in a few directions, inverting the rest makes the update blow up — you can see it here." (point to the exploding curve)
- "We truncate to rank r *online* — keep the directions carrying the dynamics, drop the noise floor — with an update that keeps the basis orthonormal as it goes."
- "Same data, bounded statistic." (point to the flat curve) "And the cost drops to **order m-r-squared** per step (say it slowly) — a handful of operations, independent of history length. Real-time, even in high dimension."

**Delivery:** This is the earliest hard evidence in the talk — a *real* curve, not a cartoon. Let them *see* the fix before you ask them to believe the guarantees.

---

## SLIDE 8 — Two guarantees you can stand behind (7:55–8:50) · 55 s · **CREDIBILITY**

**On screen:** Two headlines only.
- **Detection delay ≥ c** — the peak can't appear before the test window fills; empirically the delay is **c + a few samples** (measured **102 vs. c = 100**). Window size = your worst-case reaction budget.
- **Cost: a handful of operations per sample**, independent of history length (formula O(m r²) on screen — let the room read the exponent).

**Say:**
- "Two properties this room cares about more than any accuracy plot."
- "Detection delay is *bounded below* by the test window — the peak cannot appear before the window fills. Empirically it's the window plus a few samples: we measured 102 against a window of 100. So the window is your **design knob** for guaranteed reaction time."
- "And it's cheap — a handful of operations per sample, independent of how much history you've seen — so it runs inside the sample interval." (let the slide show O(m r²); don't fight the exponent out loud)

**Delivery (credibility):** Say **≥ c**, never "exactly c" — your own data shows 102, 198, 205, and "exactly" gets you impeached on the next slide. The stationary-convergence theorem and the probabilistic δ-bound are **deliberately cut** — δ and α are unquantified in the paper and would draw a question you can't answer. Two solid claims beat four soft ones.

---

## SLIDE 9 — Proof of the spine: control vs. fault on a controlled plant (8:50–10:05) · 75 s

**On screen:** Two-tank, **one event only** — the *doubled control response*. Top: states + the control input (a random step every 200 samples). Bottom: `Q_k` flat through every command, one clean peak at the doubled-response event. One arrow: "same input — different response."

**Say:**
- "A controlled nonlinear plant — two tanks, input delay. The controller steps the inflow every 200 samples." (point) "Watch the score stay flat through *every* command — it accounts for the input."
- "Then this:" (point to the event) "the *same* command, but the tank's response has doubled. The input didn't change — the system did. One clean peak."
- "This is the spine, proven: control action — flat. System change — flagged."

**Delivery:** This is where the central claim is *earned* on a plant with a real manipulated input. Keep it to the one event; bias and drift events live in backup B1 for Q&A.

---

## SLIDE 10 — Real-data win: catching it early (10:05–12:05) · 120 s · **HERO**

**On screen:** BESS figure. Temperature profiles (top); `Q_k` (middle) with the **threshold line drawn**, the confirmed-fault peak, and **one bold arrow on the precursor bumps**: "early warning — ~2000 samples before confirmation."

**Transition in (pivot register, say before advancing):** "That's the proof on a plant we *control*. Now a plant we *don't*."

**Say (lead with the jaw-dropper, hedge last):**
- "Real battery. Six temperature sensors, 30-second sampling. A cooling-system hardware fault on a known date. And here —" (point to precursor) "— **~2000 samples before the confirmed fault, the score already lifts.** Early onset, before anyone logged a fault."
- Setup: "The system follows a daily solar cycle, so the learning window is 24 hours. It *adapts through* the diurnal swing —" (point) "— and still flags the fault here, well above threshold." (point to main peak above the line)
- "Interpretable score, real fault, early warning — on data the model had never seen."
- **Calibration line, last (deliver calmly):** "We flagged the same slow onset in earlier work — suggestive of an unmodeled slow dynamic, not a confirmed mechanism yet."

**Delivery:** Your hero slide — the image they repeat at coffee. Lead with the precursor, not the main peak. The hedge *raises* credibility with this crowd. Note: here the input is the exogenous solar/ambient signal, not a manipulated control — if asked, say so plainly and point back to slide 9 for the manipulated-control proof.

---

## SLIDE 11 — Close: filled the box, feeds the controller (12:05–13:35) · 90 s

**On screen:** The matrix from slide 5 with the last box now ✅. Three bullets + the spine line one final time. Bottom strip: GitHub, `pip install`, QR.

**Say:**
- "Remember this table?" (name the callback) "Same matrix, one box changed: we took online DMDc — interpretable, adaptive, control-aware — and closed the one that made it unreliable: numerical stability, via online truncation."
- "What you get is a detector that stays current on its own, tells the controller's actions apart from a real fault, with a delay you can size and a cost you can afford. **That is exactly the trigger a fault-tolerant scheme consumes** — detection ready to hand off to tolerance."
- Logistics, before the closer: "Open source, one-line install, the BESS figures reproduce — try it on your data."
- **Spine line, verbatim, final time:** "*The controller acting is not a fault. The system changing is.*"
- "Thank you — happy to take questions."

**Delivery:** One ending. Name the matrix callback so the payoff registers. Logistics goes *before* the spine line so your final spoken words are the spine + thanks — one clean mic-drop. Get to "questions" with ~5 min left; a chair never minds finishing under.

---

## BACKUP SLIDES (Q&A only — do not present)

- **B1 — Two-tank, full run.** All three events: sensor bias (delay 205), doubled response (198), linear drift (195) vs. c = 200. Note honestly: the SVD-only baseline detects the doubled response slightly better; toDMDc wins on the slow drift and on overall noise.
- **B2 — The Q_k math.** Three windows (base a, test c, learning d), reconstruction-error ratio, null vs. alternative, the max(0, ·) form.
- **B3 — Online update equations.** Recursive P/A updates, negative-weight forgetting, online-SVD rotation alignment K^{U'U}.
- **B4 — Unknown B.** Augmented-state [A|B] identification — the default mode; control effect is *estimated*, so explained "to first order."
- **B5 — Parameters + threshold.** a = c = expected change duration; d = operating cycle; h ≥ 2τ_max; rank via Gavish–Donoho / 95–99% energy. Fixed τ in experiments; adaptive percentile in practice (and its bias caveat).
- **B6 — Synthetic delay benchmark.** 9 steps; **8 of 9 detected, first buried in noise by design** (Q_k is a *relative* statistic); delay 102 ± 3 vs. theory 100; SVD-only baseline 25% higher variance.

---

## ANTICIPATED Q&A (rehearse — ordered by likelihood of being asked)

1. **THE dangerous one — adaptation vs. detection:** *"Your learning window adapts the model online. A slow drift or incipient fault IS a slow change — won't the window just absorb the fault into the 'new normal' and Q_k returns to zero? How do you detect a fault you're simultaneously learning?"*
   → "Detection precedes learning every step, so we get at least the test window c before any absorption. Beyond that, faults are distinguished by *rate*, not magnitude: multi-resolution windows separate fault-rate from learning-rate. And I'll concede the honest limit — a change slower than the learning window d is *by design* treated as adaptation, because that's exactly what aging is. Faults faster than d, we catch; drift slower than d, we track. Choosing d sets that boundary deliberately."

2. **vs. observer/EKF residual (the .1/.3 crowd):** → "An observer needs a model *you* built and validated. We *identify* the model online from data and keep it current — no hand-built model, and it adapts to aging. That's the differentiator: input-aware residuals are what an observer already does; an observer that re-identifies itself online is not."

3. **Threshold without labels — isn't 95th-percentile circular?** → "Fixed τ in the experiments. In practice an adaptive percentile of Q_k during confirmed-normal operation; we note in the paper it can introduce systematic bias, so it's a deployment knob, not a guarantee. Honest open problem."

4. **Why DMD over a neural detector (TIRE)?** → "Interpretable score plus DMD modes for root cause; unsupervised, no labels; cheaper. We trade a little raw accuracy for trust and deployability — which is what gets a method past a plant operator."

5. **Stationarity in the convergence claim — realistic?** → "It's piecewise: locally stationary between change-points, where the estimate stays within O(κ⁻¹) of the batch solution. The forgetting window tracks the slow non-stationarity. I'm deliberately not claiming global convergence on non-stationary data."

6. **Unknown control matrix B?** → "Augmented-state formulation identifies [A|B] jointly — that's the default. Backup B4."

7. **Where's the manipulated control in the BESS?** → "Fair — in the BESS the exogenous input is the solar/ambient cycle, not a commanded control. The *manipulated*-control proof is the two-tank, slide 9, where the same command produces a doubled response. BESS is there for real-fault, real-data impact."

8. **Cost on the BESS?** → "O(m r²), r small (2–10) — orders of magnitude faster than the 30-second sample interval."
