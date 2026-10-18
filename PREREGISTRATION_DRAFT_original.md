# Paper 6, Stage 1 preregistration — DRAFT, NOT FROZEN: sanity checks for linear directions

> **Status: DRAFT. Not frozen. No hash recorded. Nothing in this file binds any analysis yet.**
> It was written on 2026-10-14 by the AI assistant, in a session started from the paper-4 session, after a literature check (`LITERATURE.md`) and before any code for this study was written or run. The author has not reviewed it. It becomes the Stage 1 preregistration only when the author has (1) reviewed and edited it, (2) confirmed the list of directions in §3, (3) removed this banner and every `[OPEN]` marker, and (4) recorded its SHA-256 in `PREREG_SHA256SUMS`, as in `cr4/`. Stage 2 (`PREREGISTRATION_STAGE2.md`) will fix strengths, norm grids, item counts and generation lengths after a pilot on separate items, and may not change anything in Stage 1.
>
> Items marked **[OPEN]** are decisions the author should make before freezing.

## 1. Motivation

A common kind of claim in interpretability says that a linear direction `d` in a model's residual stream *tracks* a process (its projection rises and falls with the process) or *drives* it (adding `d` changes the process). The usual evidence is (i) `d` separates two sets of prompts that differ in the process by design, and (ii) adding `d` at some norm changes the output in the expected way.

Papers 1–4 of this series show how that evidence can mislead:
- The compute–retrieve (CR) axis (Ramdan 2026a, doi:10.5281/zenodo.19455724) separates computation prompts from retrieval prompts with AUC ≥ 0.988 in 12 models.
- Under exact format and length matching it reversed in three of five base models and was mostly driven by context vocabulary (Ramdan 2026b, doi:10.5281/zenodo.23318679, Experiments 5 and 5b).
- In paper 3 (Qwen3 1.7B–14B), reports of injected thoughts ran through one content-free direction that acted as a general yes-bias, not through a representation of the injected content.
- In paper 4 (preregistered, `cr4/`; Qwen3-1.7B, Qwen2.5-7B-Instruct, Qwen3-8B, format-matched multiplication), the CR projection did not track logit-lens settling depth (Spearman ρ between −0.03 and +0.07, every 95% CI including 0). Matched-KL pushes moved settling depth by less than one layer.
  - Its main effect was to toggle reply format (show working vs answer directly; on novel two-digit items in Qwen2.5-7B, −0.51 [−0.54, −0.48] in the rate of shown working beyond random directions).
  - A label-free direction built from the same 60 calibration prompts — within-class long-minus-short — did the same (−0.53 [−0.55, −0.50]).
  - That length direction tracked settling depth better than the CR axis (ρ = +0.30, +0.32, +0.13), and separated familiar from novel problems far better (AUC ≈ 0.85 vs ≈ 0.63).

None of these failures would have been visible from calibration AUC or a fixed-norm steering demonstration. The checks that exposed them are generic and cheap:
- (a) measure the process per item from what the model does;
- (b) match stimuli on format, length and vocabulary;
- (c) compare with a label-free nuisance direction from the same calibration prompts;
- (d) compare with random pushes matched on output change (KL), not norm;
- (e) localise the stage at which the push's effect is decided.

This study packages them as a battery, in the spirit of Adebayo et al.'s (2018) sanity checks for saliency maps. It validates the battery on directions whose answer is known, and applies it to published directions on open models of at most 14B parameters.

Related work and the novelty assessment are in `LITERATURE.md`. In short, every check has precedents taken one at a time:
- process-defined labels in knowledge-awareness and reasoning-behaviour work;
- confound and control datasets in Marks & Tegmark and in Hewitt & Liang's control tasks;
- random-direction baselines in many steering papers;
- layer sweeps everywhere.

Three of the five checks already have close 2026 precedents:
- KL-matched random controls: SteerCheck (Luo et al. 2026) and Ferrara (2026), who matched on Jensen–Shannon divergence;
- format deconfounding of a reasoning probe: Sahoo et al. (2026);
- a reusable audit across several published claims: Wu et al. (2026).

What this study adds:
1. Process-defined per-item targets as a standard check, applied to every direction.
2. A structured, label-free nuisance direction from the same calibration prompts as a head-to-head competitor.
3. Oracle directions and known-answer cases that test the battery itself.
4. The whole set applied uniformly to about ten published directions.

KL matching is used here as a standard, not claimed as new.

## 2. Definitions

- **Direction claim.** A tuple (model `m`, layer `ℓ*`, reading position, vector `d`, process `X`, kind ∈ {tracks, drives, both}), taken from a paper and its released code.
- **Calibration set.** The labelled prompts (or prompt pairs) the authors used to build `d`, used unchanged.
- **Process target `T`.** A per-item number or binary outcome measured from the model's own behaviour or internals on that item: settling layer, actual refusal, actual number of thinking tokens, whether the model's own true/false verdict is "true", and so on. It is never the label of the prompt's class. Each direction's `T` is fixed in §3 before any data are seen.
- **Projection `P`.** `h_ℓ*(x) · d̂` at the authors' reading position, where `d̂ = d/‖d‖`.
- **Gap `g`.** Mean projection of the positive calibration class minus the mean projection of the negative class (paper-4 definition). It is the natural unit of push size.
- **Push.** Adding `α · d̂` to the output of block `ℓ*` at every position from the reading position to the end, including generated tokens (paper-4 protocol).
  - Where the authors' own intervention differs (for example directional ablation for refusal), it is run as well, in Check 0 only.
- **Size of a push.** `KL(clean ‖ pushed)` summed over every position from the reading position to the end of the clean continuation, teacher-forced on the clean model's own continuation. This is paper-4 Stage 2's `kl_all`.
  - The first-position KL is reported for sensitivity only, because paper-4 pilot 1 showed it is degenerate when the first output token is near-certain.

## 3. Directions under test

[OPEN] The final list. The table below is the proposal from `LITERATURE.md` §6. Stage 2 may *drop* a direction only if Check 0 (reproduction) fails, and must report the failure. It may not add directions, except as exploratory.

**How directions are built.** In each panel model, the direction is rebuilt with the authors' code, calibration data, layer rule and reading position.
- In the home model, Check 0a also compares the rebuilt vector with the released vector, where one exists (cosine), or with the paper's headline number.
- "Panel" lists the three models whose verdicts form the overall verdict.
- In the panel abbreviations, R1 means a DeepSeek-R1-Distill model and "think" means Qwen3 with `enable_thinking=True`.

| ID | Direction (source, code) | Role | Claim | Panel (home first) |
|---|---|---|---|---|
| D1 | Refusal (Arditi et al. 2024; github.com/andyrdt/refusal_direction) | positive control | tracks + drives | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B |
| D2 | Truth, mass-mean (Marks & Tegmark 2024, github.com/saprmarks/geometry-of-truth; Bürger et al. 2024's data as a second calibration set) | static-concept control | tracks + drives | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B |
| D3 | Compute–retrieve axis (Ramdan 2026a; github.com/Somebodyhere101/compute-retrieve) | known-answer case (H7) | tracks + drives | Qwen2.5-7B-Instruct, Qwen3-8B, Llama-3.1-8B-Instruct |
| D4 | Reasoning–memorisation features, LiReFs (Hong et al. 2025; github.com/yihuaihong/Linear_Reasoning_Features) | process | tracks + drives | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-8B |
| D5 | Backtracking (Ward et al. 2025, released `base_steering_vectors.pt`, github.com/jnward/latent-backtracking; Venhoff et al. 2025 recipe, github.com/cvenhoff/steering-thinking-llms) | process | tracks + drives | R1-Llama-8B, R1-Qwen-7B, Qwen3-8B think |
| D6 | Reasoning strength (Sheng et al. 2025; github.com/AlphaLab-USTC/LRM-plans-CoT) | process | tracks + drives | R1-Qwen-7B, R1-Llama-8B, Qwen3-8B think |
| D7 | Known-entity awareness, difference-of-means version (Ferrando et al. 2025; github.com/javiferran/sae_entities) | process | tracks + drives | Gemma-2-9B-it, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct |
| D8 | Persona vector: sycophancy (Chen et al. 2025; github.com/safety-research/persona_vectors) | process | tracks + drives | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Qwen3-8B |
| D9 | True-thinking direction (Zhao et al. 2025; github.com/andotalao24/Identify_true_decorative_thinking) | process | tracks + drives | R1-Qwen-7B, R1-Llama-8B, Qwen3-8B think |
| D10 | Test awareness (Abdelnabi & Salem 2025; github.com/microsoft/Test_Awareness_Steering), ported from 32B | process, secondary (cost) | tracks + drives | Qwen3-8B think, Qwen3-14B think, R1-Qwen-14B |

[OPEN] Alternates, used only if a listed direction fails Check 0a and only in this order:
1. sentiment (Tigges et al. 2024) for D2;
2. ThinkEdit (Sun et al. 2025) for D6;
3. in-advance correctness (Cencerrado et al. 2025) for D7;
4. CAA sycophancy (Rimsky et al. 2024) for D8.

**Process targets, matched stimuli and SESOIs (fixed in advance).** Unless stated otherwise, the tracking SESOI is ρ = 0.20 or AUC = 0.60.
- **D1 Refusal.**
  - *T:* the completion actually refuses (binary). The judge is WildGuard's response-refusal output, with refusal-phrase matching as a secondary measure. [OPEN] Confirm the judge.
  - *Matched stimuli:* XSTest's 250 safe prompts that look unsafe and its 200 unsafe contrasts (same templates and vocabulary), plus OR-Bench-Hard items. Refusal and the harmfulness label come apart on these.
  - *Push:* +d on harmless prompts and −d on harmful ones.
  - *Drive SESOI:* 10 points of refusal rate.
- **D2 Truth.**
  - *T:* the model's own verdict, P("True") − P("False") when the statement is posed as a true/false question in a fixed template. This is not the dataset label.
  - *Matched stimuli:* Marks & Tegmark's curated sets (cities, neg_cities, larger_than, sp_en_trans). Their "likely" set serves as a text-probability control.
  - *Drive SESOI:* 10 points of the rate of "True" verdicts.
- **D3 CR axis.** Paper 4's items, targets (settling depth D; shown working W) and SESOIs (1 layer for D, 10 points for W), unchanged.
- **D4 LiReFs.**
  - *T:* settling depth on paper 4's format-matched multiplication items, plus *CoT necessity* on matched GSM8K-style items: accuracy with chain of thought minus accuracy with a direct answer, per item, over 4 samples each.
  - *Drive SESOI:* 1 layer of settling depth, and 10 points of accuracy.
  - D4 and D3 share items and targets, so the two published "reasoning vs recall" directions are compared head to head.
- **D5 Backtracking.**
  - *T (tracking):* whether the next sentence of the model's own chain of thought is a backtrack, judged by a local judge model ([OPEN] which). The projection is read at the authors' offset before the sentence. Covariates are the sentence's position in the chain and the previous sentence's length.
  - *T (drive):* backtracking sentences per 1,000 thinking tokens.
  - *Matched stimuli:* the same problems and the same chain prefixes, compared across sentence positions matched on index.
  - *Drive SESOI:* a 25% relative change.
  - Check E's direct-path control (E2) is the decisive check here. A vector that only raises the logit of "Wait" would fail it.
- **D6 Reasoning strength.**
  - *T:* number of thinking tokens, the mean over 4 samples.
  - *Tracking:* partial ρ controlling for MATH difficulty level, question token length and subject. The plain ρ is secondary, because the published claim is about planning beyond difficulty.
  - *Drive SESOI:* a 10% relative change.
  - `N_len` (question length within difficulty level) is the natural competitor.
- **D7 Known-entity awareness.**
  - *T:* the attribute is actually recalled. Correctness on held-out attribute questions is the primary measure; the logit-lens rank of the correct attribute at the last entity token is secondary.
  - *Matched stimuli:* entities matched on token count and entity type, with popularity as a covariate. Ferrando et al.'s discarded middle items are included.
  - *Drive SESOI:* 10 points of "I don't know" or refusal on known entities, and of hallucination on unknown ones.
- **D8 Sycophancy.**
  - *T:* the answer actually flips to the user's stated wrong belief, on known-answer items scored by exact match, with no judge. The projection is read at the final prompt token.
  - *Matched stimuli:* the same question with and without the user's belief statement; the statements are matched in length and in the wrong answer's frequency.
  - *Drive SESOI:* 10 points of flip rate.
- **D9 True thinking.**
  - *T:* each step's True Thinking Score, the authors' causal measure of the step's effect on the final answer. Their labels are already process-defined, so here the battery tests (b)–(e) and the nuisance comparison, not the choice of target.
  - *Drive SESOI:* [OPEN] proposed is 10 points in the rate at which the final answer follows a perturbed step.
- **D10 Test awareness.**
  - *T:* the later chain of thought verbalises being tested (judge), plus the compliance gap between real and hypothetical versions of a task.
  - *Matched stimuli:* the authors' trigger pairs, re-matched on length and on cue tokens such as ".mock" and placeholder names.
  - *Drive SESOI:* 10 points.
  - This direction is secondary: it enters H5 and H6 only if it is run in all three panel models.

### Control directions built for every (direction, model) pair
- **Nuisance directions (Check C), from the direction's own calibration set, label-free:**
  - `N_len`, the within-class length direction. In each label class, split the calibration prompts at the median token length of the formatted prompt. `N_len` = unit vector along the mean, over classes, of (mean activation of the long half − mean activation of the short half) at `ℓ*` and the reading position. For pair-based calibration sets (two completions of one prompt), the split uses the length of the shared prompt, and both members of a pair go to the same half.
  - `N_pc1`, the first principal component of the calibration activations after centring each class on its own mean. This is the largest within-class source of variance, found without labels and without choosing a nuisance by hand. Its sign is set so that it has a positive projection on `d`.
  - `N_perm`, 20 label-permuted directions. Each is the difference of means after a random relabelling of the calibration set with class sizes kept; seeds are fixed. This is the analogue of Hewitt & Liang's control task and Adebayo et al.'s data randomisation. It is also used as an in-distribution random direction in Check D.
    - SteerCheck (Luo et al. 2026) found that same-construction sign-randomised nulls often keep substantial alignment with the real direction ("alignment leakage"). So cos(N_perm, d) is reported, and every comparison against `N_perm` is repeated on the permutations with |cos| < 0.2 as a sensitivity analysis.
- **Oracle direction `o` (positive control for the target).** Difference of means at `ℓ*` between the top and bottom terciles of `T`, computed on an *oracle split* of the matched stimuli that is disjoint from every test item. It answers one question: can this target be read and pushed linearly at `ℓ*` at all? If the oracle fails a check, a failure of `d` on that check is *uninformative*, not a negative result (see §6).
- **Random directions (Check D).** `R` Gaussian unit vectors per model (fixed seed), at norms `c · g`, with the grid fixed in Stage 2 so that it brackets the KL of every tested push. The starting grid is paper 4's `c ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4}` with `R = 12`.

## 4. Models

[OPEN] Confirm the panel. Proposal:
- **Home model.** Each direction is tested first in the model it was published on, if that model is open and has at most 14B parameters. Otherwise it is tested in the nearest member of the same family.
- **Panels.** Each direction has three panel models, listed in the §3 table.
  - **Chat panel**, for directions on chat prompts: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Qwen3-8B with `enable_thinking=False`. D7 uses Gemma-2-9B-it as its home model.
  - **Reasoning panel**, for directions defined on thinking traces: DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, and Qwen3-8B with `enable_thinking=True`. The 14B reasoning models are secondary, because long generations with CPU-streamed blocks are slow.
  - Every model runs in bf16 with its tokenizer's chat template.
- **Secondary, if time allows:** Qwen3-1.7B (on the laptop, used for the pilot), Qwen3-14B, DeepSeek-R1-Distill-Qwen-14B.
- Revisions (commit hashes) are pinned in Stage 2.
- Hardware:
  - the 8 GB laptop GPU for ≤ 2B;
  - the RTX 5090 for the rest;
  - for 14B models, blocks that do not fit stay in CPU memory and are streamed per forward pass, using paper 3's loader (same weights, same arithmetic).
- Batch size is held fixed within every comparison, because bf16 results depend slightly on batch size (paper 3).

## 5. The battery

Each check has a statistic, a pass criterion and a smallest effect size of interest (SESOI). A check is **passed** only if the 95% CI excludes 0 in the claimed sign *and* the point estimate reaches the SESOI. It is **failed** if the CI bound on the claimed side falls short of the SESOI, so the effect is shown to be smaller than the SESOI. For a positive claim with SESOI ρ = 0.20, for example, that means an upper bound below 0.20. Otherwise it is **inconclusive**.

All CIs come from an item bootstrap with 2,000 resamples. Check D uses a two-level bootstrap (items, and random directions within the random reference) as its primary interval. The item-only bootstrap, which is paper 4's, is reported alongside.

### Check 0 — Reproduction (gate) and randomised-model calibration (descriptive)
- **0a. Reproduction.** Run the authors' code at their layer, position and calibration set. Report their headline metric on their own evaluation (for example: refusal rate after directional ablation; held-out probe accuracy).
  - **Gate:** within the tolerance fixed in Stage 2, from the authors' reported number and the pilot.
  - A direction that fails the gate is reported as *not reproduced* and leaves the confirmatory analysis.
- **0b. Randomised-model calibration (the Adebayo et al. analogue).** Apply the same recipe to a copy of the model whose weights are re-initialised with the architecture's own initialiser (fixed seed). Report the calibration AUC of the resulting direction on held-out calibration-style prompts. This is *descriptive*: it shows how much of the published separation a network that has learned nothing would also give. It enters only H3 below.

### Check A — Tracking a process-defined target (check a)
- **Items.** The direction's *process test set*: the matched stimuli listed for it in §3, in one template and answer format, so that `T` varies within format. It is disjoint from calibration, pilot, oracle and Check B items. The correlation of `T` with prompt token length is reported. If |ρ| > 0.1, the partial statistic below becomes the primary one for that direction.
- **Statistic.** Spearman ρ(P, T) for continuous `T`; AUC of P for binary `T`.
- **Pass A.** The CI excludes 0 in the claimed sign, and ρ ≥ 0.20 (AUC ≥ 0.60).
- **Secondary statistics:**
  - partial ρ controlling for prompt token length and any item covariates listed for that direction in §3;
  - the same statistics for `o`, the ceiling, and for `N_len` and `N_pc1`, which enter Check C.
- Applies to every direction that makes a tracking claim. For drive-only claims it is reported but does not enter the verdict.

### Check B — Label separation survives matched stimuli (check b)
- **Matched label set.** Prompts from the direction's two classes, built to share:
  - one template and answer format;
  - token length (exactly matched pairs, or matched within ±1 token);
  - vocabulary (minimal pairs where possible).
- The matched set is used only to *evaluate* the published direction, never to rebuild it. Petrov (2026) found that refusal directions *built* from topic-matched contrasts are too small to steer, because subtraction cancels the shared topic component. That concerns construction, not evaluation.
- **Leakage diagnostics, both reported:**
  - a text-only baseline: logistic regression on bag-of-tokens, 5-fold, giving a cross-validated AUC;
  - the AUC of token length alone.

  The matched set is accepted only if the text-only AUC is ≤ 0.65. Stage 2 records the construction for each direction.
- **Statistic.** AUC of `P` for the class label on the matched set.
- **Pass B.** AUC ≥ 0.70 in the claimed direction, with CI lower bound > 0.5. Because the matched set is only accepted if text-only AUC ≤ 0.65, a pass also means the direction separates the classes better than the surface text does.
- **Reversal.** A CI entirely below 0.5 is reported as a *reversal*, as in Ramdan 2026b.

### Check C — Label-free nuisance baseline (check c)
- **Tracking:** paired bootstrap difference ρ_d − ρ_N on the Check A items, for N ∈ {N_len, N_pc1}.
- **Steering:** effect_d − effect_N at matched KL on the Check D items. Both are pushed at the same KL, using the interpolation procedure of Check D.
- **Pass C.** For each claimed kind (tracks and/or drives), `d` beats *both* nuisance directions, with the paired CI of the difference excluding 0 in `d`'s favour.
- **Flag NUISANCE.** A nuisance direction *on its own* meets the pass criterion of the same check (A for tracking, D for driving), and `d` does not beat it: the paired CI does not exclude 0 in `d`'s favour. If neither `d` nor N shows the effect, no flag is raised; the verdict is then NOT SHOWN, not NUISANCE.
- **C inconclusive.** Anything else, for example `d` passes A, no N passes A, but the paired CI includes 0.
- **Also reported:**
  - cos(d, N_len) and cos(d, N_pc1);
  - Check A re-run on `d⊥`, which is `d` with `N_len` and `N_pc1` projected out (Gram–Schmidt);
  - the fraction of ρ that `d⊥` retains.

### Check D — Steering beyond random pushes of matched output change (check d)
- **Pushes.**
  - Primary: `±1 · g` along `d̂`.
  - Both signs are run, on the items where each sign can move `T` (for example, +d on harmless and −d on harmful prompts for refusal).
  - Check D is evaluated per sign. It passes if at least one sign passes, and the other sign is not a significant reversal.
  - Secondary: the authors' published strength, if it differs.
  - Exploratory: `±0.5 g` and `±2 g`.
- **Matched-KL reference.** This is paper 4's procedure.
  - For each item and random direction, the change in `T` (pushed − clean) is interpolated linearly in log KL between the two random norms that bracket the push's KL.
  - The random reference is the mean over directions.
  - Pushes outside the bracket are excluded and counted.
  - **Effect** = mean over items of (ΔT_d − random reference).
- **Second reference.** The same effect against `N_perm` directions, rescaled to unit norm and pushed over the same norm grid, then interpolated to matched KL (in-distribution random). This reference matters because equal-norm Gaussian directions can change the output far less than directions built from data. In ITI's Table 3, for example, a random direction gave KL 0.02 against 0.27 for the mass-mean shift at the same α (see `LITERATURE.md` §2.14).
- **Pass D.** The effect against both references has its CI excluding 0 in the claimed direction and its point estimate ≥ the direction's SESOI_T. The SESOI is fixed per target in §3: for example, 1 layer for settling depth, 10 percentage points for a rate, 10% for a token count.
- **Norm-matched effect (reported for H4).** The same effect computed against random directions of the *same norm* as the push, without KL matching.
- **Format clamp (reported, and enters the FORMAT-MEDIATED flag).** Where the target can be measured with the output format fixed by a prefill (as in paper 4's direct-answer runs), Check D is run both free and clamped.
  - **Flag FORMAT-MEDIATED.** Pass D free, but an effect below SESOI_T under the clamp, with the CI excluding SESOI_T.
  - Also reported is the push's effect on a generic format measure: reply length in tokens, and whether working is shown. This is reported against random and against `N_len`.

### Check E — Stage localisation (check e)
Run only for pushes that pass D.
- **E1. Re-imposition sweep.**
  - For every later block `ℓ' ∈ {ℓ*+1, …, L−1}` (every second block if `L > 32`), apply the primary push at `ℓ*`.
  - At `ℓ'`, set the component of the hidden state along `d̂` back to its value in the clean run, at every pushed position.
  - Measure the fraction of the full effect on `T` that remains.
  - The **hand-off layer** `ℓ_h` is the smallest `ℓ'` at which the remaining fraction is ≥ 0.5: by then the push's effect is carried by components other than `d̂` itself.
- **E2. Direct-path control.**
  - Add `β · d̂` only at the output of the last block (just before the final norm), with `β` chosen so that the KL equals the primary push's.
  - Report the effect on `T` as a fraction of the primary effect.
  - This is close to Yuan & Vlachos's (2026) "direct logit bias equivalence" control. It is used here as a fixed check for every direction.
- **Pass E (process-mediated).** `ℓ_h ≤ ℓ* + 0.75 · (L − 1 − ℓ*)`, so the hand-off happens before the last quarter of the remaining depth, *and* E2 recovers < 50% of the primary effect (point estimate).
- **Flag OUTPUT-BIAS.** No hand-off before the last block, or E2 recovers ≥ 50%. In that case the push acts mainly like a bias on output tokens, as paper 3's yes-bias direction did, rather than through downstream computation.
- **E3 (descriptive).** Logit-lens settling layer of the target-relevant output token, clean and pushed.

## 6. Verdicts

The verdict concerns the *process* claim. Check B concerns the *label* claim, and is reported as a separate label verdict. A direction can track a process without separating the prompt classes it was built from. The refusal direction, for example, should follow actual refusal, including over-refusal of benign prompts that look harmful, rather than the harmfulness label. Zhao et al. (2025) report separate harmfulness and refusal directions. So B is not required for a valid process verdict.

The process verdict is given per (direction, model), in this order:
1. **NOT REPRODUCED.** Check 0a fails.
2. **TARGET NOT LINEAR.** The oracle direction `o` fails A, or (for drive claims) fails D. The direction's own A or D result is then reported but *uninformative*.
3. **NUISANCE.** The NUISANCE flag is raised in Check C for any claimed kind.
4. **FORMAT-MEDIATED** or **OUTPUT-BIAS.** It passes D, but the format clamp or Check E raises the flag.
5. **VALID TRACKER.** It passes A and C (tracking); D is not passed or not claimed.
6. **VALID DRIVER.** It passes every check its claim requires: A and C-tracking if tracking is claimed; D, C-steering and E for the drive claim.
7. **NOT SPECIFIC.** It passes A (or D), but C is inconclusive, so it is not shown to beat the nuisance directions.
8. **NOT SHOWN.** A (or D) is failed or inconclusive for the claimed kind, and no flag above applies.

The label verdict is reported alongside:
- **LABEL SURVIVES MATCHING:** passes B.
- **REVERSED:** B's CI lies entirely below 0.5.
- **LABEL DOES NOT SURVIVE:** anything else.

The combination "label survives matching" with a process verdict of NOT SHOWN is reported as **LABEL-ONLY**. In that case the direction separates the prompt classes even when they are matched, but does not track the process.

Across models, a direction's overall verdict is the verdict it receives in at least two of its three panel models. Otherwise it is MIXED. Verdicts are per model, without correction across models or directions, as in papers 4 and 5. The CIs are reported so that readers can apply their own correction.

## 7. Hypotheses (directions fixed in advance)

Known-answer cases test the battery itself and are not counted in H5–H6.

- **H1, the battery passes a direction known to work (positive control).** The refusal direction (Arditi et al. 2024) receives VALID DRIVER in at least two of its three panel models.
  - Its process target is actual refusal of each prompt, judged as in §3, on matched harmful and harmless prompts that include benign prompts that look harmful (XSTest-style).
  - **If H1 fails,** the check or checks that refusal fails are declared *miscalibrated as specified*. They are reported descriptively for every direction, and the verdicts of §6 are recomputed without them. This consequence is fixed now.
- **H2, the battery rejects directions known to carry no process signal (negative control).** The `N_perm` directions, evaluated as if they were claims about each target, fail A and D (effect below SESOI) in every model. This is an implementation check: a failure means a bug, not a finding.
- **H3, calibration separation is weak evidence (Adebayo analogue).** For at least half of the directions in §3, the direction built by the same recipe in a weight-randomised copy of the model separates held-out calibration-style prompts with AUC ≥ 0.80.
- **H4, norm matching overstates specificity.** Across all (direction, model) pairs that pass Check 0, the steering effect computed against norm-matched random directions exceeds the effect against KL-matched random directions.
  - Test: paired difference on effects standardised by SESOI_T, mean over pairs, with a bootstrap over pairs.
  - Supported if the CI excludes 0 above.
- **H5, most published process directions do not pass the battery.** Among the process-claim directions in §3, excluding the refusal and truth controls and the CR known-answer case, fewer than half receive an overall VALID TRACKER or VALID DRIVER verdict.
- **H6, label-free nuisance directions are serious competitors.** At least two of the process-claim directions receive NUISANCE in at least one of their panel models.
- **H7, known-answer retrodiction (CR axis).** The battery reproduces paper 4's conclusions for the CR axis in Qwen2.5-7B-Instruct and Qwen3-8B on paper 4's items:
  - A fails for settling depth;
  - D on settling depth stays below the 1-layer SESOI;
  - the process verdict is NUISANCE or NOT SHOWN, never VALID;
  - on reply format (shown working), the axis's matched-KL effect does not beat `N_len`.

  Paper 4's values imply NUISANCE in Qwen2.5-7B-Instruct, where `N_len` alone tracks settling depth with ρ = +0.32 [+0.20, +0.43], and NOT SHOWN in Qwen3-8B, where `N_len` gives ρ = +0.13.

  This is a retrodiction, *not* a confirmatory test, because those results are already known. Its role is to show that the battery code reproduces a known answer.

A significant effect opposite to a prediction is reported as a reversal, not as a null. Exploratory analyses do not enter the decision.

## 8. Sample sizes

[OPEN] These are proposals; Stage 2 fixes them after the pilot's run-time estimates.
- **Calibration:** the authors' set, unchanged.
- **Check A test set:** 240 items per direction where the source allows it (paper 4's M set). This gives a 95% CI half-width of about 0.12 for ρ near 0. For binary targets, at least 100 items in the smaller outcome class, or as many as the source provides, with the count reported.
- **Check B matched label set:** 120 matched pairs (240 prompts).
- **Oracle split:** 120 items, disjoint from everything else.
- **Check D:** 120 items (a fixed random subset of the Check A test set). It covers:
  - `R = 12` random directions × the Stage-2 norm grid;
  - 8 of the 20 `N_perm` directions;
  - pushes of `d`, `N_len`, `N_pc1` and `o` at the primary and secondary strengths, in both signs.
- **Long-generation targets** (thinking-token counts, backtracking events): Stage 2 may reduce this to `R = 8`, 5 norms and 80 items if the pilot's run-time estimate for the full design exceeds 24 GPU-hours per (direction, model). The reduction must be recorded.
- **Check E:** 60 items from the Check D set, with the sweep over every (or every second) later block.
- **Power note.** With 120 items, paper 4's matched-KL effects had CI half-widths of roughly 0.002–0.006 (fraction of depth) for settling depth and 0.03–0.06 for format rates. These are far below the proposed SESOIs, so the main risk is effect size, not noise.

## 9. Exploratory (labelled as such)
- Cascading randomisation, layer by layer from the top (Adebayo et al.'s full protocol), for Check 0b.
- Other layers than `ℓ*`, and the best layer chosen *by the process target* on the oracle split. This tests whether the published layer is the right one.
- Non-linear probes for `T` (an MLP on the oracle split) as a ceiling above `o`.
- Jacobian lens instead of logit lens for settling-based targets in Qwen3 (neuronpedia/jacobian-lens, as in paper 4).
- Directions the pilot reveals as worth adding.

## 10. Pilot (before Stage 2)
- **Model and items.** Qwen3-1.7B, on pilot items only. These are disjoint from the calibration, oracle, Check A, B and D items of every direction.
- **Directions.** The refusal direction and one process direction ([OPEN]: proposed is the backtracking or the reasoning-length direction, whichever reproduces first).
- **What it is for:**
  - validating code and parsing;
  - checking that the random norm grid brackets every push's KL;
  - fixing the judges (refusal, backtracking) and their agreement with a hand-labelled sample of 50 outputs per judge, labelled by the author;
  - estimating run time;
  - fixing per-direction tolerances for Check 0a.
- **What it may and may not change.** It may set norm grids, `R`, item counts, generation lengths and the matched-set constructions. It may not change the checks, pass criteria, SESOIs, verdict rules, hypotheses or the list of directions (except dropping a direction that fails Check 0a).
- **Recording.** Any change, and anything the pilot shows about the direction of effects, is recorded in Stage 2 before the main runs.

## 11. Deviations
Any departure from Stage 1 or Stage 2 is listed, with its reason, in the results and on the results page, as in paper 4.
