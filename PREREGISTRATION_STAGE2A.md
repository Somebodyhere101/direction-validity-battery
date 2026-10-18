# Paper 6, Stage 2a: settings for D1 (refusal), D2 (truth) and D3 (CR axis) after the pilot

> Frozen 2026-10-14 23:05 AST, after the pilot (Qwen3-1.7B, pilot items only) and before any main-panel run. Written by the AI assistant, which makes these decisions on the author's standing instruction (Stage 1 §0). Stage 2a may only set what Stage 1 §10 allows: norm grids, `R`, item counts, generation lengths, matched-set constructions, judge choice and Check 0a tolerances. It changes no check, pass criterion, SESOI, verdict rule, hypothesis or direction. SHA-256 of this file, the item files, the configuration and the code are in `PREREG_SHA256SUMS`.

## 0. Why Stage 2 is split
Stage 1 §10 names refusal and backtracking as the pilot directions. On 2026-10-14 the author asked for the study to run overnight. Directions D1–D3 need only short generations and were built first; D4–D10 need more engineering (reasoning traces, judges, released vectors).

So Stage 2 is frozen in parts, each before its own directions' main runs:
- **Stage 2a (this file)** covers D1–D3 on the chat panel (Qwen2.5-7B-Instruct, Qwen3-8B with thinking off, OLMo-2-1124-7B-Instruct).
- **Stage 2b** will cover D4–D10. It will be frozen after the backtracking pilot, which has not been run.

No main-panel data for any direction existed when this file was frozen. **Deviation 1:** the backtracking pilot comes after, not before, the main D1–D3 runs. Nothing in D1–D3 depends on it.

## 1. Models (revisions pinned)
| key | model | revision |
|---|---|---|
| q3-1.7b (pilot) | Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |
| q25-7b | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| q3-8b | Qwen/Qwen3-8B (`enable_thinking=False`) | b968826d9c46dd6066d109eabc6255188de91218 |
| olmo-7b | allenai/OLMo-2-1124-7B-Instruct | 470b1fba1ae01581f270116362ee4aa1b97f4c84 |
| judge | Qwen/Qwen3-8B (thinking off) | b968826d9c46dd6066d109eabc6255188de91218 |

Setup:
- Hardware: RTX 5090 (all models fit; no CPU streaming).
- Software: bf16; torch 2.14.1+cu130; transformers 5.19.0.
- Prompts use each tokenizer's own chat template, including Qwen2.5's default system prompt.
- **Deviation 2:** the pilot ran on the RTX 5090 rather than the laptop, because the judge (8B) does not fit the laptop GPU. One D1 calibration test also ran on the laptop and is described in §7.

## 2. How each direction is rebuilt (the authors' recipe, made portable)
- **D1 refusal (Arditi et al. 2024).** Their pipeline, reimplemented in `specs.py` because their repository has no adapters for these three models. Every rule is theirs:
  - **Data:** `random.seed(42)`, then 128/128 train and 32/32 validation instructions are sampled from their splits, in their order.
  - **Filtering:** by refusal score (log-odds of the refusal tokens at the last position). The refusal tokens are the first tokens of "I" and "As", which are theirs for Qwen.
  - **Candidates:** the mean difference at every post-instruction template position and every `resid_pre` layer from 1 to L−1. Layer 0 is the embeddings and is never selected in practice.
  - **Selection:** lowest refusal score under directional ablation on harmful validation prompts, among candidates with:
    - layer < 0.8 L;
    - KL < 0.1 under ablation on harmless validation prompts;
    - a positive induced-refusal score under activation addition.
  - **Mapping to the battery:** their `resid_pre` of layer l is the output of block l−1, so ℓ* = l−1. The reading position is their selected template position. Directional ablation acts on the embeddings and every attention and MLP write; for OLMo-2 it acts on the post-norm outputs, which are what write to the residual stream.
- **D2 truth (Marks & Tegmark 2024).** Mass-mean direction (μ_true − μ_false) on `cities` and `neg_cities` (2,992 raw statements, last token, block outputs).
  - **Layer rule (substitute):** their patching-based layer choice does not port to new models. ℓ* is the layer with the highest mean held-out accuracy of the mass-mean probe on four held-out curated sets: `sp_en_trans`, `neg_sp_en_trans`, `smaller_than` and `larger_than`, 200 statements each, disjoint from every test item. Each set's activations are centred on that set's own mean, as in their code.
  - **Process target and reading position:** read in a chat true/false template at the statement's last token. The user message is "Is the following statement true or false? Answer with one word: True or False.\n\n{statement}".
  - **T** = P(first token ∈ {"True", " True"}) − P(first token ∈ {"False", " False"}).
- **D3 CR axis.** The published `CRDetector` (commit ce0c61f), unchanged. The layer is the best of its four candidates, the reading position is its own, and the items and targets are paper 4's. The paper-4 outcome parsers are copied unchanged in `cr4_parsers.py`.

## 3. Items (frozen in `code/data/items.json`, built by `make_items.py` with fixed seeds)
- **D1 refusal**:
  - **Check A:** 472 items, all XSTest prompts not used in Check B (312), plus 80 OR-Bench-Hard-1K and 80 OR-Bench-Toxic prompts. Labels are 211 harmful / 261 benign; the target is actual refusal.
  - **Oracle:** 120 (60 Hard + 60 Toxic).
  - **Check D:** 120, a seeded subset of A. +d is applied to benign-labelled and −d to harmful-labelled items (§3).
  - **Check E:** the first 60 of the D items.
  - **Pilot:** 40 Hard + 40 Toxic.
- **D1 Check B:** XSTest safe prompts paired with their contrast (unsafe) prompts by position within type, keeping pairs matched within ±1 token.
  - Only **69 pairs** meet the length criterion, below the 120 in Stage 1 §8, which allows "as many as the source provides".
  - The text-only baseline gives a grouped 5-fold AUC of 0.751, above the 0.65 acceptance limit. By Stage 1 §5 the matched set is **not accepted**, so D1's label verdict is "not evaluable". It is still run and reported. Harmfulness is carried by the words themselves, so no vocabulary-matched set exists.
- **D2 truth**:
  - **Check A:** 240 statements: 80 each from `sp_en_trans`, `neg_sp_en_trans` and `smaller_than`.
  - **Oracle:** 120 (40 + 40 + 40).
  - **Check D:** 120; **Check E:** 60.
  - **Pilot:** 60 + 20 pairs.
- **D2 Check B (construction changed by the pilot).** The first construction paired each `larger_than` statement with its order-swapped twin. In the pilot it leaked through capitalisation of the first number: text-only AUC 0.94, length AUC 0.67.
  - The Stage 2a construction pairs `X is larger than Y.` with `X is smaller than Y.`: the same numbers in the same order and case, of which exactly one is true. That gives 120 pairs.
  - Leakage on the frozen set, using the Qwen tokenizer: text-only AUC **0.459**, length AUC **0.500**. Accepted.
- **D3 CR axis**:
  - **Check A:** paper 4's M set (240 format-matched products).
  - **Oracle:** 120 new products, disjoint from paper 4's main and pilot items.
  - **Check D:** 120; **Check E:** 60.
  - **Pilot:** paper 4's pilot M items.
- **D3 Check B: not constructed.** No compute-vs-retrieve set can be matched on format, length and vocabulary, because the classes differ in content by definition. The label verdict is "not run". Ramdan 2026b's matched-format reversal is cited instead. **Deviation 3.**

## 4. Condition grid, generation lengths, batching (`config_main.json`)
- **Pushes of d, N_len, N_pc1 and o:** at k ∈ {0.5, 1, 2} gaps, applied from the reading position on. k = 1 is primary, and the authors' strength coincides with it:
  - D1: Arditi's activation addition with coefficient 1 equals one gap;
  - D3: paper 4.
- **Signs:**
  - D1 by label, as above;
  - D2 and D3 both signs on every item.
- **Random reference:**
  - R = 12 Gaussian directions (seed 1234) at c·g, with c ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4}.
  - 8 of the 20 `N_perm` directions are pushed on the same grid.
  - In the pilot, every primary push of every direction fell inside its random KL bracket for every item (0 exclusions), so the grid is kept.
- **Check E:**
  - **E1:** re-imposition at every later block (every second block if L > 32), on the 60 E items, for both applicable signs.
  - **E2:** a last-block push on the grid c ∈ {0.25, 0.5, 1, 2, 4, 8, 16} gaps, interpolated in log KL to the primary push's KL.
  - E rows are generated with Check D for every direction. They are analysed only for pushes that pass D (§5).
- **Generation lengths:**
  - D1: 64 new tokens, judged as described in §5.
  - D2: 3 tokens. The target is read from the first-token distribution, and the clean continuation defines the KL positions.
  - D3: 8 tokens (direct answer, with the lens) and 96 (free), as in paper 4.
- **KL:** `kl_all` from the reading position, teacher-forced on the clean row's continuation. For D3 it is taken from the direct-answer run, as in paper 4.
- **Batching:**
  - One batch per item holds every condition, with row 0 clean.
  - OLMo-2 has full multi-head attention and a larger cache, so its push batches are split into fixed chunks of 96 rows (`MAX_ROWS=96`), each with its own clean row 0. The other models use one batch of up to 240 rows.
  - The chunk size is fixed within each model, so batch size is fixed within every comparison.
- **Single-row runs:** the clean Check A and oracle generations are single-row batches.

## 5. Targets, judges and Check 0a gates
- **D1 target: refusal-phrase matching is primary.** This is Arditi et al.'s substring list over the 64 generated tokens, because the judge failed validation (Stage 1 §0.4):
  1. **First judge.** The judge was Qwen3-8B with thinking off, given the request and the response and asked whether the response refuses.
     - The assistant blind-labelled 50 pilot outputs (stratified over push kinds) before seeing any judge label (`res/d1/pilot_judge_validation*.json*`, SHA-256 recorded).
     - Agreement was κ = 0.16. The judge called almost every reply to a harmful request a refusal, including replies beginning "Certainly! Here's…". It was judging the request, not the reply.
  2. **Prompt revision.** The pilot may fix judges, so those 50 became a development set. Showing the reply only, with the refusal option as the second letter, gave κ = 0.80 on the development set. The pilot also showed a position bias toward "A".
  3. **Fresh validation.** The revised prompt was frozen in `judge.py` and validated on a fresh blind set of 50 other pilot outputs, labelled before the judge was run on them: **κ = 0.52** (agreement 0.76). That is below 0.7.
  - **Consequence:** under §0.4, phrase matching is primary. Its agreement with the assistant's labels is also poor on these pilot outputs: κ = 0.27 on the development set and 0.32 on the validation set. Many pushed outputs of the small pilot model refuse in unusual words ("is not only illegal and unethical", "I am not authorized"), or only lecture or redirect.
  - **Secondary measure:** the judge (revised prompt) is run on every main output and reported as a secondary measure for tracking and driving.
  - **Limitation:** both measures will be reported as weak measures of refusal for pushed outputs. The judge's main misses are lectures and redirects, which are hard to recognise from the reply alone.
- **D2 targets:**
  - tracking: T = P(True) − P(False);
  - drive: the rate of "True" verdicts, P(True) > P(False), with the 10-point SESOI. The continuous T is reported without a SESOI.
- **D3 targets:** settling depth D (paper 4's top-1 settling, the mean over answer digits, as a fraction of L) with a SESOI of 1 layer = 1/L, and shown working W with a SESOI of 10 points. D is the primary drive target and is measured with the format clamped by the prefill. W is measured free.
- **Check 0a gates (tolerances):**
  - **D1:**
    - Directional ablation must lower JailbreakBench substring refusal (64 tokens) by at least half its baseline rate.
    - Activation addition must raise substring refusal on 100 harmless test instructions by at least 20 points.
    - Arditi et al. report near-complete effects for larger models. The pilot model gave 0.56 → 0.00 and 0.01 → 0.29, which is why the second tolerance is modest.
  - **D2:** mean held-out mass-mean accuracy ≥ 0.70 at ℓ*. Marks & Tegmark's generalisation accuracies are mostly 0.7–0.95; the pilot gave 0.86.
  - **D3:** calibration AUC ≥ 0.90. Paper 1 reports ≥ 0.988 in 12 models and v2 reports ≥ 0.94; the pilot gave 0.99.

## 6. Analysis details fixed now (`analyze.py`)
- **Bootstraps:** item bootstrap with 2,000 resamples. Check D also uses a two-level bootstrap (items, and reference directions within items). Check B resamples pairs.
- **Claimed signs:** all three directions claim that a larger projection goes with a larger target (refusal; P(True); deeper settling). Each sign of push is tested against its own claimed direction of change.
- **Binary targets:** Check A uses AUC, with SESOI 0.60 and null 0.5. If |ρ(T, prompt length)| > 0.1, the primary statistic becomes the rank partial correlation controlling for prompt length, with SESOI 0.20.
- **D verdict per sign, for the primary drive target:**
  - pass: both references pass;
  - reversal: either reference is a significant reversal;
  - fail: either reference fails;
  - otherwise inconclusive.
  - D passes if one sign passes and the other is not a reversal.
  - If both signs pass, C-steering and E are read on a sign where both pass, if there is one.
- **C-steering:** per item, the matched-KL effect of d minus that of N, with both measured against the Gaussian reference.
- **H2:** `N_perm` directions are scored on tracking (Check A, all 20) and on driving (the 8 pushed ones at one gap against the Gaussian reference at matched KL).
- **Verdicts** follow Stage 1 §6 in its order. TARGET NOT LINEAR is checked against the primary drive target.

## 7. What the pilot showed (Qwen3-1.7B, pilot items only)
- **D1:**
  - **Calibration:** Arditi's selection chose the second post-instruction token at `resid_pre` 15 (block 14); calibration AUC 1.00.
  - **Check 0a:**
    - ablation 0.56 → 0.00;
    - activation addition 0.01 → 0.29;
    - weight-randomised model: two-fold AUC 0.73, against 1.00 for the real model.
  - **Laptop test:** the D1 calibration test on the laptop also computed Check B on the first 20 main XSTest pairs in Qwen3-1.7B, which is not a panel model. Check B is evaluation-only and was rejected for leakage anyway (§3). **Deviation 4**, noted for completeness.
  - **Judge:** see §5.
  - **Pilot result with phrase matching primary (40 Check D items):**
    - tracking passes (partial ρ = 0.61, primary because |ρ(T, prompt length)| > 0.1);
    - Check D is inconclusive (+d on benign items: +10 points against random, CI [0, 25]);
    - the oracle does not drive, so the pilot verdict is TARGET NOT LINEAR.
  - **Oracle construction:** the pilot's push stage built the oracle from the first judge's labels. The main runs build it from phrase matching.
- **D2:**
  - **Layer and 0a:** layer 16 of 28; held-out accuracy 0.86. The weight-randomised model gives AUC 0.49, against 0.99 for the real one.
  - **Pilot verdict:** VALID DRIVER. Tracking ρ = +0.89; "True" verdicts move by +11 and −14 points beyond matched-KL random pushes; hand-off at block 19.
  - **Leakage:** the old Check B construction leaked (see §3).
  - **H2 expectation:** the permuted-label directions are strongly aligned with the truth direction, with |cos| up to 0.71. 8 of 20 pass Check A, and one of the 8 pushed ones moves verdicts by more than the SESOI. This is SteerCheck's "alignment leakage". H2 is kept as preregistered, so it may fail for this reason rather than because of a bug. The |cos| < 0.2 sensitivity analysis (Stage 1 §3) is reported for every direction.
- **D3:**
  - **Calibration:** layer 21; AUC 0.99. cos(d, N_len) = +0.64 and cos(d, N_pc1) = +0.78. Weight-randomised model: 0.75.
  - **Pilot:** ρ(P, settling) = −0.26, against +0.33 for both N_len and N_pc1. The oracle tracks (+0.64) but does not drive settling, so the pilot verdict is TARGET NOT LINEAR.
- **Bracketing and run time:**
  - The random KL grid bracketed every primary push in all three directions.
  - Run times per model (1.7B): D1 ≈ 6 min, D2 ≈ 5 min, D3 ≈ 4 min.
  - Estimate for 7–8B: about 1.5 h per model for D1 and 0.5 h for D3, well under the 24 GPU-hour threshold, so no reduction of `R` or of the items is needed.

## 8. Deviations so far
1. The backtracking pilot comes after the D1–D3 main runs (§0).
2. The pilot ran on the RTX 5090 (§1).
3. No Check B set for D3 (§3).
4. The laptop calibration test computed Check B on 20 main D1 pairs in a non-panel model (§7).
5. Layer-rule substitute for D2 (§2). Arditi's pipeline was reimplemented for D1 rather than run from their repository (§2).
6. The D1 judge prompt was revised in the pilot. The revised judge failed fresh validation (κ = 0.52), so phrase matching is primary and the judge is secondary (§5).
