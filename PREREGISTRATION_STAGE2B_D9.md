# Paper 6, Stage 2b: settings for D9 (true thinking) after its pilot

> Frozen 2026-10-15 (time in `PREREG_SHA256SUMS`), after the D9 pilot (Qwen3-1.7B, thinking on; pilot items only) and before any main-panel D9 run.
>
> The AI assistant wrote it in a session started from the main paper-6 session, on the author's standing instruction to make design choices itself and record them (Stage 1 §0). As with Stage 2a and Stage 2b-D5, it sets only what Stage 1 §10 allows.
>
> D9 was built in parallel with D4, D6, D7 and D8. To avoid collisions, its deviations are numbered D9-1, D9-2, …. The D9 code runs from `jspace/paper6_d9`; the shared-file changes it needs are in `INTEGRATION_D9.md`.

## 1. Models
| key | model | revision |
|---|---|---|
| r1-qwen-7b (home) | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 916b56a44061fd5cd7d6a8fb632557ed4f724f60 |
| r1-llama-8b | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 6a6f4aa4197940add57724a7707d069478df56b1 |
| q3-8b | Qwen/Qwen3-8B, `enable_thinking=True` | b968826d9c46dd6066d109eabc6255188de91218 |
| q3-1.7b (pilot) | Qwen/Qwen3-1.7B, `enable_thinking=True` | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |

- Every prompt is the model's chat template applied to the bare problem. There is no instruction suffix, as in Zhao et al.'s `inference_vllm.py`.
- Every prompt ends with `<think>\n`. The R1 templates add it; for Qwen3 it is appended, as for D5.
- All runs are greedy, in bf16, and use a single BOS token (D9-5).

## 2. Direction (Zhao et al.'s recipe, rebuilt in each panel model)
Source: github.com/andotalao24/Identify_true_decorative_thinking, commit `e1da4c0` (`third_party/ttd`). It was read as data; none of its code was run.

**What the release contains**
- Three example directions in `data/pt`:
  - `amc-qwen-dir-uf-ff.pt`, [29, 3584];
  - `llama8b-amc-uf-ff-dir.pt`, [33, 4096];
  - `nemotron-1d5b-amc-dir-uf-ff.pt`, [29, 1536].
- Each is fp16. Index l is the input of block l, so index L is the last block's output.
- Each points from true-thinking to decorative steps ("uf − ff").
- It also contains model outputs for some sets, and the scripts that compute the True Thinking Score.
- It contains **no** per-step TTS labels and no step files for the directions. TTS is therefore recomputed with their procedure in every model.

**Calibration set**
- Their in-domain set: the 83 AMC 2022–23 problems of `AI-MO/aimo-validation-amc`. Its fields (id, problem, answer, url) match their released AMC outputs.
- Each model writes its own greedy chain, capped at 8,192 new tokens (D9-6).
- A chain is kept if its last `\boxed{}` answer equals the reference. This is their "right" filter.

**Steps and TTS**
- **Steps.** The full output is split at every ".", their delimiter, and the empty last piece is dropped.
- **Score.** This follows their `run_checkpoint_analysis.sh` and `tts.py`.
  - *y\** is the chain's own answer.
  - *p(·)* is the confidence in y\* after the prefix plus `"\n The final answer is \boxed{"`, teacher-forced (D9-3).
  - *s′*: every number in the step becomes old + one draw from {−3, −2, −1, 1, 2}, with a fresh seed-100 generator per step. This is their `random_add_small` with `replace_all`.
  - *C′*: every earlier step is perturbed in the same way.
  - TTS = ½(|p(C,s) − p(C,s′)| + |p(C′,s) − p(C′,s′)|). The "." delimiter is used in all four conditions (D9-4).
- **Which steps.** TTS is computed for every numeric step, i.e. every step the perturbation changes. Their direction uses only numeric steps. The (C, s) early exit is computed for every step.
- **Implementation.** Each pair of conditions is run as one batch, branched from one shared KV-cache position (D9-12).
- **Case filter.** Their `is_consistent` filter is applied (D9-14).

**Classes and direction**
- **Classes.** True-thinking: TTS > 0.9. Decorative: TTS ≤ 0.005 (D9-2).
  - Up to 200 steps of each class are drawn with `random.seed(100)`, and the first 100 are used, as in `tts.py` and `run_diff_mean.sh`.
  - If a class has fewer than 100 steps, all of them are used.
- **Activation.** This follows `extract_hidden.py`.
  - The text is `"<chat prompt>. <earlier steps joined by '.'>. <step>"`.
  - The reading position is the step's last token.
- **Layer.** Their example layer is index 17 of 28. It is the layer of their released steering example `amc_qwen7b_perturb_rand_small_DT_dir-intervene17.json`, inside their stated range of 15–22. Its pre-hook input is our block output 16.
  - Other models use the same relative depth: block output round(17·L/28) − 1.
  - That gives 16 in R1-Distill-Qwen-7B and Qwen3-1.7B, 18 in R1-Distill-Llama-8B, and 21 in Qwen3-8B (D9-1).
- **Direction.** d = mean(true-thinking) − mean(decorative), the TrueThinking direction, which is −1 × their files' orientation.
  - Every claim tested here is positive. Projection rises with TTS (tracking). +d makes the answer follow the step (driving).
- **Gap.** g = ‖d‖, because d is exactly the class mean difference.
  - Their steering uses coefficient 1 on the raw vector, which is one gap. Their strength is therefore the primary push, and Check D needs no separate `d_auth` row.
- **Control directions.**
  - `N_len`: a within-class median split on the token position of the step's last token in the extraction text.
  - `N_pc1` and 20 `N_perm`: as in Stage 1, from the same calibration activations.

## 3. Items (`code/data/items_d9.json`, `make_items_d9.py`, seeds 90 and 91)
**Source and splits**
- Test problems come from the MATH test set (`EleutherAI/hendrycks_math`), Zhao et al.'s out-of-domain set (D9-8). They are restricted to:
  - Levels 1–3;
  - an integer reference answer (the last `\boxed{}` of the solution);
  - at least one digit in the problem.

  That leaves 1,588 eligible problems.
- The splits are disjoint from each other and from AMC:
  - Check 0a: 40 problems;
  - pilot: 40 (Check B pilot: 30);
  - Check B: 120;
  - oracle: 120;
  - Check A: 240;
  - Check D: 120 of the Check A problems;
  - Check E: the first 60 of D.
- **Chains.** Greedy, capped at 4,096 new tokens. y\* is the last `\boxed{}` answer; if there is none, it is the greedy early-exit answer after the whole chain.

**Check A**
- One unit per numeric step of the 240 chains (D9-9).
- The target is the step's TTS. The projection is read at the extraction position.
- The bootstrap resamples problems.
- The covariates for the partial statistic are the step's token position and its token length.
- The pilot's ρ(TTS, position) was −0.12. That is beyond 0.1, so the partial ρ is primary, as Stage 1 §5 requires.

**Oracle**
- The tercile split of TTS over the oracle chains' numeric steps (1,731 in the pilot).

**Check D units (D9-11)**
- Each Check D problem gets one perturbed step from the model's own chain.
- The preference alternates in item order between two kinds:
  - "dt": the answer before the step is y\* and the answer after s′ is not (their disengagement-test condition);
  - "et": the answer stays y\* (their engagement-test condition).
- If the preferred kind is missing, the other kind is used, then any numeric step. The draw is seeded by the problem id.
- Only steps whose intervention-format prompt is under 2,048 tokens are eligible (§4).
- In the pilot, the 20 units were 6 dt, 10 et and 4 any.

**Check B (matched label set)**
- Within each Check B chain, a step with TTS > 0.9 is paired with the nearest unused step with TTS ≤ 0.005 whose token length is within ±1. That matches template, problem vocabulary and step length (D9-13).
- Position in the chain is not matched; it is reported as the "length AUC".
- At most 120 pairs are taken. Check B is not run if there are fewer than 5 pairs.

## 4. Drive target, pushes, grid, generation
**Prompt, push and target**
- The prompt is `"<chat prompt>. <context>. <s′>" + cue`, Zhao et al.'s intervention input.
- The reading position and the start of the push are the last token of s′. The push follows Stage 1 §2: from there to the end, through the cue and the generated answer (D9-10).
- The answer is decoded greedily for 16 tokens and read up to the brace that closes `\boxed{`.
- **F** = 1 if the answer differs from y\*, i.e. it follows the perturbed step.

**Signs and SESOI**
- Every unit is pushed in both signs.
- +d is evaluated on units whose clean answer stays y\* (row-0 F = 0: engagement). −d is evaluated on units whose clean answer follows s′ (F = 1: disengagement).
- Check D passes if one sign passes and the other is not a reversal.
- The SESOI is 10 points of F (Stage 1 §0.5).

**Grid**
- Stage 2a's grid, unchanged: k ∈ {0.5, 1, 2} gaps for d, N_len, N_pc1 and o, in both signs; R = 12 Gaussian directions and 8 N_perm directions, each × c ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4}; E1 at every later block; E2 on the 7-step grid.
- **Pilot bracket.** It brackets every push. 0 of 20 units were excluded for either reference or sign.
  - The KL medians of the one-gap push were 10.6 (+d) and 6.5 (−d), against 0.1–40 across the random grid.
  - At k = 2 they were 32 and 21.

**Format clamp**
- Not applicable. The cue fixes the answer format by prefill, so FORMAT-MEDIATED cannot arise.

**Memory and batching (`P6_MEM_FRACTION=0.85`)**
- *Chain batch.* The batch size is min(10 GB, 75% of the free memory after the model) divided by the cache size of one chain at full length.
  - On running out of memory, a batch is split in two and retried.
  - Rows that end leave the batch.
- *Push batch.* Push prompts carry a chain prefix of up to 2,048 tokens, so `MAX_ROWS` is:
  - 12 for r1-llama-8b and q3-8b. In the Qwen3-8B timing probe, 24 rows of a 1,700-token prompt ran out of memory.
  - 16 for r1-qwen-7b, whose KV cache is 57 KB per token against 131–147 KB, but whose MLP is wider.
  - 40 for the pilot.
- None of these settings changes a computed value beyond floating-point reduction order.

## 5. Check 0a: reproduction and gates
**(i) Released vectors**
- In R1-Distill-Qwen-7B and R1-Distill-Llama-8B: the cosine between the rebuilt d and −released at the same layer (released index ℓ\*+1). The per-layer cosines are reported too.
- **Reliability from the pilot.** Split-half reliability of the rebuilt direction, with halves split by chain so they share none: median cosine 0.70 [0.61, 0.76] over 500 splits. The implied cosine between two independent full-size rebuilds is ≈ 0.83.
  - That is the ceiling expected for a rebuild from different chains even with an identical procedure.
  - Theirs differs in its chains, in fp16, in a double BOS, and in the delimiter handling of D9-4.

**(ii) Their headline test, at ℓ\***
- Their intervention: coefficient 1 on the raw vector, added on the step's tokens in the prompt pass only.
- Items: up to 100 engagement and 100 disengagement steps from the 40 Check 0a chains.
  - Engagement (ET): the answer before and after s′ is y\*. A flip is the answer moving off y\* under +d.
  - Disengagement (DT): the answer before s′ is y\* and after it is not. A flip is the answer returning to y\* under −d.
- Baseline: their control, 4 Gaussian vectors of the same norm.
- In the R1 models, the released vector is also run (descriptive).

**Gate (`stage2_settings.json` → `check0a.d9`)**
- reproduced = cos(rebuilt, −released) ≥ **0.5** (R1 models only), **and** max(ET advantage, DT advantage) ≥ **5 points** over the random vectors.
- **Tolerances.**
  - The cosine bound sits well below the ≈ 0.83 reliability ceiling and far above chance (|cos| ≈ 0.02 in 3,584 dimensions).
  - The 5-point bound sits below every TrueThinking-vs-random advantage in their Table 2. The smallest is DT on MATH in Llama-3.1: 54.2 against 47.9, i.e. 6.3 points.
  - Their numbers are the best over layers; ours are at one fixed layer.
- **Pilot.** ET 1.0% against 0.25% random (+0.75 points). DT 35% against 20.75% random (+14.25 points). The pilot passes on DT, and has no released vector.

## 6. What the pilot showed (Qwen3-1.7B, thinking on; pilot items only)
**Calibration**
- Of 83 AMC chains, 43 were correct and boxed, and 39 finished within the cap. Every correct chain passed the `is_consistent` filter.
- They held 7,846 steps, of which 5,556 were numeric.
- TTS:
  - mean 0.067;
  - 7.9% > 0.3;
  - 1.8% > 0.7;
  - 1.2% > 0.9;
  - 59% ≤ 0.005.

  That is close to the paper's Qwen-2.5 AIME figures (mean ≈ 0.03, 6.4% > 0.3, 2.3% > 0.7).
- Pools: 3,296 decorative and **65** true-thinking steps, so the direction uses 65 + 100.
- Layer 16: ‖d‖ = g = 89.5, against a median residual norm of 249.
- Calibration AUC: d 0.985 (in sample), N_len 0.36, N_pc1 0.59.
- cos(d, N_len) = −0.06 and cos(d, N_pc1) = +0.20.
- Check 0b: two-fold AUC 0.96 in the real model and 0.47 in the weight-randomised one.

**Check B**
- 13 pairs from 30 chains. AUC of d 0.91 [0.83, 0.99], but the text-only AUC was 0.68 > 0.65, so the pilot's matched set would be rejected.
- Length AUC 0.32.
- The main set (120 chains) should give about 50 pairs. The construction is not changed.

**Check A (40 chains, 3,450 numeric steps)**
- TTS mean 0.09: 0.96% > 0.9 and 49% ≤ 0.005.
- Of the 40 chains, 37 had the right answer and 72% finished within the cap.

| projection | partial ρ [95% CI] | verdict |
|---|---|---|
| d | +0.33 [+0.20, +0.44] | pass |
| N_len | −0.09 | fail |
| N_pc1 | +0.08 | fail |
| oracle | +0.44 | pass |
| d⊥ | +0.33 | 102% of d retained |

- C-tracking: pass. d beats N_len by +0.41 [+0.24, +0.58] and N_pc1 by +0.24 [+0.14, +0.34].
- N_perm: 3 of the 20 label-permuted directions "pass" Check A.
  - Two have ρ ≈ 0.31 and cos 0.40–0.45 with d: the alignment leakage already seen for D1–D5 (H2).
  - One has ρ = 0.21 with cos 0.04, i.e. without leakage. So directions unrelated to d can also track TTS weakly.

**Check D at one gap (20 units: 12 engagement, 8 disengagement)**
- **+d (engagement):** raw ΔF +0.08, but Gaussian pushes at matched KL gave +0.22.
  - Effect −0.13 [−0.24, −0.04] against Gaussian pushes: a **reversal**.
  - Effect −0.05 [−0.16, +0.05] against permuted-label pushes.
- **−d (disengagement):** effect −0.00 [−0.26, +0.21], inconclusive.
- **Oracle:** reversal (+) and inconclusive (−). It therefore fails D.
- **Pilot verdict: TARGET NOT LINEAR.** Tracking passes A and C, but neither d nor the oracle drives the answer beyond KL-matched random pushes.
- **Label verdict:** NOT EVALUABLE, because the matched set was rejected.

**Run time (RTX 5090)**
- Pilot stages: cal 69 min, clean 11 min, push 4 min, analyse 1 min.
- Qwen3-8B timing probe (pilot problems only):
  - chain generation 59 ms per token, with batch 12 at the 4,096 cap and batch 6 at the 8,192 cap;
  - scoring 0.13–0.15 s per numeric step;
  - Qwen3-8B chains on MATH L1–3 often reach the 4,096 cap (5 of 12).

**Main-run estimate per model**
- q3-8b: cal ≈ 4 h (AMC chains ≈ 1.9 h, Check B chains ≈ 1.2 h), clean ≈ 3.7 h, push ≈ 1.7 h. Total ≈ 9.5 h.
- r1-llama-8b: ≈ 8.5 h.
- r1-qwen-7b: ≈ 5 h.
- Panel: ≈ 23 GPU-hours. Every (direction, model) is below the 24-hour threshold of Stage 1 §8, so R and the item counts are not reduced.

**Pilot runs discarded**
- The first pilot calibration ran out of GPU memory in its first batch of chains (21 chains at 8,192 tokens; 8.8 GB reserved but unallocated). It wrote no output.
- The clean, push and analyse jobs queued after it failed at once for want of its output.
- The batch rule of §4 replaced it. Only the second, complete run (`*_r2` logs) is analysed.

## 7. Deviations and decisions
- **D9-1 (layer).** Zhao et al. report the best layer of a sweep ("15–22" for Qwen-2.5). The battery needs one layer, so their released example's layer, index 17 of 28, is used, at the same relative depth in the other models.
- **D9-2 (decorative threshold).**
  - Their code selects decorative steps with TTS ≤ 0. In their fp16 pipeline that means a probability saturated at exactly 1.0 in all four conditions.
  - With fp32 log-probabilities exact zeros do not occur, so the paper's own decorative cut-off, TTS ≤ 0.005, is used.
- **D9-3 (confidence).** p(y\*) is teacher-forced, which is the paper's definition, Pr(y\*|q,C,s).
  - Their code multiplies greedy-path probabilities when the greedy answer is not y\*. Otherwise it is the same.
  - "The answer is y\*" is tested exactly: greedy decoding gives y\*'s tokens followed by "}".
- **D9-4 (delimiters).** Their code drops the "." before the perturbed step in the (C, s′) condition, and adds an extra space before s in (C′, s). Here the "." delimiter is used in all four conditions, as the paper defines them.
- **D9-5 (BOS).** Their pipeline tokenises chat-templated text with special tokens added, which gives two BOS tokens in the R1 models. One BOS is used here.
- **D9-6 (caps and y\*).**
  - Their generation had no cap. Here it is 8,192 new tokens for AMC calibration chains and 4,096 for MATH chains.
  - For test chains without a boxed answer, y\* is the forced early-exit answer. Calibration keeps only correct boxed chains, as they did.
- **D9-7 (own chains).**
  - Their released chains do not cover every panel model. The AMC file labelled qwen7b is in Qwen2.5-Math style, without `<think>`.
  - No step labels are released.
  - So every model's direction is rebuilt from its own chains, with TTS recomputed by their procedure.
- **D9-8 (test problems).** MATH Levels 1–3 with integer answers, their out-of-domain set, rather than AMC/AIME. This bounds chain length and keeps answers integer.
- **D9-9 (Check A units).** Numeric steps only. Their drop-step variant for steps without numbers is not used, so T has one definition throughout.
- **D9-10 (pushes).** Check D uses Stage 1's push from the reading position onward. Their own intervention (step tokens, prompt pass only) is run in Check 0a, as Stage 1 §2 requires.
- **D9-11 (Check D units).**
  - One unit per Check D problem, chosen from the model's own chain.
  - Both signs are pushed. The engagement/disengagement split is made on the clean answer in row 0.
  - Units are restricted to prompts under 2,048 tokens.
- **D9-12 (numerics).** Each TTS pair is computed as one batch from one shared cache prefix, so identical prompts give identical scores.
  - In the smoke test, cached branches and plain forward passes differed by up to 0.07 nats on a low-confidence answer (p ≈ 0.3). This is bf16 reduction order.
  - Check A's step activations are read from the cached branches. The calibration activations come from plain forward passes, as in their `extract_hidden.py`. Relative difference: 1.8% of the hidden-state norm.
- **D9-13 (Check B).** Step length is matched; position is not (no within-chain construction can match it).
- **D9-14 (case filter).** Their `is_consistent` reads the (C, s′) answer at every checkpoint. For steps without numbers, (C, s′) equals (C, s), and that answer is used.
- **D9-15 (code location).** The D9 code runs from `jspace/paper6_d9`: copies of the shared files with the changes in `INTEGRATION_D9.md`. The hashes are in `PREREG_SHA256SUMS`.
- **D9-16 (no judge).** D9 needs no judge. TTS is computed causally, and F is parsed from the boxed answer, so Stage 1 §0.4 does not apply.
