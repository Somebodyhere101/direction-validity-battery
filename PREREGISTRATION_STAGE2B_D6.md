# Paper 6, Stage 2b: settings for D6 (reasoning strength) after its pilot

> Frozen 2026-10-15 19:15 AST, after the D6 pilot (Qwen3-1.7B, pilot items only, plus run-time probes of the panel models on pilot items) and before any main-panel D6 run.

The AI assistant wrote this on the author's standing instruction (Stage 1 §0), in a session that built D6 in parallel with D4, D7 and D9. Like the other Stage 2 parts, it sets only what Stage 1 §10 allows: norm grids, R, item counts, generation lengths, matched-set constructions and Check 0a tolerances. It changes no check, pass criterion, SESOI, verdict rule, hypothesis or direction. Hashes are in `PREREG_SHA256SUMS`. No main-panel D6 data existed when it was frozen.

## 1. Models
| key | model | revision |
|---|---|---|
| r1-qwen-7b (home) | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 916b56a44061fd5cd7d6a8fb632557ed4f724f60 |
| r1-llama-8b | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 6a6f4aa4197940add57724a7707d069478df56b1 |
| q3-8b | Qwen/Qwen3-8B, `enable_thinking=True` | b968826d9c46dd6066d109eabc6255188de91218 |
| q3-1.7b (pilot) | Qwen/Qwen3-1.7B, `enable_thinking=True` | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |

Setup: bf16, torch 2.14.1+cu130, transformers 5.19.0, RTX 5090 with `P6_MEM_FRACTION=0.85`. The code runs from `jspace/paper6_d6` on DQ. It uses the frozen `p6lib.py` and `specs.py` unchanged, plus D6's own files (§9).

## 2. Direction (Sheng et al.'s recipe, rebuilt in each panel model)
Source: github.com/AlphaLab-USTC/LRM-plans-CoT, commit `2ba2485`, in `third_party/LRM-plans-CoT`. It was read as data; none of its code was run. They release no vectors: their steering scripts load a vector file the user builds.

- **Prompt.** Their `embed.py` template, rendered exactly as their code renders it: `"Please reason step by step, and put your final answer within \x08oxed{}\nThis is the problem:\n" + problem + "\n    "`.
  - The `\b` of `\boxed` in their Python source is a backspace character, and the template ends in a newline and four spaces. Both are kept byte for byte, because the vector is defined on these prompts.
  - The user message goes into the model's chat template with the generation prompt. The R1 templates end it with `<think>\n`; for Qwen3, `<think>\n` is appended, as for D5.
  - Their steering-evaluation script indents the template by four spaces. The calibration form is used for every prompt here (**D6-1**).
- **Calibration set.** Their question file `Data/Questions/MATH.json` (git blob `d5d8a78`): the 7,500 MATH training problems.
  - By level: 564 / 1,348 / 1,594 / 1,690 / 2,304 at levels 1–5. The two "Level ?" problems count as level 3, as in their `AnalyzeOneModel.py`.
  - Saved as `code/data/calibration_d6.json`.
- **Activations.** The hidden state at the last prompt token (their `embed.py`: `hidden_states[:, -1, :]`), at every block output.
- **Vector at every layer.** r = the mean over k = 2..5 of (mean of level k − mean of level 1). This is the mean of their four `r_{k←1}` (`utils.construct_diff_df`, then the mean in `AnalyzeOneModel.py`).
- **The battery's claim tuple.**
  - **ℓ\*.** Their single-layer index, hidden state `int(0.8 × (L + 1))` ("Extract steering vectors / Single layer" in `AnalyzeOneModel.py`), i.e. block output `int(0.8 × (L + 1)) − 1`. That is 22 in the 28-layer Qwen models, 25 in R1-Distill-Llama-8B and 28 in Qwen3-8B.
    - Their paper says the single-layer figure uses "the layer with the highest averaged similarity". That layer is data-dependent; it is reported but not used (**D6-2**).
  - **d and reading position.** d = r at ℓ\*. The reading position is the last prompt token, the `\n` after `<think>`.
  - **Gap.** g = ‖d‖, the projection gap under the authors' level-mean weighting. The pooled gap (levels 2–5 against level 1) is also reported.
- **Label classes for the controls** are the five levels:
  - `N_len`: within each level, the long half minus the short half by formatted prompt length, averaged over levels (Stage 1 §3: "question length within difficulty level").
  - `N_pc1`: the first principal component after centring each level on its own mean.
  - `N_perm`: the authors' construction after a random permutation of the five level labels (sizes kept), 20 seeds.
  - The calibration AUC, Check 0b and Check B use levels 2–5 against level 1.
- **The authors' intervention** is used in Check 0a and as the secondary `d_auth` push in Check D.
  - It adds λ·r^(i) to the input of every decoder layer i ≥ 1, i.e. to block outputs 0 … L−2. It acts at every position from 0, prompt and generated tokens alike (their `steer_qwen2_vllm.py`).
  - Their r^(0), at the embedding output, is exactly zero, because the last prompt token is always the same token.
  - **λ_a, R1-Distill-Qwen-7B:** 0.2, the end of their sweep (−0.2 … 0.2).
  - **λ_a, R1-Distill-Llama-8B:** 0.02, their Table 10 value for that model ("more sensitive").
  - **λ_a, Qwen3:** Qwen3 is not in their study. The rule was fixed before the λ sweep was read: λ_a is the smallest value in {0.2, 0.5, 1.0} at which the pilot model, Qwen3-1.7B, meets the Check 0a gate of §6 on the 20 Check 0a problems; if none does, λ_a = 0.2. **None did (§7), so λ_a = 0.2 for Qwen3-8B.**
  - As a Check D push, it is run from position 0 at every layer, both signs, 4 samples. It is KL-matched like any other push. Stage 1 names "the authors' published strength" as a secondary push; here their intervention differs in layers as well as strength (**D6-3**).

## 3. Process target and sampling
- **T** = the number of thinking tokens: tokens generated after the prompt (which ends with `<think>\n`) and before `</think>`, mean over 4 samples (Stage 1 §3).
- **Sampling.** Temperature 0.6 and top-p 0.95, the settings of their `generator.py` and `ApplySteeringHF.py` and DeepSeek's recommendation. Their vLLM steering evaluation left top-p at 1.0 (**D6-4**).
  - Sampling is Gumbel-max inside the nucleus.
  - The noise is a fixed hash of (problem, sample index, token step, token id) (`p6lib_d6.py`). The clean chain and every pushed chain of the same sample therefore share their random numbers until they diverge.
- **Cap: 4,096 thinking tokens** (the authors used 16,384) (**D6-5**). A chain that reaches the cap without `</think>` is counted at the cap (T = 4,096) and flagged `capped`. A chain that ends its turn without `</think>` (rare) is counted at its length.
  - The capped share is reported per condition.
  - Check A is repeated, as a sensitivity analysis, on items none of whose clean samples were capped.
  - In the pilot timing probes, the share of clean chains that hit the cap was 18.8% (R1-Distill-Qwen-7B), 10.0% (R1-Distill-Llama-8B), 27.5% (Qwen3-8B) and 26% (Qwen3-1.7B). With an 8,192 cap, the 8B models fit only 6–7 rows in memory and the reduced design alone would need more than 60 GPU-hours per 8B model, so the cap is set by compute.
- **Clean stage.** Up to 512 tokens after `</think>` are generated so that the boxed answer can be read. Accuracy is descriptive only.

## 4. Items (`code/data/items_d6.json`, `make_items_d6.py`, seeds 60–64)
- **Source.** MATH test problems: EleutherAI/hendrycks_math, revision `21a5633`. The 5,000 rows were fetched as JSON through the HF datasets-server API (`third_party/hendrycks_math/test.json`), because DQ has no parquet reader (**D6-12**).
  - They are disjoint from the calibration set, which is the training split. One test problem whose text also occurs in the training file is dropped.
- **Splits.** Check 0a uses 20 MATH-500 problems (HuggingFaceH4/MATH-500, revision `6e4ed1a`, their steering evaluation set), 4 per level. Every other split comes from the 4,499 test problems outside MATH-500 and is stratified by level:
  - pilot: 40 (8 per level); pilot Check D: 15 (3 per level; 10 used, §7); pilot Check E: the first of these (2 used);
  - oracle: 120 (24 per level);
  - Check A: 240 (48 per level);
  - Check D: 120 of the A problems, 24 per level, ordered so that every prefix cycles through the levels (the first 80 hold 16 per level);
  - Check E: the first 60 of D.
- **Check A statistic.** The partial Spearman ρ(P, T) controlling for MATH level (dummies), subject (dummies) and the problem's token length (Stage 1 §3). The plain ρ is secondary. The bootstrap resamples problems.
- **Check B (matched label set).** Level-1 test problems are paired with level-2–5 problems of the same subject whose formatted prompts are within ±1 token.
  - Pairs are made greedily in a fixed seeded order, at most 120 of them, from a pool disjoint from every other split (254 level-1 and 3,485 level-2–5 problems). The pilot used a separate pool and 30 pairs.
  - Pairs are matched per model at run time, because the tokenizers differ. They are used for evaluation only.
  - The set is accepted if its text-only AUC is ≤ 0.65 (Stage 1 §5). In the pilot model it was 0.62, with a length AUC of 0.505.

## 5. Drive target, pushes, grid, generation, run time
- **Drive target:** T under the push, with the same cap and sampling.
  - Each push row's change dT is computed against the clean chains of the same samples.
  - **SESOI:** 10% of the mean clean T over the model's Check D items (Stage 1 §3).
- **Pushes**, from the reading position on, at every generated token:
  - d at k ∈ {0.5, 1, 2} gaps, both signs. k = 1 is primary, with 4 samples. k = 0.5 and 2 are exploratory, with 1 sample each (sample 0).
  - N_len, N_pc1 and o at k = 1, both signs, 4 samples.
  - The authors' intervention (§2) at ±λ_a, 4 samples.
  - Random and permuted reference directions: one sample per direction and norm (sample index j mod 4), against that sample's clean chain (**D6-6**). Each reference point is therefore a mean over 8–12 single-sample changes.
- **KL.** Teacher-forced on the clean sample 0's chain, up to and including `</think>` (or the cap), summed from the reading position (Stage 1 §2). It is computed once per distinct push.
- **Grid.** The pilot grid c ∈ {0.25, 0.5, 1, 2, 4, 8, 16} bracketed every push in every pilot item (0 exclusions; §7).
  - **Full design (Stage 1 §8):** R = 12 Gaussian directions × c ∈ {0.25, 0.5, 1, 2, 4, 8, 16}, 8 `N_perm` directions × the same 7 norms, 120 Check D items.
  - **Reduced design (Stage 1 §8 long-generation rule):** R = 8 × c ∈ {0.5, 1, 2, 4, 8}, 8 `N_perm` × the same 5 norms, the first 80 Check D items (16 per level).
  - Chains per Check D item: 188 (full) or 128 (reduced).
- **Run-time estimate and the reduction** (`estimate_d6.py`; the pilot's push file plus timing probes of each panel model at cap 4,096 with the continuous-batching engine). Values are GPU-hours per (direction, model), at the probes' slot counts. The main runs reserve less memory and get about 15% more slots in the 8B models, so these are upper bounds.

  | model | throughput (tok/s, slots) | full design | reduced design | applied | Check E, per passing sign |
  |---|---|---|---|---|---|
  | R1-Distill-Qwen-7B | 966 (36) | 19.5 | 9.6 | **full** (≤ 24 h) | 3.4 |
  | R1-Distill-Llama-8B | 470 (14) | 32.9 | 16.2 | **reduced** | 7.6 |
  | Qwen3-8B | 336 (12) | 64.5 | 31.6 | **reduced** (still > 24 h) | 8.9 |

  The rule is applied per model, as Stage 1 §8 states it per (direction, model) (**D6-7**). Qwen3-8B stays above 24 h after the reduction. Stage 1 permits no further reduction, so it is run as is.
- **Check E** runs only for the push signs that pass Check D (Stage 1 §5), after Check D has been analysed (`pushE … auto`) (**D6-8**). It uses 4 samples, E1 at every later block (every second if L > 32) and E2 on c ∈ {0.25, 0.5, 1, 2, 4, 8, 16}. Batches run in lockstep with the clean twin in row 0, which re-imposition reads.
- **A model that fails the Check 0a gate** skips its clean, push and Check E stages. Under Stage 1 §5 it is "not reproduced" and leaves the confirmatory analysis in that model. The skipped stages can be run later as exploratory (`P6_IGNORE_GATE=1`) (**D6-9**).
- **Generation engine** (`p6lib_d6.py`; **D6-10**):
  - *Continuous batching.* Finished chains leave the batch and queued chains take their slots.
  - *KV cache.* A preallocated cache that grows in chunks, instead of transformers' DynamicCache, which copies the whole cache at every token.
  - *Decoding attention.* Grouped-query heads are handled as one batched matmul, instead of SDPA's expansion of keys and values to every query head when a padding mask is present. Prompt passes keep SDPA.
  - *Pilot check.* The decoding attention gave the same argmax as SDPA, with a largest logit difference of 0.375 on a logit scale of 34, i.e. bf16 rounding.
  - *Allocator.* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, set by `run_d6.py`. The cache buffers change size often, and under the default allocator a cache-like test on DQ left 6.6 GB reserved but unusable, against 0.8 GB with expandable segments.
  - *Out-of-memory fallback.* If PyTorch's capped allocator runs out while decoding, the quarter of rows with the least work done is re-queued; their noise reproduces their chains. If it runs out while admitting rows, the merge is rolled back and the rows are re-queued.

## 6. Check 0a gate
- **(i) Descriptive.** The structure the authors report, at ℓ\* and across layers:
  - the pairwise cosines of the four `r_{k←1}`;
  - their norms by level;
  - the layer of highest mean cosine;
  - the calibration AUC and ρ(projection, level).
- **(ii) Gate: their headline steering result** ("steering … can change the reasoning strength"). Their intervention at ∓λ_a, 4 samples on the 20 Check 0a problems, sharing noise with λ = 0. The gate passes if both hold:
  - T(−λ_a) ≤ **0.90** · T(0), i.e. −λ_a shortens thinking by at least the drive SESOI of 10%;
  - T(+λ_a) ≥ T(0), i.e. +λ_a does not shorten it.

  The authors' own reported changes are larger than this threshold. For R1-Distill-Qwen-7B on GPQA they report about −85% at −0.2 and +20% at +0.2; for R1-Distill-Llama-8B at ±0.02 on MATH500, −12% and +6%. The 4,096 cap censors the increases, so only the absence of a decrease is required at +λ_a (`stage2_settings_d6.json`; **D6-11**).

## 7. What the pilot showed (Qwen3-1.7B, thinking on)
- **Calibration.**
  - ℓ\* = 22; g = ‖d‖ = 161.3 (pooled gap 175.7), against a median residual norm of 1,085.
  - Calibration AUC (levels 2–5 vs 1): d 0.82, N_len 0.81, N_pc1 0.78. ρ(projection, level) = 0.57.
  - **d is mostly a length / within-level-variance direction:** cos(d, N_len) = +0.83 and cos(d, N_pc1) = +0.86.
  - The four `r_{k←1}` are nearly parallel (mean pairwise cosine 0.97 at ℓ\*; 0.98 at block 17, the highest). Their norms grow with level (65, 133, 185, 268), as the authors report.
  - |cos(d, N_perm)| reaches 0.81, so the low-|cos| sensitivity analysis of Stage 1 §3 applies.
- **Check 0b.** Two-fold calibration AUC 0.81 in the trained model and 0.49 in the weight-randomised one.
- **Check B (30 pilot pairs).** Text-only AUC 0.62 (accepted). Length AUC 0.505. d AUC 0.75 [0.66, 0.85]; N_len 0.73.
- **Check 0a.** At λ = 0.2: T(0) = 2,626, T(−0.2) = 2,568 (×0.98), T(+0.2) = 2,720 (×1.04); 29–34% of chains capped. The pilot model **fails** the gate at the authors' Qwen strength.
- **λ sweep** (same 20 problems × 4 samples; streaming engine, so different random draws from Check 0a; T(0) = 2,510):

  | λ | T | ratio to λ = 0 | capped |
  |---|---|---|---|
  | −1 | 3,348 | ×1.33 | 81% |
  | −0.5 | 2,363 | ×0.94 | 31% |
  | +0.5 | 3,104 | ×1.24 | 42% |
  | +1 | 4,096 | — | 100% |

  - At −1 the chains turn into repetitive gibberish that never ends.
  - At −0.5 a few chains skip thinking altogether, but the mean falls by only 6%.
  - +λ lengthens thinking coherently; at +1 no chain closes `</think>` within the cap.
  - So in the pilot model the steering lengthens but does not shorten reasoning by 10% before it breaks the output. No λ meets the gate.
- **Clean chains (40 pilot problems).**
  - Mean T is 2,236, and 19% of chains are capped. T rises with level (1,597 → 2,958).
  - ρ(T, level) = 0.45 and ρ(T, question length) = 0.53.
  - Plain ρ(P, T) = 0.49, about the same as N_len's 0.51. The preregistered partial ρ is +0.02 [−0.41, 0.47] (40 items; inconclusive).
- **Push (10 Check D problems, 144 chains each).**
  - A one-gap push of d changes KL as much as a random direction of the same norm: median KL 13.6 vs 12.1.
  - The grid c ∈ {0.25 … 16} bracketed every push in every item: d, the controls, k = 0.5/2 (KL 1–103) and the authors' intervention (KL 12–132).
  - **One-gap d against Gaussian pushes of matched KL:** +d −54 tokens [−267, +195] (fail; SESOI 220), −d −93 [−310, +68] (inconclusive).
  - **The authors' intervention at λ = 0.2, against matched-KL Gaussian pushes:** +129 [−20, +323] and −163 [−352, +3]. Both are in the claimed direction but not shown.
  - The oracle failed Check D for −d and was inconclusive for +d. With only 10 items, none of this is decisive. It shows only that the pipeline runs and the grid brackets.
- **Check E (2 problems).** Code check only. Once a pushed chain diverges, it behaves like a fresh sample, so all rows of an item share the same offset from its clean chain; the matched-KL references share it too.
- **Pilot runs stopped and re-run** (all their outputs were discarded or kept only as superseded files):
  - The first push run (15 items, 236 chains each) ran out of memory under the first engine and was stopped. It was re-run with 10 items and 144 chains each (7 norms, 4 permuted directions).
  - The first Check E run started on the slow engine and was stopped. It was re-run on 2 items.
  - The clean chains and Check 0a of the pilot ran on the first (lockstep) engine. Their values are valid; only their timing is not used.
  - Timing probes at cap 8,192 were replaced by probes at 4,096, once 8,192 proved infeasible for the 8B models.
  - The first λ sweep crashed with an out-of-memory error while admitting rows: allocator fragmentation, and an allocation path the fallback did not cover. Both were fixed (§5) and the sweep was re-run.

## 8. Main runs
The job lines are in `code/q_d6main.txt`, appended to DQ's `jobs.txt` after the queued D4 main jobs, home model first. Per model they run, in order:
1. `cal`;
2. `clean` (skipped if the Check 0a gate fails);
3. `push` (likewise);
4. `analyze_d6`;
5. `pushE auto` (only for signs that pass Check D);
6. `analyze_d6` again.

The estimated total, excluding Check E, is about 67 GPU-hours (19.5 + 16.2 + 31.6, upper bounds), or about 37 if Qwen3-8B fails its Check 0a gate (its calibration alone is about 1 h).

## 9. Deviations and decisions (numbered D6-n, so they do not collide with other directions' Stage 2b files)
1. **D6-1** Their calibration template (with its backspace and trailing spaces) is used for every prompt, including the steering checks, which their evaluation script indented by four spaces (§2).
2. **D6-2** Layer: their code's single-layer index, not the "highest averaged similarity" layer named in the paper's text (§2).
3. **D6-3** Their intervention (every layer, every position, ±λ_a) is the secondary Check D push, not a single-layer push at their strength (§2).
4. **D6-4** Sampling: top-p 0.95 (their generation scripts), not their vLLM evaluation's 1.0 (§3).
5. **D6-5** Cap of 4,096 thinking tokens (theirs: 16,384); capped chains counted at the cap (§3).
6. **D6-6** Reference and exploratory push rows use one sample each, not T's 4-sample mean (Stage 1 §3), for cost. Every reference curve averages 8–12 directions (§5).
7. **D6-7** The long-generation reduction is applied per model: R1-Distill-Llama-8B and Qwen3-8B reduced, R1-Distill-Qwen-7B full. Qwen3-8B remains above 24 GPU-hours (§5).
8. **D6-8** Check E is generated only for signs that pass Check D, after the analysis (as Stage 1 §5 states). D1–D5 generated E rows up front (§5).
9. **D6-9** Models that fail the Check 0a gate skip the later stages (§5).
10. **D6-10** D6 needs sampled long generations: its own engine (`p6lib_d6.py`) and runner (`run_d6.py`, `analyze_d6.py`), not `run.py`'s greedy path (§5; `INTEGRATION_D6.md`).
11. **D6-11** Check 0a gate defined from the authors' headline and the drive SESOI. For Qwen3, λ_a is set by a rule fixed before the pilot's λ sweep was read (§2, §6).
12. **D6-12** MATH test rows fetched through the datasets-server API as JSON (§4).
13. **D6-13** Pilot changes: runs stopped and re-run, and the pilot Check D/E item counts and grid trimmed (§7).
