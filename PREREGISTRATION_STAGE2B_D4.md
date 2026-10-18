# Paper 6, Stage 2b (part 2): settings for D4 (LiReFs, reasoning–memorisation features) after its pilot

> Frozen 2026-10-15 12:12 AST, after the D4 pilot (Qwen3-1.7B, pilot items only) and before any main-panel D4 run. The AI assistant wrote it on the author's standing instruction (Stage 1 §0). Like Stage 2a, it sets only what Stage 1 §10 allows. Hashes are in `PREREG_SHA256SUMS`.

## 1. Models
The chat panel of Stage 2a §1, with the same revisions: Qwen2.5-7B-Instruct, Qwen3-8B with thinking off, and OLMo-2-1124-7B-Instruct. Thinking is off in the pilot model too (Qwen3-1.7B).

## 2. Direction (Hong et al.'s recipe, `github.com/yihuaihong/Linear_Reasoning_Features` @4f95e7e)
- **Calibration data:** their 3,000 MMLU-Pro questions (`mmlu-pro-3000samples.json`); reasoning means `memory_reason_score > 0.5` (1,379 questions).
- **Prompt and reading position:** their hidden-state cache prompt `Q: {question}\nA: `, raw text with no chat template, read at the last token.
- **Candidate directions:** one per layer, mean(reasoning) − mean(memory). Their `hidden_states` index l is the output of block l − 1.
- **Layer rule (theirs):** for every layer l > 2, scale the component along that layer's direction by 1.1 at every block input and every attention and MLP output (their `set_act_modify_hooks`, scale 0.1). Then run 5-shot chain-of-thought MMLU-Pro on 200 of the 3,000 questions (200 new tokens, their prompt format and answer parser, including the random guess when no answer is found), and keep the layer with the highest accuracy on the reasoning subset.
  - **Deviation 12 (fixed sample):** their script draws a new 200 for each layer. Here one fixed sample (`random.Random(8888)`) is used for every layer, so layers are compared on the same questions.
  - **Deviation 13 (few-shot source):** the 5-shot exemplars are rebuilt from MMLU-Pro's validation split. Their copy is in a format neither machine can read without `pyarrow`, so the identical 70 rows were fetched from the Hugging Face dataset viewer.
- **Battery reading position:**
  - **D4:** the last token of the chat-formatted prompt, which is D3's reading position, on the same items.
  - **D4G:** the last token of the chat prompt, before the assistant prefill.
- **N_len:** uses the token length of the raw calibration prompt.

## 3. Items and targets
- **D4:** paper 4's items and targets, exactly as D3 (Stage 2a §3, §5). Settling depth D (SESOI one layer) is the primary drive target; shown working W (10 points) is secondary. The preregistration's "D4 and D3 share items and targets" is taken literally, so the two published "reasoning versus recall" directions meet on identical items.
- **D4G:** 400 GSM8K test problems not used by D5 (same shuffle, rows 420 onward).
  - **Splits:** pilot 40, oracle 120, Check A 240, Check D 120, Check E 60.
  - **Tracking target:** chain-of-thought necessity, i.e. accuracy with chain of thought minus accuracy answering directly. Each is measured over 4 samples at temperature 0.7: chain of thought with the assistant prefill "Let's think step by step." and up to 384 tokens; direct with the prefill "The answer is " and 16 tokens.
  - **Drive target:** greedy direct-answer accuracy under the push, SESOI 10 points. **Deviation 14:** the preregistration names accuracy without saying which mode. Pushing during 384-token sampled chains on 170 rows per item would cost far more than the rest of D4, so the drive target is the direct answer.
- **Check B:** not constructed for D4 or D4G (as for D3: reasoning and memory questions differ in content by definition).

## 4. Check 0a gate
- **Gate:** the rebuilt direction separates held-out reasoning from memory questions (two-fold AUC on the calibration set) at **AUC ≥ 0.80**. This is the authors' first headline and the kind of reproduction check Stage 1 §5 itself gives as an example ("held-out probe accuracy").
- **Reported but not gated:** their causal headline, that the intervention improves reasoning accuracy. It is reported as the change in reasoning accuracy on 200 fresh questions with their intervention at the selected layer.
- **Why it is not the gate:**
  - In the pilot the sweep picked layer 8 on a gain of one question in 53 (28.3% → 30.2%, in-sample).
  - 23 of the 25 layers lowered or did not change accuracy.
  - On fresh questions the intervention lowered reasoning accuracy from 41.4% to 29.9%.
  - The causal question is what Check D tests with matched controls; a few-point accuracy effect measured on 200 items cannot serve as a rebuild check.
  - Their models were base models prompted with raw text; the panel models are chat models, which may matter.
  - **This decision was made after seeing the pilot and is reported as such (deviation 15).**

## 5. Grid, batching, run time
- **Grid:** as Stage 2a §4: k ∈ {0.5, 1, 2}; R = 12 × 7 norms; 8 N_perm × 7 norms; E1 and E2.
- **Batching:**
  - the sweep uses batches of 16 left-padded prompts (6 for OLMo-2, whose larger cache would not fit 16 five-shot prompts);
  - D4 pushes use the D3 settings (OLMo-2 in chunks of 96 rows);
  - `P6_MEM_FRACTION=0.85` throughout.
- **Pilot run time (1.7B):** calibration including the sweep 48 min; D4 clean and push 4 min; D4G clean and push 2 min.
- **Main estimate:** about 1.5–3 h per model for calibration (the sweep dominates) plus about 45 min for the battery, so about 7–10 GPU hours for D4 in total. That is under 24 per model, so no reduction is applied.

## 6. What the pilot showed (Qwen3-1.7B)
- **Calibration:** selected `hidden_states` layer 8 (block 7); calibration AUC 0.97; two-fold AUC 0.97. Weight-randomised model: 0.62.
- **Overlaps:** cos(d, N_len) = +0.10; cos(d, N_pc1) = +0.39.
- **D4 (60 items):** tracking of settling depth is inconclusive (ρ +0.11 [−0.18, +0.37]). The length direction raises the NUISANCE flag. A one-gap push leaves settling depth unchanged beyond matched random pushes. The oracle does not drive settling depth.
- **D4G (40 items):** tracking of chain-of-thought necessity is inconclusive (ρ +0.18 [−0.12, +0.46]), and the push leaves direct-answer accuracy unchanged.
- **Bracketing:** every push fell inside its random KL bracket (no exclusions).
