# Paper 6: deviations log after freezing

Events after a Stage 2 freeze that the frozen files do not record. They are listed in order, in AST (DQ clock = AST − 2 h).

## 2026-10-15 00:54 AST: performance-only patch to the frozen Stage 2a code (`p6lib.py`)
- **Problem:** D1's push run on OLMo-2-7B filled the 32 GB GPU. OLMo has full multi-head attention, so its cache is large, and the code also computed logits at every prompt position plus an unused cache in the teacher-forced pass. Windows then paged GPU memory and the run slowed to ~90 s per item, against ~4 s for Qwen3-8B.
- **Fix:** the patch (`code/p6lib_stage2a_perf_patch.diff`, six edits) adds:
  - `logits_to_keep=1` (only the last position's logits, which are all the code used);
  - `use_cache=False` in the teacher-forced pass;
  - freeing the cache after generation and `torch.cuda.empty_cache()`;
  - an optional cap on PyTorch's share of GPU memory (`P6_MEM_FRACTION=0.85`).
- **Effect on results:** none of these changes what is computed, apart from floating-point reduction order in the final-layer projection.
- **Which runs used which code:**
  - unpatched: D1 calibration, clean and push for Qwen2.5-7B-Instruct and Qwen3-8B, and D1 calibration and clean for OLMo-2;
  - patched: D1's OLMo-2 push (restarted from scratch, so every OLMo push item uses the same code), the judge passes, and all D2 and D3 runs.
- **Kept for the record:** the frozen file is kept as `jspace/paper6/p6lib_stage2a.py` on DQ and in `code_stage2a_frozen.zip`.
- **Superseded output:** the partial unpatched OLMo push (`olmo-7b_main_push_superseded.jsonl`, 1–2 items) is kept and not analysed.

## 2026-10-14 ~22:10–22:50 AST: D5 pilot process overlap
- **What happened:**
  - Stopping a D5 pilot run stopped only the venv launcher (`python.exe`); its interpreter (`python3.12.exe`) kept running.
  - For about 30 minutes it ran alongside D1's judge pass, D1's Qwen2.5-7B push and a second pilot attempt, slowing them.
- **Effect on results:**
  - D1 outputs are unaffected: separate processes and files.
  - The D5 pilot file written during the overlap was discarded. The analysed pilot is the later complete run.

## 2026-10-15 01:06 AST: `analyze.py` output fields
- Two fields, `drive_target` and `drive_sesoi`, were added to the summary JSON for the results page. No computation changed.
- The Stage 2b hash of `analyze.py` is for the version before this edit. Both versions are kept in this folder's history; the change is the single `summary = dict(...)` line.

## 2026-10-15 10:45 AST: D5 secondary judge on pushed outputs limited to the home model
- The secondary backtracking judge over every pushed D5 output would take about 3 GPU hours per model: 702,059 sentences for R1-Distill-Llama-8B alone.
- The judge failed its validation (κ = 0.64), so it is secondary only.
- To free the GPU for the pilots of D6, D7 and D9, the pass is stopped once the home model's file is written.
- Result: the judge on the pushed outputs is reported for R1-Distill-Llama-8B only. The judge on the clean chains (tracking) is complete for all three models.

## 2026-10-15 ~11:45 AST: the D5 judge pass over pushed outputs will be completed after all
- The secondary judge turned out to matter. Against its labels, the backtracking vector tracks judged backtracking in all three models (AUC 0.69 to 0.73, CIs above 0.66), while against the primary keyword labels it does not (AUC 0.47 to 0.54).
- In R1-Distill-Llama-8B, a one-gap push also moves judged backtracking beyond both random references.
- The judge pass over the pushed outputs of R1-Distill-Qwen-7B and Qwen3-8B will therefore be run when the GPU is free (about 6 GPU hours), rather than dropped as decided at 10:45.
- The preregistered primary measure stays the keyword list. The judge results are reported as secondary, next to it.

## 2026-10-15 ~21:05 AST: D4G pushes in chunks of 96 rows (batching only)
- **What happened:** the D4G push for Qwen2.5-7B-Instruct ran out of GPU memory after 66 of 120 Check D items. Each item's 189 rows were prefilled in one batch, and longer GSM8K prompts pushed the MLP's working memory over the limit. Stage 2b-D4 §5 had set 96-row chunks only for OLMo-2.
- **Change:** the D4G pushes for Qwen2.5-7B-Instruct and Qwen3-8B now run in chunks of 96 rows (`MAX_ROWS=96`), as OLMo-2's already did.
  - Chunking changes batch composition only, not what is computed. Row 0 of every chunk is the item's clean row.
  - Stage 1 asks for a fixed batch size within every comparison, so the Qwen2.5-7B-Instruct push was restarted from the first item. It was not resumed.
- **Superseded output:** the partial file is kept as `res/d4g/q25-7b_main_push_superseded_oom.jsonl` and is not analysed.
- **2026-10-16 ~03:00 AST, OLMo-2:** the D4G push for OLMo-2 ran out of memory at the same item (the 67th Check D item), even in 96-row chunks; OLMo-2's full multi-head attention cache is about four times Qwen's. It was restarted from the first item in chunks of 48 rows. The partial file is kept as `res/d4g/olmo-7b_main_push_superseded_oom.jsonl` and is not analysed. The D4 push for OLMo-2 on D3's items completed in 96-row chunks, as Stage 2b-D4 §5 sets.
- **2026-10-16 ~04:20 AST, D8:** the D8 push for Qwen3-8B ran out of memory after 70 of 200 Check D items (189 rows of a long ARC question in one batch). It was restarted from the first item in chunks of 96 rows, and the pending OLMo-2 push was set to chunks of 48 rather than 96. Qwen2.5-7B-Instruct's D8 push had already completed unchunked and is kept, since each model's push uses one setting throughout. The partial file is kept as `res/d8/q3-8b_main_push_superseded_oom.jsonl` and is not analysed.

## 2026-10-18: exploratory push-strength analyses (after all main runs; not part of any verdict)
- **Two analyses were added** under Stage 1 §9 (exploratory), after every preregistered verdict was final. They test whether TARGET NOT LINEAR holds at larger strengths.
  - `explore_oracle_k.py` reruns the preregistered Check D code on pushes at 0.5 and 2 gaps that every main run already contained.
  - `oracle_sweep.py` pushes d and the oracle at 4 and 8 gaps for D3, D4, D4G and D8, on the first 80 Check D items, against the run's own Gaussian directions with the random grid extended to 24 gaps.
- **Reference:** only the Gaussian reference is used; the permuted reference was not rerun.
- **Results:** the paper's §5.4 and `results/exploratory/`.

## 2026-10-20: corrections from a pre-submission review (paper version 3; nothing rerun or recomputed)
- **The 4- and 8-gap sweep is post hoc.** The 2026-10-18 entry above files both exploratory analyses under Stage 1 §9. Stage 1 lists ±0.5 g and ±2 g as exploratory strengths of Check D (§5), which covers `explore_oracle_k.py`, but neither §5 nor §9 lists a sweep to 4 and 8 gaps. `oracle_sweep.py` is therefore a post hoc analysis, added after every verdict was final. It covers D3, D4, D4G and D8 only; D5, D6, D7 and D9 were not swept, and no reason was recorded at the time.
- **Evaluable items.** Several cells lost most of their items, because pushed outputs could not be scored or their KL fell outside the random grid. Paper version 3 treats a cell with fewer than half of its items evaluable as not evaluable and lists every cell that lost items (its Appendix B).
- **D4 gate.** As Stage 2b-D4 §4 records, the Check 0a gate was set on separation after the pilot. Under Stage 1's gate on the authors' headline effect (their intervention improving reasoning accuracy), D4 would be NOT REPRODUCED in all three models. Paper version 3 says so.
- **FORMAT-MEDIATED.** Stage 1's format clamp was never evaluated, because no target was measured both with the format free and under a prefill.
- **Oracle failure in the code.** `analyze.py` counts the oracle as failing Check A or D whenever it does not pass. In every TARGET NOT LINEAR verdict the oracle's one-gap push failed or reversed in both signs against at least one reference, so no verdict depends on this. It does decide the counterfactual D7 verdicts that the paper reports for the case in which the gate had passed.
