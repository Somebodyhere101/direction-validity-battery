# D6 (reasoning strength): integration notes for the main session

D6 was built in a parallel session that did not edit any shared file (`run.py`, `analyze.py`, `p6lib.py`, `specs.py`, `spec_d5.py`, `stage2_settings.json`). Everything D6 needs lives in new files and runs from its own DQ folder, `C:\Users\CAAB\jspace\paper6_d6`. That folder also holds unchanged copies of `p6lib.py`, `specs.py`, `data/items.json` and `data/calibration.json`, because `specs.py` loads the last two at import.

**Nothing has to be merged for D6 to run.**

## New files (all under `paper6/code/`)
| file | role |
|---|---|
| `spec_d6.py` | `class D6(Spec)`, D6's version of `spec_d5.py`. It holds the authors' prompt, layer rule and calibration (vectors at every layer); the level-wise `N_len`, `N_pc1` and `N_perm`; the items; Check B pairing; λ_a per model; and the Check 0a gate (`gate_d6`). |
| `make_items_d6.py` → `data/items_d6.json`, `data/calibration_d6.json` | The item pools (MATH test, and MATH-500 for Check 0a) and the authors' calibration file. |
| `p6lib_d6.py` | The sampled-generation engine (described below). |
| `run_d6.py` | D6's runner, adapted from `run.py`: stages `cal`, `clean`, `push`, `pushE`, plus the pilot-only `timing`, `lamsweep` and `selftest`. |
| `analyze_d6.py` | D6's copy of `analyze.py`: partial ρ with level and subject dummies, per-row dT, relative SESOI, conditional Check E, and gate-skipped models. Output `res/d6/<key>_main_summary.json` has the same top-level keys as the other directions, plus `cap`, `design`, `grid_bracket` and `run_seconds`. |
| `estimate_d6.py` | The run-time estimate. |
| `config_d6_pilot.json`, `config_d6_main.json`, `stage2_settings_d6.json` | Grids, cap and sample counts (main: the full or reduced design per model), and the Check 0a gate. |
| `q_d6main.txt` | The main job lines (appended to DQ `jobs.txt`). |

## If D6 is later folded into the shared runner
These changes would be needed. None is required now.
1. **`run.py`.**
   - Add `SPECS["d6"] = D6`.
   - D6 cannot use the greedy `Model.generate` path: its target needs 4 sampled chains per item and long generations. Its `clean`, `push` and `pushE` branches would have to be moved over from `run_d6.py`.
   - `p6lib.n_len`, `n_pc1` and `n_perm` assume two classes. D6 uses the five levels (`spec_d6.n_len_levels`, `n_pc1_levels`, `n_perm_levels`).
2. **`analyze.py`.**
   - `T` for d6 is `rec["T"]`, and each push row carries its own `dT`, computed against the same samples' clean chains.
   - Check A's primary statistic is always the partial ρ with dummies (`covariates()` in `analyze_d6.py`).
   - `DRIVE` is `[("T", 0.10 × mean clean T on the D items)]`.
   - Check E reads `<key>_<phase>_pushE.jsonl`.
   - The gate comes from `stage2_settings_d6.json`, through `spec_d6.gate_d6`.
3. **`aggregate.py` / `build_page.py`.** Add `d6` to the directions read, from `paper6_d6/res/d6/*_main_summary.json`. A model that failed the gate has a short summary with `verdict = "NOT REPRODUCED"` and no Check A–E.

## Sampled-generation engine (`p6lib_d6.py`)
- **Sampling.** Temperature 0.6, top-p 0.95, as Gumbel-max inside the nucleus.
  - The noise is a hash of (problem id, sample index, token step, token id) (`gumbel_rows`). A chain is therefore the same whenever and in whatever batch it runs, up to bf16 batch effects.
  - Pushed and clean chains of the same sample share their random numbers.
- **`stream_generate`** (clean, Check 0a, Check D): continuous batching.
  - Up to `slots` chains run at once. Finished chains leave the batch, and queued chains are admitted in groups.
  - Each admitted group runs its prompt pass on its own (with its pushes from the last prompt token, or from position 0 for the authors' every-position steering). It is then merged into the shared KV cache, left-padded to a common column layout. Leading columns no chain uses are dropped.
  - On an out-of-memory step, the quarter of chains with the least work done is re-queued and the slot count lowered.
- **`GrowLayer`.** A `DynamicLayer` whose keys and values live in a buffer that grows by 256 columns and is written in place. The stock `DynamicLayer` concatenates, copying the whole cache at every token.
- **`p6_decode_attention`.** Registered with transformers' `AttentionInterface` as `"p6decode"`, with SDPA's mask format.
  - Single-token decoding with grouped-query heads runs as one batched matmul. SDPA expands keys and values to every query head whenever a padding mask is present.
  - Prompt passes and teacher-forced passes go to SDPA unchanged.
  - Same argmax as SDPA in the self-test; largest logit difference 0.375 at a logit scale of 34.
- **Allocator.** `run_d6.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch loads. It works on DQ (Windows). In a cache-like allocation test it cut reserved-but-unused memory from 6.6 GB to 0.8 GB, and without it the λ-sweep pilot crashed on fragmentation. The other directions' long runs might benefit too.
- **`sample_generate`** (Check E): lockstep batches with the clean twin in row 0, which `Model.ops["reimp"]` reads.
- **`tf_kl`.** `Model.tf_kl` with separate push and KL start positions.
- **Throughput at cap 4,096** (tokens per second, slots):
  - R1-Distill-Qwen-7B: 966 (36);
  - R1-Distill-Llama-8B: 470 (14);
  - Qwen3-8B: 336 (12);
  - Qwen3-1.7B: 1,095 (38), against 356 for the first lockstep engine.

## Job flow on DQ (`q_d6main.txt`)
Per model, in this order:
1. `cal`;
2. `clean`;
3. `push`;
4. `analyze_d6`;
5. `pushE auto`;
6. `analyze_d6` again.

Notes:
- `clean`, `push` and `pushE` exit at once if the model failed the Check 0a gate (`P6_IGNORE_GATE=1` overrides this).
- `pushE auto` reads the summary and runs Check E only for the signs that passed Check D.
