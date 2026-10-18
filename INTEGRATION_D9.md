# D9 (true thinking): integration notes for the main session

D9 was built in a parallel session that did not edit any shared file. Its runs use the DQ folder `C:\Users\CAAB\jspace\paper6_d9`. That folder holds copies of the shared files as they were in `paper6/code` at 2026-10-15 ~11:20 AST, after D4/D4G/D8 were registered, with the changes below applied. `code/d9_tools/make_dq_copy.py` applies them by exact string replacement and stops if any target string has moved.

## New files (under `paper6/code/`)
| file | role |
|---|---|
| `spec_d9.py` | `class D9(Spec)` plus the chain, TTS and intervention machinery (see its docstring) |
| `d9_clean.py` | D9's clean stage, which replaces `run.py clean` for D9. It produces chains, TTS, Check A projections and the Check D unit choice. Writes `res/d9/<key>_<phase>_clean.jsonl` and `_clean_h.npz` in the layout that `run.py push` and `analyze.py` read |
| `make_items_d9.py` → `data/items_d9.json` | item pools: AMC calibration problems, plus MATH L1–3 integer-answer splits |
| `d9_tools/` | `make_dq_copy.py` (stages the DQ folder), `stage2_settings_d9.json` (Check 0a gate), `fetch_d9_data.py` (dataset download), `smoke_d9.py`, `timing_d9.py`, `envcheck_d9.py` |

## Changes needed in shared files
1. **`run.py`** needs two lines:
   - registration, after `SPECS["d8"] = D8`:
     `from spec_d9 import D9; SPECS["d9"] = D9  # noqa: E402,E702`
   - the per-sentence oracle branch in `push`: change `if DIR == "d5":` to `if DIR in ("d5", "d9"):`.
     - D9's clean records keep one record per step under `"sentences"` (`j`, `T`), and its hidden states are keyed `"<id>:<j>"`, as D5's are.
     - `binary=spec.binary_T or DIR == "d5"` stays as it is, so D9's oracle is the tercile split of the continuous TTS.
2. **`analyze.py`**, five additions:
   - `ITEMS[...]`: `d9=json.load(open(os.path.join(HERE, "data", "items_d9.json"), encoding="utf-8"))`.
   - `T_row`: `if DIR == "d9": return row["gen"]["F"]`. F = 1 when the early-exit answer differs from y*, i.e. it follows the perturbed step.
   - `DRIVE`: `elif DIR == "d9": DRIVE = [("F", 0.10)]; BINARY = False`. The tracking target (TTS) is continuous, so ρ is used.
   - Check A: a D5-like branch, `elif DIR == "d9":`. Units are the numeric steps of each Check A chain (`id = "<problem>:<j>"`). Covariates are `[pos, step_len]`: the prompt length up to the step's last token, and the step's own token length. The bootstrap groups are problems.
   - `check_D`: `if DIR == "d9" and (sign == 1) != (rec["rows"][0]["gen"]["F"] == 0): continue`. Every unit is pushed in both signs. +d is evaluated where the clean answer stays y* (Zhao et al.'s engagement test), and −d where it follows the perturbed step (disengagement test).
   - The Check 0a gate: `elif DIR == "d9":`, as set out in Stage 2b D9 §5.
3. **`stage2_settings.json`**: add `"d9": {"cos_released_min": 0.5, "flip_adv_min": 0.05}` under `check0a`. This is the same as `d9_tools/stage2_settings_d9.json`; see Stage 2b D9 §5.
4. **`p6lib.py`, `specs.py`, `spec_d5.py`, configs**: no change. D9 uses `config_main.json` / `config_pilot.json` as they are (see Stage 2b D9 §4 on bracketing).
5. **`aggregate.py` / `build_page.py`**: add `d9` to the directions read. D9's summary JSON has the same top-level keys as the other directions.

## How the stages run for D9
```
python run.py d9 <key> cal <phase>      # direction (AMC chains + TTS), N_len/N_pc1/N_perm, Check 0a (cosine + ET/DT), Check B, Check 0b
python d9_clean.py <key> <phase>        # Check A / oracle chains, TTS, projections, Check D units
python run.py d9 <key> push <phase>     # generic push stage (needs MAX_ROWS, see Stage 2b D9 §4)
python analyze.py d9 <key> <phase>
```
`run.py clean` must not be used for D9: its generic branch would call `spec.prompt_ids` on Check A problems, which have no unit.

The push stage needs a small `MAX_ROWS`, because its prompts carry a chain prefix of up to 2,048 tokens. Use 12 for r1-llama-8b and q3-8b, and 16 for r1-qwen-7b (Stage 2b D9 §4). Chain generation sizes its own batches and splits a batch that runs out of memory.

The DQ copy of the shared files (`dq_stage_d9/` in `paper6/`, deployed to `jspace/paper6_d9`) was taken before the shared files' later edits. Its hashes are in `PREREG_SHA256SUMS` under Stage 2b (D9).

## Files written under `res/d9/` (per model key)
- `<key>_cal_chains.jsonl`, `<key>_0a_chains.jsonl`, `<key>_B_chains.jsonl`: the calibration (AMC), Check 0a and Check B chains, with per-step scores. Calibration problems are the same in pilot and main, so these files are not split by phase.
- `<key>_<phase>_chains.jsonl`, `_clean.jsonl`, `_clean_h.npz`: the Check A and oracle chains, with the derived clean records.
- `<key>_<phase>_cal.pt|json`, `_dirs.npz`, `_push.jsonl`, `_summary.json`: as for other directions.
