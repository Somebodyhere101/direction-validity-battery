# D7 integration notes (for the main session)

D7 (known-entity awareness) was built in a parallel session. It did not edit any shared file. It runs from its own DQ folder, `C:\Users\CAAB\jspace\paper6_d7`. That folder holds:
- the frozen D5 deployment's shared code (`jspace/paper6b`; the hashes match the Stage 2b D5 lines of `PREREG_SHA256SUMS`);
- the D7 files listed below;
- the three changed copies in `code/d7_deploy/`.

## New files (D7 only)
| file | role |
|---|---|
| `code/spec_d7.py` | class `D7(Spec)`, with the interface of `spec_d5.D5`: `calibrate`, `cal_acts`, `check0a`, `items`, `prompt_ids`, `signs`, `outcome`, `B_set`; plus the authors' correctness and refusal functions |
| `code/run_d7.py` | the D7-only stages: `label` (per-model known/unknown labels; must run before `cal`) and `lens` (secondary target; runs after `cal`) |
| `code/make_items_d7.py` | builds `code/data/items_d7.json` and `code/data/sitelinks_d7.json` (already built and frozen; do not rerun, because Wikidata sitelink counts change) |
| `code/blind_d7.py` | blind validation sample and scoring for the refusal phrase list |
| `code/data/items_d7.json`, `code/data/sitelinks_d7.json` | frozen item pools and the popularity covariate |
| `third_party/sae_entities/` (laptop), `jspace/third_party/sae_entities/` (DQ: data subset only) | the authors' release, read only |
| `jspace/paper6_d7/vendor/` (DQ only) | `thefuzz` 0.22.1 and `rapidfuzz` 3.14.6, installed with `uv pip install --target` (the shared venv has neither). `spec_d7.py` appends `code/vendor` to `sys.path` if it exists, so a merged folder needs the same `vendor/` directory or `thefuzz` in the venv |

## Shared-file changes needed to run D7 from a merged folder
All are additive; nothing else changes. The exact diffs against the frozen D5 versions are in `code/d7_deploy/*.diff`.

1. **`run.py`**: two lines (`code/d7_deploy/run_d7.diff`).
   - The registration, after `SPECS["d5"] = D5`:
     ```python
     from spec_d7 import D7; SPECS["d7"] = D7  # noqa: E402,E702  (Stage 2b D7)
     ```
   - A generic oracle hook in the push stage, which leaves every other direction unchanged. It replaces `o = oracle(Ho, To, binary=spec.binary_T or DIR == "d5")` with:
     ```python
     o = spec.oracle_dir(Ho, To, orc) if hasattr(spec, "oracle_dir") else oracle(Ho, To, binary=spec.binary_T or DIR == "d5")
     ```
     D7 builds its oracle within entity type (Stage 2b D7 §3, deviation D7-6).

   Nothing else is needed:
   - D7 uses the mode name `direct`, so `oracle_T` falls through to `rec["direct"]["D"]` (D7's `D` = correct).
   - The clean and push stages record the lens top-1 per step. That gives E3, the settling layer of the first answer token.
   - The D7 Check D items are appended to the `A` list returned by `D7.items()` only so that `run.py` can find them. `analyze.py` uses the frozen Check A pool alone.
2. **`analyze.py`**: ten additive edits (`code/d7_deploy/analyze_d7.diff`):
   - load the D7 items and the model's labels, and add the D pool to `meta` as `Dpool`;
   - `_sign7` and the `T_row` branch for the `drv` / `ans` targets;
   - `DRIVE` for d7;
   - the Check A covariates: prompt length, log popularity, type dummies;
   - the Check 0a gate for d7: the mean over types of the held-out separation AUC, and the steering rise beyond random;
   - an `extra7` block written as `summary["d7"]`: label composition of Check A, the middle items, per-type AUCs, the lens target, and Check B's popularity AUC;
   - `drive_target` / `drive_sesoi` in the summary, as in the logged 01:06 change.

   If the current shared `analyze.py` already has the D4/D4G additions, the D7 hunks apply next to them: each hunk anchors on D5-era lines that are still present.
3. **`stage2_settings.json`**: a `"d7"` entry: `{"sep_mean_auc_min": 0.7, "refusal_rise_min": 0.2}`.
4. **Aggregation and page** (`aggregate.py`, `build_page.py`): D7 summaries have the standard fields, plus `summary["d7"]`. D7's drive target is `drv` with SESOI 0.10, and the push signs depend on the label (−d on known, +d on unknown entities), as for D1.

## Order of jobs per model (main)
`run_d7.py label` → `run.py d7 cal` → `run.py d7 clean` → `run.py d7 push` → `run_d7.py lens` → `analyze.py d7` (CPU).

The ready-made job lines are in `code/jobs_d7_main.txt`. They are written, not queued.
