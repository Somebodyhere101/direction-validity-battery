# Does the Direction Track the Process?

Code, preregistrations and results for **"Does the Direction Track the Process? A Preregistered Validity Battery for Published Hidden-State Directions"** (Sam Ramdan, October 2026).

- Paper: [`paper/direction-validity-battery.pdf`](paper/direction-validity-battery.pdf)
- Per-item results: `results-per-item-part01.zip` … `part10.zip` in the Zenodo record (kept out of git because of their size; unzip all ten into one folder)
- Earlier papers in the series: the compute–retrieve axis ([all versions](https://doi.org/10.5281/zenodo.19444314); [version 2 with the preregistered causal test](https://doi.org/10.5281/zenodo.23374358)) and [A Compounding Fingerprint](https://doi.org/10.5281/zenodo.23318679)

## What the paper does

A published direction in a model's hidden states is usually defended by two facts: it separates two sets of prompts, and pushing along it changes the output. The battery asks more of it. Each check has a preregistered pass criterion and a smallest effect of interest.

| check | question |
|---|---|
| 0a | Does the direction rebuilt with its authors' recipe reproduce their headline effect? (gate) |
| 0b | How well does the same recipe separate the prompts in a model with randomised weights? |
| A | Does the projection follow a per-item target measured from what the model actually did? |
| B | Does the label separation survive stimuli matched for template, length and vocabulary? |
| C | Does it beat two directions built from the same calibration prompts without their labels? |
| D | Does a push steer the target beyond random pushes that change the output by the same KL? |
| E | Is the effect handed on to later layers, or does it act as an output bias? |
| oracle | Can a direction built from the target itself read and push it at that layer at all? |

It was applied to nine published directions on eight open models (27 direction × model pairs):
- **controls:** refusal and truth;
- **known-answer case:** the compute–retrieve axis;
- **process directions:** LiReFs, backtracking, reasoning strength, known-entity awareness, the sycophancy persona vector and true thinking.

## What it finds

- **The controls pass:** refusal and truth are VALID DRIVER overall.
- **No process direction receives a valid verdict (0 of 6).**
  - Four are TARGET NOT LINEAR: even the oracle cannot push the target with a one-gap push at the published layer.
  - Two are NOT REPRODUCED: their authors' headline steering effect does not appear in at least two of three models.
- **Readable but not pushable at the preregistered strength.** The oracle reads the process in 17 of 19 process runs but drives it in none, against 6 of 6 for the controls.
  - Exploratory pushes of up to eight gaps separate the targets. Choices of answer (hallucinating, giving in to the user, following a perturbed reasoning step) become pushable along the oracle at two to eight gaps, in 6 of 9 runs. The course of the computation (settling depth, chain-of-thought necessity, backtracking) moves in 1 of 12 runs. The published directions themselves move their target in 1 of the 12 runs tested at four and eight gaps.
- **Behind the verdicts:**
  - Known-entity awareness and true thinking track their processes in every model and beat both label-free directions.
  - LiReFs moves reply format rather than computation.
  - The sycophancy vector behaves like agreement with the user rather than giving in.
  - Reasoning strength, where it reproduces, is a length direction.
- **Hypotheses:** H1 supported; H2 not supported; H3 not supported (4 of 9 directions separate their prompts at AUC ≥ 0.80 in weight-randomised models); H4 supported (norm-matched baselines overstate effects by 0.25 of a smallest effect of interest [0.07, 0.49]); H5 supported; H6 not supported (one NUISANCE); H7 partly.

All claims are about these models, layers and push strengths. Intervals are 95% bootstrap intervals.

## Layout

| path | contents |
|---|---|
| `PREREGISTRATION.md` | Stage 1: the battery, pass criteria, verdict rules, process targets, SESOIs and hypotheses, frozen 2026-10-14 before any code |
| `PREREGISTRATION_STAGE2A.md`, `PREREGISTRATION_STAGE2B_D*.md` | Stage 2: per-direction settings, frozen after pilots on separate items and before each main run |
| `PREREG_SHA256SUMS` | SHA-256 and freeze time of every frozen file. Paths are relative to this folder |
| `DEVIATIONS_LOG.md` | Every departure from the frozen plan, with its reason |
| `INTEGRATION_D6.md`, `_D7`, `_D9` | Notes from the three directions that were built in parallel and run from their own folders |
| `code_stage2a_frozen_nodata.zip`, `dq_stage_d9/` | The frozen Stage 2a code (without the two withheld data files) and D9's frozen copies of the shared files |
| `LITERATURE.md` | The literature review and novelty assessment written before Stage 1 |
| `code/` | Everything that was run (see below) |
| `code/data/` | Frozen item pools. See [`DATA.md`](DATA.md) for what is included and how the rest is rebuilt |
| `results/` | Per-run summaries (`results/<direction>/<model>_main_summary.json`), the cross-direction aggregate, the exploratory strength analyses, and the results page (`page.html`) |
| `paper/` | The PDF and its LaTeX source |

Direction keys: `d1` refusal, `d2` truth, `d3` compute–retrieve axis, `d4` LiReFs on paper 4's items, `d4g` LiReFs on GSM8K, `d5` backtracking, `d6` reasoning strength, `d7` known-entity awareness, `d8` sycophancy persona vector, `d9` true thinking. Model keys: `q25-7b` Qwen2.5-7B-Instruct, `q3-8b` Qwen3-8B, `olmo-7b` OLMo-2-1124-7B-Instruct, `r1-qwen-7b` and `r1-llama-8b` the DeepSeek-R1 distills.

## Code

- `p6lib.py`: the model wrapper (steering hooks, re-imposition for Check E, generation, teacher-forced KL) and the control directions.
- `specs.py`, `spec_d4.py` … `spec_d9.py`: one class per direction. Each holds the authors' recipe, its items, prompts, reading position and outcome.
- `run.py <dir> <model> <cal|clean|push> [main|pilot]`: calibration (direction, N_len, N_pc1, N_perm, Checks 0a, 0b, B), clean outcomes (Check A, oracle), and pushes (Checks C, D, E). D6 runs from `run_d6.py` with its sampled-generation engine `p6lib_d6.py`. D7 and D9 have extra stages in `run_d7.py`, `d9_clean.py`.
- `analyze.py <dir> <model> [main]`: the preregistered checks and verdicts, giving `res/<dir>/<model>_main_summary.json` (`analyze_d6.py` for D6). `aggregate.py` computes H1–H7 across directions. `build_page.py` builds the results page.
- Extraction and judges: `d8_extract.py` (persona-vector extraction with a local judge), `judge.py` (refusal), `judge_bt.py` (backtracking).
- Exploratory, not preregistered: `explore_oracle_k.py` (matched-KL effects at every push strength present in a run) and `oracle_sweep.py` (oracle and direction pushes at 4 and 8 gaps, with an extended random grid).

The code was run from separate copies of `code/` on the GPU machine, one per group of directions. `INTEGRATION_D*.md` lists the few lines in which those copies differ.

## Verifying the preregistration

From the repository root:

```
grep -v "^#\|^20" PREREG_SHA256SUMS | sha256sum -c
```

Most frozen files verify. Expected exceptions:
- **Edited after an earlier freeze:** `code/run.py`, `code/analyze.py`, `code/p6lib.py`, `code/stage2_settings.json` and `DEVIATIONS_LOG.md` were extended for later directions, and each change is recorded in `DEVIATIONS_LOG.md`.
  - These files appear under several freezes, so only the line from the latest freeze can match.
  - The Stage 2a versions are in `code_stage2a_frozen_nodata.zip`.
  - D9's frozen copies are in `dq_stage_d9/`.
- **Not redistributed:** the data files listed in `DATA.md` and the third-party sources.
- **Withheld because they contain model-written text that was not reviewed for release:** two pilot text files (`res/d5/pilot_bt_validation.jsonl`, `res/d7/pilot_blind_d7.jsonl`) and D6's λ-sweep pilot output.
  - The blind labels made from the two text files are included.

## Environment

```
pip install -r requirements.txt
```

The runs used Python 3.12.10, torch 2.14.1 (CUDA 13.0), transformers 5.19.0 and one RTX 5090 (32 GB), on Windows 11. Models run in bf16 with their chat templates. Generation is greedy, except for D6, D8's extraction and D4G's chain-of-thought target, which sample with fixed seeds. Third-party repositories go in `third_party/` at the commits listed in `DATA.md` and the Stage 2 documents.

## Use of AI assistance

An AI system (Claude, Anthropic), working under the author's direction, reviewed the literature, wrote the preregistrations and code, ran the experiments and analyses, and drafted the paper. The author directed the work and is responsible for its content.

## Licence

Code: MIT (`LICENSE`). Included data keep their original licences (`DATA.md`).
