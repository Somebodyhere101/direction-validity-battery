# Does the Direction Track the Process?

Code, preregistrations and results for **"Does the Direction Track the Process? A Preregistered Validity Battery for Published Hidden-State Directions"** (Sam Ramdan, October 2026).

- Paper: [`paper/direction-validity-battery.pdf`](paper/direction-validity-battery.pdf) (version 3) · Zenodo: [version 3, doi:10.5281/zenodo.23532066](https://doi.org/10.5281/zenodo.23532066); [all versions, doi:10.5281/zenodo.23450399](https://doi.org/10.5281/zenodo.23450399); version 2 is [doi:10.5281/zenodo.23496934](https://doi.org/10.5281/zenodo.23496934) and version 1 is [doi:10.5281/zenodo.23450400](https://doi.org/10.5281/zenodo.23450400)
- Per-item results: `results-per-item-part01.zip` … `part10.zip` in the Zenodo record (kept out of git because of their size; unzip all ten into one folder)
- Earlier papers by the author: the compute–retrieve axis ([all versions](https://doi.org/10.5281/zenodo.19444314); [version 4, with the preregistered causal test](https://doi.org/10.5281/zenodo.23374358)), [A Compounding Fingerprint](https://doi.org/10.5281/zenodo.23318679) and [the injected-thought study](https://doi.org/10.5281/zenodo.23389066)

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
| oracle | Does a direction built from the target itself read it at that layer, and does a one-gap push along it move it? (A null push does not show that no direction can.) |

It was applied to nine published directions on five open 7–8B models (six configurations: Qwen3-8B with thinking off and on):
- **controls:** refusal and truth;
- **known-answer case:** the compute–retrieve axis;
- **process directions:** LiReFs, backtracking, reasoning strength, known-entity awareness, the sycophancy persona vector and true thinking.

## What it finds

- **The controls pass:** refusal and truth are VALID DRIVER overall.
- **No process direction receives a valid verdict (0 of 6).**
  - Four are TARGET NOT LINEAR: the oracle's one-gap push at the published layer does not pass Check D. That is a null for the oracle's push, not proof that the target cannot be pushed: the oracle is a reading direction, and in at least three runs the published direction moves its target where the oracle does not.
  - Two are NOT REPRODUCED overall: known-entity awareness (a difference-of-means rebuild on non-home chat models) and reasoning strength (reproduced only in its home model). LiReFs passes a reproduction gate that was set on class separation after its pilot; under Stage 1's gate on the authors' headline effect it would be NOT REPRODUCED too.
- **Read but not pushed at the preregistered strength.** The oracle reads the process in 17 of 19 process runs, but its one-gap push passes Check D in none of them, against 6 of 6 for the controls.
  - Exploratory pushes (0.5 to 2 gaps in every run; a post hoc sweep to 4 and 8 gaps for D3, D4, D4G and D8 only) separate the targets only in part. Answer choices (hallucinating, giving in to the user, following a perturbed reasoning step) move along the oracle in 5 of 9 runs. Settling depth and chain-of-thought necessity move in 1 of the 8 runs evaluable at four or eight gaps. Backtracking was pushed along the oracle only to two gaps.
  - The published directions move their own targets in 9 of 19 process runs at some strength, backtracking among them at its authors' 7.7 gaps. LiReFs never moves settling depth or chain-of-thought necessity, and the sycophancy vector never moves giving in.
- **Behind the verdicts:**
  - Known-entity awareness tracks its process and beats both label-free directions in all three models; true thinking tracks in all three and beats both in two.
  - LiReFs moves reply format rather than computation.
  - The sycophancy vector behaves like agreement with the user rather than giving in.
  - Reasoning strength, where it reproduces, is nearly the within-class first principal component of its calibration prompts.
- **Hypotheses:** H1 supported; H2 not supported; H3 not supported (4 of 9 directions separate their prompts at AUC ≥ 0.80 in weight-randomised models); H4 supported (norm-matched baselines overstate effects by 0.25 of a smallest effect of interest [0.07, 0.49] on average, but the median is 0.01 and two of 23 pairs carry most of the mean); H5 supported; H6 not supported (one NUISANCE); H7 partly.

All claims are about these models, layers and push strengths. Intervals are 95% bootstrap intervals.

## Versions

- **Version 3 (21 October 2026, doi:10.5281/zenodo.23532066).** Corrections after a pre-submission review. Version 3 corrects the attribution in versions 1 and 2, which stated that the battery idea was the author's: it was proposed by the AI system (see "Use of AI assistance"). The model count is corrected (five open 7–8B models, six configurations; version 2 said eight). The exploratory claims are restated for what was tested: the push-strength table now shows the published direction beside the oracle, marks cells with fewer than half their items evaluable, and says that the 4- and 8-gap sweep is post hoc. The paper now says what a null oracle push does and does not show, that LiReFs' reproduction gate was redefined after its pilot, and corrects several smaller misstatements against the per-run summaries. It adds a figure, a code-and-data statement, an ORCID, a first-page note on AI assistance and a shorter abstract. No preregistered verdict or result value changed. The code and results are unchanged; `DEVIATIONS_LOG.md` gains a dated correction entry.
- **Version 2 (20 October 2026).** The related work, the discussion and the novelty statement in the introduction are revised. The paper now positions itself against 2026 studies that report the same gap between reading a target and pushing it, the closest being Xun's *Lagged Coupling* (arXiv:2609.01048). The read/push dissociation itself is not claimed as new. What the paper claims is the battery: per-item process targets, the oracle as a diagnostic, label-free competitors, weight-randomised calibration, and preregistered breadth. No result, number or verdict changed, and the code and results are unchanged.
- **Version 1 (18 October 2026).** First release.

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
| `D9_MAIN_JOBS.txt` | D9's main-run job list as queued on the GPU machine |
| `code/` | Everything that was run (see below) |
| `code/data/` | Frozen item pools. See [`DATA.md`](DATA.md) for what is included and how the rest is rebuilt |
| `results/` | Per-run summaries (`results/<direction>/<model>_main_summary.json`), the cross-direction aggregate, the exploratory strength analyses, and the results page (`page.html`) |
| `paper/` | The PDF and its LaTeX source |

Direction keys: `d1` refusal, `d2` truth, `d3` compute–retrieve axis, `d4` LiReFs on D3's items, `d4g` LiReFs on GSM8K, `d5` backtracking, `d6` reasoning strength, `d7` known-entity awareness, `d8` sycophancy persona vector, `d9` true thinking. Model keys: `q25-7b` Qwen2.5-7B-Instruct, `q3-8b` Qwen3-8B, `olmo-7b` OLMo-2-1124-7B-Instruct, `r1-qwen-7b` and `r1-llama-8b` the DeepSeek-R1 distills.

## Code

- `p6lib.py`: the model wrapper (steering hooks, re-imposition for Check E, generation, teacher-forced KL) and the control directions.
- `specs.py`, `spec_d4.py` … `spec_d9.py`: one class per direction. Each holds the authors' recipe, its items, prompts, reading position and outcome.
- `run.py <dir> <model> <cal|clean|push> [main|pilot]`: calibration (direction, N_len, N_pc1, N_perm, Checks 0a, 0b, B), clean outcomes (Check A, oracle), and pushes (Checks C, D, E). D6 runs from `run_d6.py` with its sampled-generation engine `p6lib_d6.py`. D7 and D9 have extra stages in `run_d7.py`, `d9_clean.py`.
- `analyze.py <dir> <model> [main]`: the preregistered checks and verdicts, giving `res/<dir>/<model>_main_summary.json` (`analyze_d6.py` for D6). `aggregate.py` computes H1–H7 across directions. `build_page.py` builds the results page.
- Extraction and judges: `d8_extract.py` (persona-vector extraction with a local judge), `judge.py` (refusal), `judge_bt.py` (backtracking).
- Exploratory, not part of any verdict: `explore_oracle_k.py` (matched-KL effects at every push strength present in a run; Stage 1 lists 0.5 and 2 gaps as exploratory strengths) and `oracle_sweep.py` (post hoc: oracle and direction pushes at 4 and 8 gaps for D3, D4, D4G and D8, with an extended random grid).

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

An AI system (Claude, Anthropic), working under the author's direction, reviewed the literature, wrote the preregistrations and code, ran the experiments and analyses, and drafted the paper. The preregistrations record that the author had not reviewed their details line by line when they were frozen. The idea of packaging the checks from the author's earlier studies as a validity battery and applying it to published directions was proposed by the AI system in reply to the author's open-ended question about where a significant advance might lie. The author decided to run the study, delegated the preregistration and design choices to the AI system, provided the computing hardware, and released it. Earlier versions of this paper stated that the battery idea was the author's; that was incorrect and is corrected here. The author did not independently check the code, analyses or technical claims. The author directed the work and is responsible for its content. No direction vectors are released.

## Licence

Code: MIT (`LICENSE`). Included data keep their original licences (`DATA.md`).
