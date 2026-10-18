# Data: what is included, and how to rebuild what is not

All item pools are frozen with fixed seeds and hashed in `PREREG_SHA256SUMS`. The `make_items*.py` scripts rebuild every file below from the original sources at the listed commits. They are deterministic, so a rebuild from the same sources should match those hashes byte for byte. Rebuilds were not re-verified for this release.

## Included in `code/data/`

| file | direction | built from (licence) |
|---|---|---|
| `items_d4g.json` | D4 (GSM8K items) | GSM8K test set (MIT) |
| `items_d5.json` | D5 | GSM8K test set (MIT) |
| `items_d6.json`, `calibration_d6.json` | D6 | MATH and MATH-500 (MIT). The calibration file holds the 7,500 MATH training problems as listed in Sheng et al.'s repository |
| `items_d7.json`, `sitelinks_d7.json` | D7 | Ferrando et al.'s entity lists, `javiferran/sae_entities` @5ec3696 (Apache-2.0); Wikidata sitelink counts fetched 2026-10-15 (CC0) |
| `items_d8.json` | D8 | ARC test set, Challenge and Easy (CC BY-SA 4.0) |
| `persona/` | D8 | Chen et al.'s sycophancy trait data and coherence prompt, `safety-research/persona_vectors` (Apache-2.0) |
| `lirefs/mmlupro_validation.json` | D4 | MMLU-Pro validation split, used for the five-shot exemplars (MIT) |

## Not included, and how to rebuild

These files are left out because they contain data from repositories that carry no licence, or competition problems whose redistribution terms are unclear. Clone the sources into `third_party/` and run the builder.

| file | contains | rebuild |
|---|---|---|
| `calibration.json`, `items.json` | D1: Arditi et al.'s harmful and harmless instructions (`andyrdt/refusal_direction` @9d852fa, Apache-2.0), XSTest (@d7bb5bd, CC BY 4.0), OR-Bench-Hard and -Toxic (`bench-llm/or-bench`). D2: Marks & Tegmark's curated statements (`saprmarks/geometry-of-truth` @5d1c630, no licence). D3: paper 4's items. | `python -I code/make_items.py third_party code/data` |
| `lirefs/mmlu-pro-3000samples.json` | Hong et al.'s 3,000 MMLU-Pro questions with their reasoning/memory scores (`yihuaihong/Linear_Reasoning_Features` @4f95e7e, no licence) | copy `mmlu-pro-3000samples.json` from that commit into `code/data/lirefs/` |
| `items_d9.json` | D9: the 83 AMC 2022–23 problems of `AI-MO/aimo-validation-amc`, and MATH test problems | `python -I code/d9_tools/fetch_d9_data.py third_party/d9_data`, then `python -I code/make_items_d9.py third_party/d9_data` |

D5's calibration chains (Ward et al.'s annotated chains, `jnward/latent-backtracking` @9c5a640, no licence) and D9's released reference vectors (`andotalao24/Identify_true_decorative_thinking` @e1da4c0, no licence) are read from `third_party/` at run time and are not redistributed either. D6's code reads `AlphaLab-USTC/LRM-plans-CoT` @2ba2485 the same way.

## Per-item results

`results-per-item-part01.zip` … `results-per-item-part10.zip` (on Zenodo, not in git; unzip all ten into one folder) hold every main-run record: clean outcomes, pushes, Check E rows, D9's chains, the exploratory strength sweep, and the judge scores. Every model-written string longer than 40 characters has been removed: answers, chains of thought, and completions to harmful prompts. Ids, labels, parsed answers, targets, projections and KL values are kept. `results/strip_report.json` (in part 10) counts what was removed, by field. One file, `results/d9/q3-8b_main_clean.jsonl`, is split by line into `.part1.jsonl` and `.part2.jsonl`; concatenate them. No direction vectors are distributed. They can be rebuilt with the code, but the refusal-removal direction in particular is deliberately not published.
