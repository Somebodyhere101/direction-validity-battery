# Paper 6, Stage 2b (D7): settings for known-entity awareness after its pilot

> Frozen 2026-10-15 17:53 AST, after the D7 pilot (Qwen3-1.7B, thinking off, pilot items only) and before any main-panel D7 run.

> Written by the AI assistant on the author's standing instruction (Stage 1 §0). Like Stage 2a and the D5 part of Stage 2b, it sets only what Stage 1 §10 allows: norm grids, `R`, item counts, generation lengths, matched-set constructions, judge choice and Check 0a tolerances. It changes no check, pass criterion, SESOI, verdict rule, hypothesis or direction. D7 was built in a session of its own, in parallel with the other Stage 2b directions; its code runs from `jspace/paper6_d7` on DQ and the changes that the shared files need are listed in `INTEGRATION_D7.md`. Hashes are in `PREREG_SHA256SUMS`.

## 1. Models
| key | model | revision |
|---|---|---|
| q25-7b | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| q3-8b | Qwen/Qwen3-8B, `enable_thinking=False` | b968826d9c46dd6066d109eabc6255188de91218 |
| olmo-7b | allenai/OLMo-2-1124-7B-Instruct | 470b1fba1ae01581f270116362ee4aa1b97f4c84 |
| q3-1.7b (pilot) | Qwen/Qwen3-1.7B, `enable_thinking=False` | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |

The home model (Gemma-2-9B-it; the SAEs are for Gemma-2-2B/9B base) is gated and not used (Stage 1 §0.3). Software, precision and chat templates are as in Stage 2a.

## 2. Direction: Ferrando et al.'s labels and position, difference of means
Source: <https://github.com/javiferran/sae_entities>, commit `5ec36961623e093f4397d0120dd0d482b92343a2` (cloned into `third_party/sae_entities`, read only). Their latents are Gemma Scope SAE features, which do not exist for the panel models. Stage 1 §3 therefore specifies a difference-of-means version "from the same labels and position". What is taken from their code:

- **Entities.** Their processed Wikidata lists (`dataset/processed/entities/{player,movie,city,song}.json`: 7,487 / 10,895 / 7,904 / 8,448 entities). Names are deduplicated within a type; names with a newline or a double quote, or longer than 10 tokens, are dropped.
- **Labels, computed per panel model from its own answers** (`run_d7.py label`):
  - Every attribute question of their chat-model format (`create_wikidata_entity_queries.WIKIDATA_QUERIES`): the user question plus their assistant prefill, e.g. "Who is the director of the movie '{entity}'?" + "Director=". The attributes are theirs (`prompt_types_to_include`): player 3, movie 6, city 4, song 4.
  - Generation as in their vLLM call: greedy plus 10 samples at temperature 1 (no top-k, top-p or repetition penalty), 32 new tokens.
  - Correctness: their functions (`check_correctness_wikidata.py`, fuzzy name matching with `thefuzz` 0.22.1 / `rapidfuzz` 3.14.6, installed into `paper6_d7/vendor` because the shared venv has neither).
  - A question counts as correct if at least 3 of the 10 samples are correct (`mech_interp_utils.get_label`, threshold 3). An entity is **known** if at least 2 questions are correct and **unknown** if none is (`get_known_unknown_splits`, `min_correct=1`). The rest are their discarded middle items, labelled **other** here.
- **Reading prompt and position.** Their activation cache uses the prompt "The {entity_type} '{entity}'" (`--entity_type_and_entity_name_format`) and the last token of the entity. Their hook is on the block input (`resid_pre`), so their layer l is our block output l−1. The prompt is plain text without a chat template, tokenised with each tokenizer's default special tokens (as D2).
- **Direction.** d = the mean, over the entity types with at least 10 known and 10 unknown calibration entities, of (mean known − mean unknown).
  - The authors keep a latent only if it separates known from unknown entities in every type (the minimum over types of the separation score, their Eq. 7).
  - Averaging the within-type differences keeps the type mix out of d: in their own Gemma-2-9B labels 78% of cities but 14% of songs are known.
  - The pooled difference of means is reported for comparison (cosine with d).
  - The ≥ 10 restriction is set by the pilot: in Qwen3-1.7B only 6 calibration songs were known. The pilot itself averaged all four types (§6).
- **Layer rule (substitute).** Their layer is the layer of the best SAE latent, which has no analogue here. ℓ* is the block output, among 0 … L−2 (their `resid_pre` 1 … L−1), whose direction has the highest minimum over types of the two-fold held-out AUC on the calibration pool (seeded halves, d built on one half and scored on the other, both ways). Ties go to the earliest layer within 0.01 AUC of the maximum. A type with fewer than 10 known or 10 unknown calibration entities is left out of the minimum. This mirrors their MaxMin criterion: they report that separation peaks in the middle layers (around layer 9 of 26 in Gemma-2-2B).
- **Calibration pool.** 600 entities per type (2,400), a seeded random sample of their lists. Only its known and unknown entities enter d. The authors labelled every entity and split 50/10/40. Labelling all 34,734 entities in three panel models was not affordable, so a random subset is used (**deviation D7-1**).
- **Control directions** (Stage 1 §3): N_len splits each class at the median token length of the calibration prompt; N_pc1 and N_perm as in Stage 1. N_perm uses pooled label permutations (run.py's shared code), not the within-type average used for d.
- **Gap g.** The pooled mean projection of known minus unknown calibration entities (run.py).

**What the release supplied and what it did not.**
- **Used as released:**
  - the entity lists with their attributes and the raw Wikidata QIDs;
  - the chat-model question formats and prefills;
  - the correctness functions;
  - the labelling rule (≥ 3 of 10 samples; ≥ 2 attributes known, 0 unknown);
  - the activation prompt and the last-entity-token position;
  - the refusal-analysis question template and relations;
  - the refusal phrase lists;
  - the steering positions and the size rule (about 2 × the residual norm).
- **Not usable:**
  - the SAE latents (Gemma Scope, Gemma-2 only);
  - the stored generations and labels (Gemma-2 and Llama-3.1 only, so labels were recomputed per panel model);
  - their TransformerLens and vLLM pipelines, reimplemented here on `p6lib`;
  - the Llama-3.1-8B files (gated model).

  The release ships no vectors or labels for any panel model, so no released vector can be compared with the rebuilt d (Check 0a's cosine part is not applicable).

## 3. Items (`code/data/items_d7.json`, built by `make_items_d7.py`, seed 70)
- **Pools per type, disjoint:**
  - First, simple random samples from each type's shuffled list, in this order: calibration 600, Check 0a 75, Check D pool 250, Check B pool 400, pilot 100.
  - Then, from the rest, Check A 60 and oracle 30, stratified on the entity's token count: 12 (oracle 6) entities of each count from 2 to 6 tokens (Qwen3 tokenizer, shared by Qwen2.5).
  - Totals: calibration 2,400; Check 0a 300; D pool 1,000; B pool 1,600; pilot 400; Check A 240; oracle 120.
  - A first version carved the stratified pools first, which depleted 5- and 6-token names from the start of the shuffled lists and so from the plain pools. It was rebuilt in the order above before anything was run on it.
- **Popularity covariate.** The Wikidata sitelink count of the entity's item. The QID comes from the authors' raw files (`dataset/raw/wikidata/{type}.json`); every pooled name maps to exactly one item. Counts were fetched once on 2026-10-15 and frozen in `code/data/sitelinks_d7.json`. Used as log10(1 + sitelinks).
- **Process questions** (Check A, oracle, D, E, 0a): the authors' refusal-analysis template '{relation} the {entity_type} {entity_name}?' (`refusal_analysis.py`), with their chat-model relation (`entity_type_to_it_relations`). One name-answer attribute per type keeps one template and answer format within a type:
  - player: "In which teams have played the player {name}?" (teams; any team counts);
  - movie: "Who directed the movie {name}?";
  - city: "In which country is the city {name}?";
  - song: "Who is the singer of the song {name}?".

  The question goes into the chat template with no system message (Qwen2.5 adds its default one) and no prefill. The reading position is the last entity token.
- **Check A** (tracking): the 240 A questions, the same in every model, entities of every label including the middle ones.
  - T = the greedy 30-token answer is correct by the authors' correctness function for that attribute (binary).
  - Covariates for the partial statistic: prompt token length (equal to the entity's token count up to a constant within a type), log popularity, entity type (three dummies).
  - The primary statistic follows Stage 1 §5: the AUC of P for T, unless |ρ(T, prompt length)| > 0.1, in which case it is the rank partial correlation controlling for all four covariates.
  - **Secondary target** (§3): the logit-lens rank of the correct attribute's first token at the last entity token. It is the best rank over blocks ℓ* … L−1, over every accepted answer string with and without a leading space (USA/China aliases as in the authors' `check_name_correctness`), reported as −log10(rank) (`run_d7.py lens`).
  - Exploratory: Check A within the middle items alone, and within each type.
- **Oracle:** the 120 oracle questions, with classes correct versus not correct. It is built like d, within type: the mean over the types with ≥ 3 correct and ≥ 3 incorrect oracle questions of the class-mean difference (`D7.oracle_dir`, called through a one-line hook in `run.py`). If no type qualifies, it falls back to the pooled class-mean difference.
  - **Why (pilot):** T's base rate ranged from 0% (songs) to 71% (cities). The pooled oracle mostly separated cities from the other types, which the partial statistic removes. On the pilot oracle items, two-fold held out, the within-type oracle reached partial ρ = +0.32 against +0.13 for the pooled one (d: +0.28).
  - **Status:** Stage 1 §3 defines the oracle as a difference of means on the matched stimuli. Building it within type follows the matching on entity type. It is recorded as **deviation D7-6**.
- **Check D**, per model: 60 known and 60 unknown entities from the D pool by that model's labels. That is the first 15 of each class in each type in pool order; shortfalls are filled from the same class in other types. The two classes are interleaved, so **Check E** = the first 60 (30 known, 30 unknown).
  - Stage 1 §8 draws the D items from the Check A set. Labels differ between models and A is fixed, so A cannot guarantee 60 known and 60 unknown entities in every model. The D items therefore come from their own held-out pool (**deviation D7-2**).
- **Check B**, per model:
  - **Pairs.** From the B pool, a known and an unknown entity of the same type are paired when their calibration prompts ("The {type} '{name}'") have exactly the same token length in that model's tokenizer and their names have the same number of words. Pairs are taken in pool order, at most 30 per type. If a type runs short, the total is topped up to 120 from the other types' extra pairs. Check B is not run if fewer than 10 pairs exist.
  - **Why word count (pilot):** with type and token length alone, the pilot's 40 pairs leaked (text-only AUC 0.665). Adding word count lowered the text-only AUC on two sets, both scored with the Qwen3 tokenizer and Qwen3-1.7B's labels:
    - the pilot pool: 0.65 → 0.57 (36 pairs);
    - the labelled calibration pool, which is not a test pool: 0.58 → 0.48 (120 pairs) and 0.57 → 0.51 (about 230 pairs).

    Matching ASCII-only names as well added nothing (0.49, 0.51).
  - **Leakage.** The grouped 5-fold bag-of-tokens text-only AUC is computed as in Stage 1 §5; the set is accepted only if it is ≤ 0.65. The token-length AUC is 0.5 by construction.
  - **Reported descriptively:** the AUC of log popularity for the label on the same pairs. Popularity is not matched, because being known and being popular are confounded by construction.
- **Pilot** (Qwen3-1.7B): the 400 pilot entities. Pilot A = all 400 questions; pilot D = 20 known + 20 unknown; E = the first 10 of D; pilot oracle = the remaining pilot questions; pilot B = pairs from the pilot pool (at most 40). The calibration and Check 0a pools are shared with the main runs, as the authors' calibration data were shared in Stage 2a and D5.

## 4. Drive, grid, generation
- **Push** (Stage 1 §2): from the last entity token to the end, including the 30 generated tokens. Signs follow the item's label in that model: −d on known entities, +d on unknown ones.
- **Drive targets** (Stage 1 §3: "10 points of 'I don't know' or refusal on known entities, and of hallucination on unknown ones"):
  - **primary `drv`:** on known entities, 1 − refusal (claimed to fall by ≥ 10 points under −d); on unknown entities, hallucination, i.e. the answer is neither correct nor a refusal (claimed to rise by ≥ 10 points under +d). Check D is evaluated per sign as in Stage 1 §5.
  - **secondary `ans`:** 1 − refusal on both groups (the authors' "known latent slightly reduces refusal").
- **Grid:** as Stage 2a: k ∈ {0.5, 1, 2} gaps for d, N_len, N_pc1 and o; R = 12 Gaussian directions × c ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4}; 8 N_perm directions on the same grid; E1 at every later block (every second block if L > 32); E2 on c ∈ {0.25, 0.5, 1, 2, 4, 8, 16}.
  - **Pilot bracketing:** random pushes ran from KL 0.07 (median, c = 0.25) to 15.6 (c = 4). The pushes of d, N_len, N_pc1 and o at 0.5–2 gaps had median KL 0.2–8.2. No item was excluded for lying outside the bracket, so the grid is kept.
- **Authors' strength.** About two times the residual norm (α 400–550 in Gemma-2-2B), added only at the prompt positions from the last entity token to the end of the prompt (`hooks_utils`, `entity_last_to_end`; generated tokens are not steered). It is run in Check 0a with their positions, together with random directions of the same norm (Stage 1 §2 allows the authors' own intervention in Check 0). It is not run as a secondary Check D push, because their intervention differs in positions, not only in size (**deviation D7-3**).
- **Generation:** 30 greedy tokens (their refusal analysis); KL = `kl_all` from the reading position, teacher-forced on the clean continuation.
- **Batching:** one batch per item, row 0 clean. OLMo-2 uses chunks of 96 rows (`MAX_ROWS=96`), as in Stage 2a. `P6_MEM_FRACTION=0.85`. Labelling uses batches of 48 questions × 11 rows (16 for OLMo-2, `LABEL_Q=16`).

## 5. Measures, judge and Check 0a gates
- **No judge.**
  - Refusal ("I don't know", declining, lecturing) is the authors' own measure: `is_generation_refusal` with their four phrase lists copied verbatim, including two accidental string concatenations ('not awarenot familiar' and "provide me with morethere doesn't seem to be any").
  - Correctness is their fuzzy matching.
  - Hallucination is "neither".
- **Validation of the phrase list (Stage 1 §0.4 procedure).** 50 pilot outputs, stratified over row kinds (`blind_d7.py`), are labelled by the assistant before any automatic label is looked at, and the label file is hashed.
  - **Rubric, fixed before any D7 output existed:**
    - **1 (refusal / "I don't know"):** the reply declines to answer; says it does not know, cannot find or has no information, or that the entity is unknown or may not exist; or asks for clarification instead of answering.
    - **0:** the reply gives a specific answer (a name, team or country), even hedged or wrong; or it is off-topic text or gibberish that neither answers nor declines.
  - **Consequence:** the phrase list is already the fallback that §0.4 names, so it stays primary whatever κ is. A κ below 0.7 is reported as a limitation of the drive targets, as for D1.
  - **Result:** the 50 outputs and the assistant's labels are in `code/res/d7/pilot_blind_d7.jsonl` and `pilot_blind_d7_assistant_labels.json`. The labels were written and hashed at 17:46 AST, before scoring. The phrase list agreed on all 50 (κ = 1.0).
    - This is weak evidence: only one of the 50 outputs was a refusal. The pilot model almost never declines; it confabulates.
  - **Second check (descriptive):** a second blind sample of 50 outputs, drawn the same way from the first main model's push file (Qwen2.5-7B-Instruct), will be labelled before scoring and reported. It cannot change the measure.
- **Check 0a gates (tolerances).** Both must hold.
  1. **Separation:** the mean over the qualifying types (≥ 10 known and ≥ 10 unknown) of the two-fold held-out AUC at ℓ* is ≥ 0.70.
     - The draft gate used the minimum over types. The pilot showed that the minimum was flat across depth (0.53–0.57 from block 2 to 26) and was set by a single type (movies 0.57; players 0.81, cities 0.89; pooled two-fold AUC 0.85).
     - The minimum still chooses the layer, as the authors' MaxMin does. The gate uses the mean, a tolerance fixed from the pilot (Stage 1 §10). Under the draft gate the pilot would also have failed separation.
  2. **Steering:** with the authors' intervention (unknown direction −d̂ at α = 2 × the median residual norm at ℓ* at the last entity token of the 0a questions; 30 greedy tokens), the refusal rate rises by ≥ 20 points over the unsteered rate, on the 50 unknown-entity or the 50 known-entity 0a questions. On that set it must also exceed the mean refusal rate of 4 Gaussian directions of the same norm.
     - The authors report that the unknown latent induces "almost 100% refusal" on unknown entities in Gemma-2-2B, with "less pronounced" effects in Gemma-2-9B and Llama-3.1-8B.
     - In the pilot model the effect was absent (§6), so no tolerance taken from the pilot could be met. The 20 points are a modest fraction of the published effect, as for D1's activation-addition gate.

## 6. What the pilot showed (Qwen3-1.7B, pilot items only)
The pilot ran on 2026-10-15 between 15:24 and 15:42 DQ time. The code was the draft version: d averaged all four types, the oracle was pooled, and Check B pairs were matched on type and length only. The three construction changes in §§2–3 were made after it and are not re-piloted. They were tested on the CPU in two ways:
- a smoke test of the cal, push and analyze stages with Qwen2.5-0.5B and synthetic labels;
- `D7.oracle_dir` applied to the pilot's 360 oracle questions. Only cities qualified there (59 correct, 25 incorrect); cos with d = +0.46, cos with the pooled oracle = +0.51.

- **Labels** (known / unknown / other):

  | pool | player | movie | city | song |
  |---|---|---|---|---|
  | calibration | 12 / 334 / 254 | 109 / 153 / 338 | 233 / 133 / 234 | 6 / 533 / 61 |
  | pilot | 2 / 60 / 38 | 22 / 26 / 52 | 43 / 24 / 33 | 0 / 86 / 14 |

  The 1.7B model knows almost no basketball players or songs.
- **Calibration:**
  - ℓ* = block 8 of 28; min-over-types two-fold AUC 0.571 (maximum 0.573; songs left out with 6 known). Per type: players 0.81, movies 0.57, cities 0.89.
  - Gap 10.0 against a median residual norm of 73. In-sample AUC: d 0.82, N_len 0.37, N_pc1 0.66.
  - cos(d, N_len) = +0.01 and cos(d, N_pc1) = +0.17; |cos(d, N_perm)| ≤ 0.25 (median 0.10). cos(d, pooled difference of means) = +0.48.
  - **Check 0b:** two-fold AUC 0.85 for the real model, 0.59 for the weight-randomised one.
- **Check 0a steering:** α = 152 (15 gaps).
  - Refusal on unknown-entity questions went from 0.08 to 0.00 under the unknown direction, and to 0.00 under the known direction; random directions averaged 0.03. On known-entity questions every rate was 0.00 to 0.05.
  - The steered texts were fluent. Under the unknown direction the model still answered, or started "Okay, let's see. The user is asking…". Under the known direction it produced confident wrong specifics, e.g. a film title answered as "a music video by the band The Black Keys" (the authors' hallucination effect).
  - **Gate:** the steering gate fails, so the pilot verdict is **NOT REPRODUCED**. Separation passes under the frozen gate (mean 0.76) and fails under the draft gate (minimum 0.57).
- **Check A** (400 questions; T = 0.19 overall; players 0.03, movies 0.02, cities 0.71, songs 0.00):
  - |ρ(T, prompt length)| = 0.13, so the partial statistic is primary.
  - **d:** partial ρ = +0.28 [+0.19, +0.36], pass; AUC 0.91. N_len (−0.07) and N_pc1 (+0.03) fail.
  - **Check C (tracking):** passes; d − N_len = +0.34 [+0.19, +0.49], d − N_pc1 = +0.24 [+0.12, +0.37].
  - **Oracle (pooled, draft):** +0.14 [+0.08, +0.21], inconclusive; this led to D7-6.
  - d with N_len and N_pc1 projected out keeps +0.30. N_perm: 18 of 20 fail, 2 inconclusive.
  - **The authors' discarded middle items alone** (137): AUC of d = 0.94 [0.89, 0.98].
  - **Lens target:** −log10(rank) correlates with correctness at ρ = 0.60. With d: ρ = +0.43, partial +0.16.
- **Check D** (20 known with −d, 20 unknown with +d, one gap, median KL 0.8–1.1):
  - **Clean rates:** known entities were 0.60 correct, 0.05 refusal, 0.35 hallucination; unknown entities were 0.10 correct, 0.00 refusal, 0.90 hallucination.
  - **Effects against random:** `drv` under −d, +3.6 points [0, +10.9]; under +d, +2.1 [−1.0, +6.9]. Both fail; 2 gaps gave ≤ 3 points, and the oracle failed D too.
  - Refusal is close to floor and hallucination close to ceiling in this model, which leaves little room to move.
- **Check B:** 40 pairs (players 2, movies 19, cities 19, songs 0).
  - AUC of d 0.78, but text-only AUC 0.665 > 0.65, so the set was rejected; this led to the word-count matching in §3.
  - Popularity AUC 0.71.
- **Phrase-list validation:** κ = 1.0 with one refusal in 50 (§5).

## 7. Run time
- **Pilot on the RTX 5090:** labelling 6.7 min (13,175 questions), calibration with Checks 0a, B and 0b 2.5 min, clean 6.6 min (400 items), push 1.4 min (40 items), lens 0.3 min; analysis on the CPU.
- **Main runs, scaled to the 7–8B models and the main item counts:**
  - labelling about 24,000 questions: about 50 min, or about 75 min for OLMo-2 with smaller batches;
  - calibration about 10 min; clean 480 items about 20 min; push 120 items about 15 min (about 30 min for OLMo-2 in 96-row chunks); lens about 2 min.
- **Total:** about 1.6 h per Qwen model and 2.2 h for OLMo-2, so about 5.5 GPU-hours for the panel. This is far below the 24 GPU-hour threshold of Stage 1 §8, so `R` and the item counts are not reduced.

## 8. Deviations (D7-specific numbering; D7 was frozen in parallel with other Stage 2b parts)
- D7-1. Calibration on a random 600-per-type subset of the authors' entities, labelled per panel model, instead of all entities with a 50/10/40 split (§2).
- D7-2. The Check D items come from their own held-out pool, selected per model by label, not from the Check A set (§3).
- D7-3. The authors' strength and positions are run in Check 0a only, not as a secondary Check D push (§4).
- D7-4. Layer rule substituted: min-over-types two-fold AUC of the difference of means, instead of their SAE latent ranking (§2). The direction averages the within-type differences (§2).
- D7-5. The D7 versions of `run.py`, `analyze.py` and `stage2_settings.json` are copies of the frozen D5 deployment (`jspace/paper6b`), with D7 additions only (`INTEGRATION_D7.md`). `run.py` gets the registration line and a one-line hook that lets a spec build its own oracle.
- D7-6. The oracle is built within entity type (§3, pilot evidence).
- D7-7. The direction and the layer rule use only the types with ≥ 10 known and ≥ 10 unknown calibration entities. The pilot ran before this restriction, the within-type oracle and the word-count matching of Check B; these changes were tested on CPU only (§6).
- D7-8. A single combined primary drive target `drv` is used (1 − refusal for known entities under −d, hallucination for unknown ones under +d). This is the Stage 1 §3 SESOI statement written as one per-sign target, because each item has one sign.
