# Paper 6 groundwork: literature review for "sanity checks for linear directions"

Compiled 2026-10-14 by the AI assistant, for the author's review. No experiments were run.

**How this was searched.**
- Three parallel web searches:
  - the canonical validation literature;
  - a 2024-07 to 2026-10 novelty sweep, in extended mode;
  - published directions with released code.
- Every paper from 2026 that bears on the novelty verdict was then opened on arXiv by the assistant and checked against its abstract or full text. These are marked ✓.
- Items marked ~ were seen only in search results or reported by a search agent, and have not been opened.
- arXiv IDs of the form 26xx.xxxxx are after the assistant's training cutoff. They exist as described at abstract level, but their full methods have not been read.

## 0. Verdict in brief

**The battery as a whole has not been done.** No paper found applies one fixed set of validity checks to many published directions with:
- per-item process targets;
- a label-free nuisance competitor;
- KL-matched random pushes;
- stage localisation.

**Taken one at a time, the checks are not new, and two of them are already taken:**
- **(d) KL-matched random pushes are taken.** SteerCheck (Luo, Liang & Xuan, Aug 2026) matches every confirmatory direction to an off-target KL band. Ferrara (Aug 2026) matches random controls on Jensen–Shannon divergence and states that "norm matching alone systematically understates the control". Cite both. Do not claim KL matching as a contribution.
- **(e) Stage localisation and layer sweeps are routine.** The specific *hand-off* test, re-imposing the clean projection at later layers, has precedents in path patching and in Yuan & Vlachos's (Aug 2026) "direct logit bias equivalence" and early-layer controls.
- **(b) Matched stimuli have single-case precedents:**
  - Devbunova (Mar 2026) histogram-matches length for evaluation-awareness probes on Llama-3.1-8B.
  - Sahoo et al. (Jun 2026) show that reasoning-mode probes on Qwen3-14B fall from 100% to chance once format is regressed out.
  - Marks & Tegmark's curated true/false sets.
  - XSTest-style contrast sets.
- **(c) is only partly occupied.** Hewitt & Liang's control tasks, label shuffles and sign-randomised nulls (SteerCheck) are *unstructured* label-free baselines. Sahoo et al. and Yuan & Vlachos regress out or test hand-picked structured confounds. No paper found builds a structured nuisance direction *from the same calibration prompts without their labels* (within-class length, within-class first principal component) and pits it head-to-head against the published direction on tracking and steering. Paper 4 of this series is the only example.
- **(a) has the weakest prior art as a general check.** Several papers already define their *own* labels from behaviour: Ferrando et al.'s known/unknown entities, Venhoff et al.'s annotated reasoning behaviours, Zhao et al.'s True Thinking Score, Kirch et al.'s on-policy probe data. None re-scores *other people's* published directions against per-item measures of what the model did.

**Closest overlaps, ranked:**
1. SteerCheck: a preregistered steering-attribution audit, KL-matched, with three null families; one behaviour (CAA anti-sycophancy) on Qwen3-14B.
2. Wu, Zhao & Chen (Aug 2026): a reusable audit protocol applied to four published "AI neuroscience" claims across 17 models.
3. Sahoo et al. (Jun 2026): format deconfounding plus a random-direction control for one reasoning-mode probe; the nearest *result*.
4. Yuan & Vlachos (Aug 2026): four controls for whether a steering "fix" is a label bias.
5. Devbunova (Mar 2026): matched-format re-test of evaluation-awareness probes.

**What remains open, and should carry the paper:**
1. Process-defined per-item targets as a standard check.
2. The label-free structured nuisance direction as a competitor.
3. Oracle directions and known-answer cases (refusal; the CR axis) that test the battery itself.
4. Breadth: about ten published directions, including several *process* claims about reasoning (reasoning vs memorisation, backtracking, reasoning strength, true vs decorative thinking).

**Risks:**
- Five related preprints appeared between August and October 2026, so the window is closing.
- The title "Sanity checks for …" is already used for SAEs by Korznikov et al. (Feb 2026). A distinct title is safer, for example "Does the direction track the process? A validity battery for linear directions in LLMs".

## 1. The battery, for reference

A *direction claim* says that a vector `d` at layer `ℓ*` tracks a process (the projection covaries with it) or drives it (adding `d` changes it). The battery's checks:
- **(a) Process-defined target.** `T` is measured per item from the model's own behaviour or internals: settling depth, actual refusal, actual backtracking, thinking-token count. It is never the prompt-class label.
- **(b) Matched stimuli.** Format, length and vocabulary are matched, with leakage verified by text-only and length-only baselines.
- **(c) Label-free nuisance baseline.** `N_len` (within-class long-minus-short) and `N_pc1` (within-class PC1) are built from the same calibration prompts and compared head-to-head with `d`. Label-permuted directions serve as an unstructured null.
- **(d) Random pushes matched on output change.** Paper 4's interpolation in log KL, with Gaussian and permuted-label references.
- **(e) Stage localisation.** The layer at which the push's effect hands off from `d` itself to other components, plus a direct-path (final-layer) control.

Two further elements test the battery itself:
- **Oracle directions.** A difference of means on the process target itself; this is a ceiling and a steerability check.
- **A randomised-model calibration check**, the closest analogue of Adebayo et al.'s model-randomisation test.

The full specification is in `PREREGISTRATION_DRAFT.md`.

## 2. Required and foundational sources

Each entry gives what the work does, the checks it covers (full, partial = part., or none), and how the battery differs. Every URL appeared in search results or was opened.

**2.1 Hewitt & Liang (2019), "Designing and Interpreting Probes with Control Tasks."** EMNLP-IJCNLP 2019. <https://aclanthology.org/D19-1275/> · <https://arxiv.org/abs/1909.03368>
- **Does:** a *control task* assigns each word type a random but consistent label. *Selectivity* = linguistic-task accuracy − control-task accuracy. It measures how much of a probe's accuracy comes from the probe's own expressivity.
- **Covers:** (c) part., in spirit: the control uses the same inputs, but with random labels, not a structured nuisance. Per-layer selectivity only for (e).
- **Battery differs:** the label-free competitor is a *structured* nuisance direction (within-class length; within-class PC1) built from the same prompts. It is compared on tracking a *measured process* and on steering, not on probe accuracy. `N_perm` is the battery's direct analogue of a control task.

**2.2 Adebayo, Gilmer, Muelly, Goodfellow, Hardt & Kim (2018), "Sanity Checks for Saliency Maps."** NeurIPS 2018. <https://arxiv.org/abs/1810.03292>
- **Does:** two randomisation tests. Model-parameter randomisation (also cascading from the top layer) and data randomisation (training on permuted labels). An explanation that does not change fails. Guided Backprop and Guided GradCAM fail; gradients and GradCAM pass.
- **Covers:** none of (a)–(e) directly. It is the template for asking whether an explanation depends on what it claims to explain.
- **Battery differs:** the battery borrows the *form* (cheap, generic, method-agnostic checks with clear pass/fail) and both randomisation tests: Check 0b for the model, `N_perm` for the data. Its main checks are against *nuisance* and *output-change* nulls, which saliency maps do not need.

**2.3 Tan, Chanin, Lynch, Paige, Kanoulas, Garriga-Alonso & Kirk (2024), "Analysing the Generalisation and Reliability of Steering Vectors."** NeurIPS 2024. <https://arxiv.org/abs/2407.12404> · code <https://github.com/dtch1997/steering-bench>
- **Does:** CAA difference-of-means vectors on about 40 datasets. Steerability is the slope of the A/B logit difference across multipliers. It finds high per-item variance, anti-steerable items, and spurious "steerability bias" toward the A/B or Yes/No position.
- **Covers:** (b) part.: answer positions are balanced but length is not matched. (e) is a layer sweep. No random baseline; no (a), (c), (d).
- **Battery differs:** it asks whether the direction tracks or drives a *process* measured per item, against a nuisance direction and KL-matched nulls, rather than how reliably a vector moves its own MCQ logit.

**2.4 Wu, Arora, Geiger, Wang, Huang, Jurafsky, Manning & Potts (2025), "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders."** ICML 2025. <https://proceedings.mlr.press/v267/wu25a.html> · <https://arxiv.org/abs/2501.17148> · code <https://github.com/stanfordnlp/axbench>
- **Does:** concept detection is scored by AUROC on generated data, with polysemy hard negatives. Steering is judged by an LLM on concept, instruction and fluency, with the factor chosen on held-out instructions. Prompting wins at steering, DiffMean at detection, and SAEs are not competitive.
- **Covers:** (a) part.: steering is judged on actual outputs. (b) part.: hard negatives. No random baseline, no KL matching, fixed layers.
- **Battery differs:** AxBench *ranks methods* across many synthetic concepts. The battery *audits specific published claims* about processes, and asks whether the published direction beats confounds and nulls.

**2.5 Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee & Nanda (2024), "Refusal in Language Models Is Mediated by a Single Direction."** NeurIPS 2024. <https://arxiv.org/abs/2406.11717> · code <https://github.com/andyrdt/refusal_direction>
- **Does:** difference of means, harmful − harmless, at every layer and post-instruction position.
  - Selection (App. C.1) keeps the candidate with the lowest bypass score, subject to induce score > 0, KL on harmless prompts under ablation < 0.1, and layer < 0.8L.
  - Effects are scored on actual refusal (substring match plus Llama Guard 2), with side-effect checks on MMLU, ARC, GSM8K, TruthfulQA and CE loss.
  - No random-direction control was found in the main experiments.
- **Covers:** (a) part.: outcomes are behavioural, but the direction comes from prompt labels and there is no per-item tracking. (e) part.: layer and position sweep. KL is used as a *side-effect filter*, not to match controls. No (b), (c) or (d).
- **Battery differs:** it is used as the *positive control*, re-scored on actual refusal on XSTest/OR-Bench items where refusal and harmfulness come apart, with a nuisance direction and KL-matched nulls. Zhao et al. (2025, §4) show that harmfulness and refusal are encoded separately, which supports using refusal, not harmfulness, as the target.

**2.6 Marks & Tegmark (2024), "The Geometry of Truth."** COLM 2024. <https://arxiv.org/abs/2310.06824> · code <https://github.com/saprmarks/geometry-of-truth>
- **Does:**
  - Curated true/false datasets.
  - Mass-mean (MM) probes.
  - Causal interventions that add a direction over the end-of-statement tokens, scored by the normalised indirect effect (NIE) on P(TRUE) − P(FALSE).
  - The `likely` dataset (non-factual text ending in a probable vs an improbable token) is a *text-probability control*.
- **The closest precedent for (c).** In their Table 2 (NIE on sp_en_trans), an MM direction trained on `likely` steers false→true almost as well as one trained on `cities`:
  - LLaMA-2-13B: 0.70 vs 0.77;
  - LLaMA-2-70B: 0.68 vs 0.58.

  It is far weaker true→false: 0.54 vs 0.90, and 0.27 vs 0.89. (Checked by the assistant on the arXiv HTML, v3.)
- **Covers:** (b) part.: templated sets with negations. (c) part.: a confound direction, but built from *different prompts with confound labels*. (e) part.: patching localises the effect to end-of-statement tokens. (a) none: labels are ground truth, not the model's own verdicts. No random-direction control.
- **Battery differs:** the confound direction is built *from the same calibration prompts without labels*. The target is the model's own verdict. Pushes are compared at matched KL.

**2.7 Park, Choe & Veitch (2024), "The Linear Representation Hypothesis and the Geometry of LLMs."** ICML 2024. <https://arxiv.org/abs/2311.03658> · code <https://github.com/KihoPark/linear_rep_geometry>. And **Park, Choe, Jiang & Veitch (2025), "The Geometry of Categorical and Hierarchical Concepts in LLMs."** ICLR 2025. <https://arxiv.org/abs/2406.01506>
- **Does:**
  - Formalises "linear representation" in embedding and unembedding space, with a causal inner product.
  - Builds concept directions from counterfactual word pairs.
  - Controls include random word pairs, off-target directions, and (2025) shuffled-unembedding nulls.
- **Covers:** (b) part.: counterfactual pairs. (c) part.: shuffled nulls only. No process targets and no stage analysis.
- **Battery differs:** Park et al. give the *theory* of what a linear representation is for token-level concepts. The battery is an *empirical audit* of residual-stream directions that claim to track processes. It tests whether the claimed variable is the one represented, not whether some variable is represented linearly.

**2.8 The causal-abstraction line (Geiger and colleagues).**
- **Papers:**
  - Geiger, Lu, Icard & Potts, "Causal Abstractions of Neural Networks," NeurIPS 2021. <https://arxiv.org/abs/2106.02997>
  - Geiger, Wu, Potts, Icard & Goodman, "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations" (DAS), CLeaR 2024. <https://arxiv.org/abs/2303.02536> · <https://proceedings.mlr.press/v236/geiger24a.html>
  - Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability," JMLR 26(83), 2025. <https://jmlr.org/papers/v26/23-0058.html>
  - Wu, Geiger, Icard, Potts & Goodman, "Interpretability at Scale" (Boundless DAS), NeurIPS 2023. <https://arxiv.org/abs/2305.08809>
- **Does:** aligns a hypothesised high-level variable with a (rotated) subspace, scored by interchange-intervention accuracy (IIA) on counterfactual pairs. Boundless DAS adds:
  - a random-rotation baseline (IIA 0.83 → 0.53);
  - control token positions;
  - failing alternative causal models;
  - a randomly initialised LLaMA-7B;
  - layer × position IIA heatmaps.
- **Covers:** (b) largely, since counterfactual pairs are matched by construction. (e) part.: heatmaps. (a) none: the target is a hypothesised algorithm variable. No nuisance direction and no output-matched nulls.
- **Battery differs:** causal abstraction asks whether a subspace *implements a variable of a hypothesised algorithm*. The battery asks the cheaper, prior question for already-published difference-of-means directions: do they track or drive the *measured* process better than a confound? Two cautions from this line apply to the battery's interpretation:
  - Makelov et al. (2.9): a subspace patch can work through a dormant pathway.
  - Sutter et al. (2.10): with unrestricted maps, any network can be mapped to any algorithm.

**2.9 Makelov, Lange, Geiger & Nanda (2024), "Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching."** ICLR 2024. <https://arxiv.org/abs/2311.17030>
- **Does:** shows that a subspace patch can produce the intended end-to-end effect by activating a *dormant* parallel pathway. Diagnostics: nullspace and rowspace decomposition, dormancy checks, comparison with full-component patching.
- **Covers:** (e) part.
- **Battery differs:** Check E asks *where* a push's effect hands off from `d` to other components, and whether a final-layer push of equal KL does as much. Makelov et al. ask whether the pathway is the model's own. The two are complementary; a VALID DRIVER verdict does not rule out their illusion.

**2.10 Sutter, Minder, Hofmann & Pimentel (2025), "The Non-Linear Representation Dilemma."** NeurIPS 2025. <https://arxiv.org/abs/2507.08802>
- **Does:** proves that, with unrestricted alignment maps, causal abstraction is vacuous. Shows 100% IIA in randomly initialised models.
- **Covers:** a model-randomisation null for causal abstraction.
- **Battery differs:** supports restricting the battery to *linear* directions as published, and motivates Check 0b.

**2.11 Canby, Davies, Rastogi & Hockenmaier, "How Reliable are Causal Probing Interventions?"** (earlier title: "Measuring the Reliability of Causal Probing Methods: Tradeoffs, Limitations, and the Plight of Nullifying Interventions"). IJCNLP-AACL 2025. <https://arxiv.org/abs/2408.15510>
- **Does:** scores interventions by *completeness* (the target property changed) and *selectivity* (non-targets unchanged), judged by oracle probes.
- **Covers:** (c) part., in spirit: off-target damage is measured.
- **Battery differs:** strengths are not matched across methods there. The battery matches on output KL and compares with a nuisance direction.

**2.12 The probing-validity classics.**
- **Papers:**
  - Lasri, Pimentel, Lenci, Poibeau & Cotterell, "Probing for the Usage of Grammatical Number," ACL 2022. <https://arxiv.org/abs/2204.08831>
  - Elazar, Ravfogel, Jacovi & Goldberg, "Amnesic Probing," TACL 2021. <https://arxiv.org/abs/2006.00995>
  - Ravichander, Belinkov & Hovy, "Probing the Probing Paradigm," EACL 2021. <https://arxiv.org/abs/2005.00719>
  - Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances," Computational Linguistics 2022. <https://aclanthology.org/2022.cl-1.7/>
- **Does:**
  - *Encoded is not used.* Amnesic projections test behavioural use.
  - Removing the same number of *random* directions is the control (Elazar et al.; Lasri et al.). KL is reported but not matched.
  - Lasri et al. localise use by layer.
- **Covers:** (a) part. (use judged on behaviour), (d) part. (dimension-matched random removal), (e) part. (Lasri).
- **Battery differs:** these are the conceptual ancestors of "tracks vs drives". The battery carries them over to modern steering claims with output-matched nulls and nuisance competitors.

**2.13 Farquhar, Varma, Kenton, Gasteiger, Mikulik & Shah (2023), "Challenges with Unsupervised LLM Knowledge Discovery."** <https://arxiv.org/abs/2312.10029>. And **Levinstein & Herrmann (2023/24), "Still No Lie Detector for Language Models."** <https://arxiv.org/abs/2307.00175>
- **Does:**
  - CCS and PCA follow an injected binary feature (random words such as "Banana"/"Shed", or a character's opinion) instead of truth.
  - Truth probes fail on negations; CCS largely learns negation.
- **Covers:** (c) part.: a *structured* nuisance in the same prompts, but injected by the experimenter, and tested against unsupervised methods.
- **Battery differs:** the nuisance is one that occurs *naturally* in the published calibration set (length; dominant within-class variance). The comparison is with supervised published directions, on process targets.

**2.14 Why norm-matched random controls are weak (precedents for (d)).**
- **Heimersheim & Mendel (2024)**, "Activation plateaus & sensitive directions in GPT2" (LessWrong interim report). <https://www.lesswrong.com/posts/LajDyGyiyX8DNNsuF/interim-research-report-activation-plateaus-and-sensitive-1>
  - Equal-norm perturbations give very different KL depending on direction.
  - The authors later lowered their confidence in the "sensitive directions" part. Lee & Heimersheim (2024, <https://arxiv.org/abs/2410.12555>) show that the choice of baseline drives that result.
- **Li et al. (2023), ITI**, <https://arxiv.org/abs/2306.03341>, Table 3. At the same α = 20 on LLaMA-7B:
  - random direction: KL 0.02, True×Info 31.2%;
  - mass-mean shift: KL 0.27, True×Info 42.3%.

  The random null moved the output an order of magnitude less. (Checked by the assistant.)
- **SteerCheck (2026, §4)** states the argument directly: matching the norm leaves the output sensitivity free.
- **Battery differs:** none of the pre-2026 work matches nulls on KL. SteerCheck and Ferrara (2026) do, and are cited as prior art for (d).

**2.15 Steering-evaluation and reliability papers (2024–2025).**
- **Pres, Ruis, Lubana & Krueger (2024)**, "Towards Reliable Evaluation of Behavior Steering Interventions in LLMs." <https://arxiv.org/abs/2410.17245>
  - Four evaluation properties; likelihood-based scoring. CAA is weaker than previously reported.
  - The baseline is the unintervened model; no random or KL controls.
- **Braun, Eickhoff, Krueger, Bahrainian & Krasheninnikov (2025)**, "Understanding (Un)Reliability of Steering Vectors." <https://arxiv.org/abs/2505.22637>
  - Reliability is predicted by the cosine similarity of training activation differences and by class separation along the vector.
- **Im & Li (2025)**, "A Unified Understanding and Evaluation of Steering Methods." <https://arxiv.org/abs/2502.02716>
  - The mean of differences is optimal; layer sweeps.
- **Da Silva et al. (2025)**, "Steering off Course," ACL 2025. <https://arxiv.org/abs/2504.04635>
  - 36 models. A constant final-logit shift reproduces DoLa's TruthfulQA gains: a trivial structured baseline matching a method, in the same spirit as Check E2.
- **Bhalla, Srinivas, Ghandeharioun & Lakkaraju (2024/25)**, "Towards Unifying Interpretability and Control: Evaluation via Intervention." <https://arxiv.org/abs/2411.04430>
  - Intervention success against coherence, as Pareto curves. Close in spirit to comparing at matched output damage, but without a null.
- **Battery differs:** these evaluate *methods* or *reliability*. None asks whether a given direction tracks or drives the process claimed for it, beyond confounds.

**2.16 Benchmarks with counterfactual matching.**
- **Mueller et al. (2025)**, "MIB: A Mechanistic Interpretability Benchmark," ICML 2025. <https://arxiv.org/abs/2504.13151>
  - The causal-variable track scores featurisers by IIA on counterfactual pairs with layer and position search. DAS is best.
- **Huang, Wu, Potts, Geva & Geiger (2024)**, "RAVEL," ACL 2024. <https://arxiv.org/abs/2402.17700>
  - Cause and isolate scores for disentangling entity attributes.
- **Covers:** (b) by construction; (e) part.; RAVEL's isolate score is close to an off-target check.
- **Battery differs:** these benchmark featurisers on *synthetic* causal variables. They do not audit published directions on process targets.

**2.17 Model-randomisation nulls for other interpretability tools (Adebayo descendants).**
- **Heap, Lawson, Farnik & Aitchison (2025)**, "Sparse Autoencoders Can Interpret Randomly Initialized Transformers." Its v2 title is "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers," listed at ICLR 2026. <https://arxiv.org/abs/2501.17727>
- **Korznikov et al. (2026)**, "Sanity Checks for Sparse Autoencoders," in §4.
- **Battery differs:** Check 0b applies the same idea to difference-of-means directions, as a descriptive check on how much calibration AUC means.

**2.18 Background for settling depth and stages.**
- nostalgebraist (2020), logit lens: <https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru>
- Belrose et al. (2023), tuned lens: <https://arxiv.org/abs/2303.08112>
- Lad, Gurnee & Tegmark, "The Remarkable Robustness of LLMs: Stages of Inference?" NeurIPS 2025: <https://arxiv.org/abs/2406.19384>. Layer deletion and swapping keep most accuracy; the paper proposes four stages.

## 3. Prior work by check (summary)

| Check | Already standard? | Best precedents | What the battery adds |
|---|---|---|---|
| (a) process-defined target | Partial. Own labels are sometimes behavioural; no one re-scores *others'* directions | Ferrando et al. 2025 (known = answered correctly); Venhoff 2025 and Ward 2025 (annotated traces); Zhao et al. 2025 (True Thinking Score); Kirch et al. 2025/26 (on-policy probe data); Arditi 2024 (behavioural outcomes) | Per-item `T` for every published direction, tracked *and* pushed. Oracle direction on `T` as a ceiling |
| (b) matched stimuli | Common by construction in causal-abstraction benchmarks; rare in steering | DAS, MIB, RAVEL; Marks & Tegmark; XSTest (Röttger et al., NAACL 2024, <https://arxiv.org/abs/2308.01263>); Devbunova 2026; Sahoo et al. 2026 | Matched sets for *evaluation* (not construction, per Petrov 2026); text-only and length-only leakage baselines; output-format clamp |
| (c) label-free nuisance baseline from the same prompts | **Not found** in this form | Unstructured: Hewitt & Liang (control tasks), label shuffles, SteerCheck's sign-randomised nulls. Structured but built otherwise: Marks & Tegmark's `likely`, Farquhar's injected words, Levinstein & Herrmann's negation, Da Silva's logit shift, Sahoo's regressed-out format | `N_len` and `N_pc1` from the *same* calibration prompts without labels, compared head-to-head on tracking and steering; residualised `d⊥` |
| (d) KL-matched random pushes | Rare; new in 2026 | SteerCheck 2026 (KL band); Ferrara 2026 (JS-matched); papers 3–4 of this series. Before that, norm-, α- or dimension-matched nulls (Elazar, Lasri, ITI, Boundless DAS) | Interpolation in log KL per item; in-distribution (permuted-label) second reference; H4 tests how much norm matching overstates effects |
| (e) stage localisation | Layer sweeps routine; "where is it decided" rare | Makelov et al. 2024; Lasri et al. 2022; Gao et al. 2026 (late output-sensitive component); Yuan & Vlachos 2026 (direct-logit equivalence) | Hand-off layer from re-imposing the clean projection; final-layer direct-path control at matched KL; OUTPUT-BIAS flag |

## 4. 2025–2026 work that overlaps (closest first)

Each entry gives what the work does and the checks it covers: full, partial (part.) or none.

1. ✓ **SteerCheck: Attribution Specificity and Alignment Leakage in Activation-Steering Audits.** Luo, Liang, Xuan. arXiv:2608.24335, 25 Aug 2026. <https://arxiv.org/abs/2608.24335>
   - A preregistered attribution audit with gates for mean, protected-tail, polarity, transfer and semantic claims.
   - Off-target KL on a disjoint neutral prompt bank (16-token horizon) is held in a band of 0.09–0.11 for every confirmatory direction.
   - Null families: isotropic random, PCA-subspace, and sign-randomised same-construction directions.
   - Main finding: sign-randomised nulls keep substantial alignment with the target ("alignment leakage"; 25% of draws exceed cosine 0.5), which limits what that comparator can distinguish. Many results are mixed or negative.
   - Scope: one behaviour (CAA anti-sycophancy/truth-consistency) at one layer, primarily Qwen3-14B (960 null directions), with DeepSeek-V2-Lite and Gemma-3-27B in secondary studies. Output length is not matched. The paper does *not* use the phrase "sanity checks".
   - **Covers:** (d) full; (c) part. (random label-free nulls, not structured nuisance); (e) part.; (a) none; (b) none.
   - **Difference:** one direction versus many; operational behaviour scores rather than process targets; no structured nuisance direction; no stage hand-off.
2. ✓ **When Is a Steerable Concept Representation Real? Measurement Confounds in a Cross-Family Audit of Neuroscience Parallels in LLMs.** Wu, Zhao, Chen. arXiv:2608.08159, 8 Aug 2026. <https://arxiv.org/abs/2608.08159>
   - Audits four published claim types: concept steerability, world map, number magnitude, language localisation. Uses 17 models from 0.6B to 72B.
   - Shows that "steerability grows with scale" is an artefact of raw units, a fixed operating point and the choice of readout. With norm-comparable interventions and held-out operating points, steering survives but the trend vanishes.
   - The protocol, stimuli and code are released.
   - **Covers:** (b) part., (d) part. (norm-comparable, not KL), (e) part.; (a) none; (c) none.
   - **Difference:** the closest *multi-claim audit*, but about static concepts and "AI neuroscience" parallels, not process claims; no nuisance direction and no KL matching.
3. ✓ **Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States.** Sahoo, Jain, Chadha, Chaudhary. arXiv:2606.02907, Jun 2026, TrustNLP @ ACL 2026. <https://arxiv.org/abs/2606.02907>
   - On Qwen3-14B (layer 32), probes for deductive, inductive and abductive reasoning reach 100%. They fall to chance after regressing out source dataset, option count and response length.
   - Steering along the centroid direction is no better than 20 random directions (p = 0.29).
   - Agreement of the reasoning traces with the intended mode is 42.5%, against 33% chance.
   - **Covers:** (a) part. (trace check), (b)/(c) part. (regressed-out structured confounds), (d) part. (random directions; the abstract does not say whether norms were matched); (e) none.
   - **Difference:** a single case on the authors' own probes. It is the strongest published precedent for this series' paper-2 and paper-4 findings, and should be cited prominently.
4. ✓ **Evaluating the Semantic Specificity of Representation Steering in Language Models.** Yuan, Vlachos. arXiv:2608.29431, 29 Aug 2026. <https://arxiv.org/abs/2608.29431>
   - Introduces Cross-Rule Transfer. Late-layer steering that "repairs" contradiction blindness drops accuracy on rules the model already handled from 99.6% to 40.4%, revealing a global label bias.
   - Four controls: direct-logit-bias equivalence, control-vector label flipping, cross-model grafting, early-layer checks.
   - **Covers:** (e) part. (the direct-logit control is the same idea as Check E2); (c) part.; (a) part.; (b) none; (d) none.
   - The search agent reports a validation checklist for steering claims in the full paper (Table 10). Not verified (~).
5. ✓ **Is Evaluation Awareness Just Format Sensitivity?** Devbunova. arXiv:2603.19426, Mar 2026, ICLR 2026 workshop "I Can't Believe It's Not Better". <https://arxiv.org/abs/2603.19426>
   - A 2×2 design (evaluation vs deployment × benchmark vs casual format) on Llama-3.1-8B-Instruct, with character length histogram-matched (30 bins) and layers 8–31 swept.
   - The probes track benchmark format. They are the author's own, not a re-test of published vectors.
   - **Covers:** (b) full, for one case; (e) part.
6. ✓ **What Does Activation Steering Control?** Gao, Peng, Wakamiya, Aramaki. arXiv:2608.22985, Aug 2026. <https://arxiv.org/abs/2608.22985>
   - CAA and ITI-style steering follow the *answer-extraction index* rather than the semantic label. Answer identifiers are reassigned while the intervention is held fixed.
   - The effect sits in later layers and in a low-rank output-sensitive component: 15.4% of the direction's squared norm carries 96.3% of the effect.
   - **Covers:** (b) part., (e) part. This is the same failure mode Check E's direct-path control targets.
7. ✓ **Open-Weight Masked Introspection.** Ferrara. arXiv:2608.20569, Aug 2026. <https://arxiv.org/abs/2608.20569>
   - Introspection study over 8 open models. Random controls are calibrated to match the treatment's Jensen–Shannon divergence; the paper states "norm matching alone systematically understates the control".
   - Impact matching succeeded for one model only (Qwen2.5-7B-Instruct). The others used unit-norm random directions.
   - **Covers:** (d), as prior art for divergence matching. Not a direction-validity study.
8. ✓ **Reproducing steering against evaluation awareness in a large open-weight model.** Read, Schoen, Aranguri, Bloom (UK AISI). LessWrong, 10 Apr 2026. <https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1>
   - On GLM-5, control vectors from unrelated contrast pairs ("book on the top vs bottom shelf") changed misaligned-action rates as much as the evaluation-awareness vectors, up to about 80%.
   - Strong motivation for (c) and (d). The model is far above 14B.
9. ✓ **What Do Compliance Detectors Read?** Sadhu, Sengupta, Sankarapu, Seth. arXiv:2608.16852, Aug 2026. <https://arxiv.org/abs/2608.16852>
   - Guard models and activation probes ignore the governing rule ("rule blindness").
   - A preregistered test of the authors' activation readout fails, because a bag-of-words model matches it. This is precedent for the text-only baseline in Check B.
10. ✓ **The Impact of Off-Policy Training Data on Probe Generalisation.** Kirch, Dower, Skapars, Yannakoudakis, Lubana, Krasheninnikov. arXiv:2511.17408, ACL 2026. <https://arxiv.org/abs/2511.17408>
    - Across eight behaviours, probes trained on synthetic or off-policy responses generalise worse than those trained on what the monitored model actually did, especially for intent-defined behaviours such as deception.
    - Precedent for (a) on the probe-training side.
11. ✓ **LLMs Encode Harmfulness and Refusal Separately.** Zhao, Huang, Wu, Bau, Shi. arXiv:2507.11878, 2025 (v5 Jul 2026). <https://arxiv.org/abs/2507.11878>
    - A harmfulness direction distinct from Arditi et al.'s refusal direction. Steering harmfulness changes the model's judgement; steering refusal changes only refusal.
    - This is why the battery's refusal control uses *actual refusal* as `T`, not the harmfulness label (see `PREREGISTRATION_DRAFT.md` §6).
12. ✓ **On the Failure of Topic-Matched Contrast Baselines in Multi-Directional Refusal Abliteration.** Petrov. arXiv:2603.22061, Mar 2026. <https://arxiv.org/abs/2603.22061>
    - On Qwen 3.5 2B, refusal directions *built* from topic-matched harmful/harmless pairs fail to steer, because subtraction cancels the shared topic component.
    - Design caveat for (b): use matched sets to *evaluate* published directions, not to rebuild them. The draft follows this.
13. ✓ **Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?** Korznikov et al. arXiv:2602.14111, Feb 2026. <https://arxiv.org/abs/2602.14111>
    - Random baselines match trained SAEs on interpretability, sparse probing and causal editing.
    - Precedent for the Adebayo-style framing and the title, for SAEs rather than directions.
14. ✓ **On the Non-Identifiability of Steering Vectors in LLMs.** Venkatesh, Kurapath. arXiv:2602.06801, 2026. <https://arxiv.org/abs/2602.06801>
    - Many orthogonal perturbations give indistinguishable behaviour. Conceptual support for not reading a steering effect as identifying "the" representation.
15. ✓ **Same Outcome, Different Readout.** Li et al. arXiv:2609.22850, Sep 2026. <https://arxiv.org/abs/2609.22850>
    - Construct-validity framing for one valence direction in a maze task. Decodability and steerability are shown to be insufficient for identification.
    - Closest in *framing* ("what does the direction represent?"), but one direction.
16. ✓ **Position: Mechanistic Interpretability Must Disclose Identification Assumptions for Causal Claims.** Lin, Liu. arXiv:2605.08012, May 2026. <https://arxiv.org/abs/2605.08012>
    - Audits 10 (and 30) papers for missing identification assumptions and proposes a disclosure norm.
    - Motivation; runs no checks.
17. ✓ **Contrastive Covariance Subspaces for Activation Steering.** Xenos et al. arXiv:2610.14025, Sep/Oct 2026. <https://arxiv.org/abs/2610.14025>
    - A *method* that projects difference-of-means vectors away from directions shared with other classes. It is evidence that nuisance variation inside difference-of-means vectors is now recognised, but it is a fix, not an audit.
18. ✓ **Beyond Fixed Directions: Adaptive Representation Analysis of Reasoning and Memorization in LLMs.** Nabi. arXiv:2608.21919, Aug 2026. <https://arxiv.org/abs/2608.21919>
    - On Qwen3-0.6B, a reasoning/recall direction (AUROC 1.00) rotates substantially after GRPO (cosine about 0.45) while staying decodable.
    - Relevant to D3 and D4.

Also relevant (~, not opened by the assistant; reported by the search agent):
- SteeringControl / SteeringSafety (Siu et al. 2025, arXiv:2509.13450): a benchmark of steering methods with entanglement measures, on Qwen2.5-7B and Llama-3.1-8B.
- FaithSteer-BENCH (arXiv:2603.18329).
- RePS and HyperSteer, follow-ups to AxBench.
- Moskvoretskii et al. 2026 (arXiv:2605.13329, ✓ abstract): persona vectors traced through OLMo-3 pretraining. The search agent's claim that it uses label-shuffle baselines is *not* confirmed by the abstract.
- Kumar 2026, "Pressure-Testing Deception Probes" (arXiv:2605.27958): a style confound.
- Goldowsky-Dill et al. 2025, Apollo deception probes (arXiv:2502.03407): labels from graded actual responses, with an Alpaca control set.

## 5. Novelty verdict, in detail

**Is it already done? No, but most of its parts are.** An honest framing for the paper:
- **Not new; cite and adopt:**
  - KL- or divergence-matched random controls: SteerCheck; Ferrara.
  - Random and label-shuffled nulls: Hewitt & Liang; Adebayo et al.; SteerCheck.
  - Layer sweeps.
  - Format deconfounding of a single probe: Sahoo et al.; Devbunova.
  - Multi-claim audits: Wu et al., for static "AI neuroscience" claims.
  - Text-only baselines: Sadhu et al.
  - Direct-logit equivalence: Yuan & Vlachos.
- **Probably new, and the paper's claims:**
  1. **Process-defined targets as the default test.** The battery asks whether the published direction follows *what the model does* on each item, under matched format, rather than whether it separates the prompt types it was built from. No audit found does this across directions.
  2. **The label-free structured nuisance direction from the same calibration prompts as a competitor.** Paper 4's within-class length direction beat the CR axis at tracking settling depth (ρ = +0.13 to +0.32 against −0.03 to +0.07) and at separating familiar from novel problems (AUC about 0.85 against about 0.63), and matched it at toggling reply format. Nothing comparable was found for other published directions.
  3. **Known-answer validation of the battery:** a positive control (refusal), oracle directions per target, a known-failure case (the CR axis, from paper 4) and a randomised-model check. These say when a failure is informative. SteerCheck has positive controls for its own audit, but not oracle directions per target.
  4. **Breadth on reasoning-process claims.** LiReFs, backtracking, reasoning strength and true thinking have not been re-tested by anyone, as far as the searches found. That includes no length-matched re-test of the backtracking vectors and none of the reasoning-strength vector.
- **Not claimable:** "first to show steering vectors can be confounded" (Tan et al. 2024; Sahoo et al.; Gao et al.; AISI), and "first KL-matched controls" (SteerCheck; Ferrara; papers 3 and 4 of this series used them too).

**Design lessons from the literature, already applied in the draft:**
- Petrov: never rebuild directions from topic-matched sets; only evaluate on them.
- SteerCheck's alignment leakage: report cos(N_perm, d) and add a low-cosine subset as a sensitivity analysis.
- Zhao et al.: refusal and harmfulness are separate, so the positive control's target is actual refusal, and the label check (B) is reported separately from the process verdict.
- ITI's Table 3 and SteerCheck's argument (§2.14): equal-norm random directions can move the output an order of magnitude less than the tested direction. Hence KL matching, the permuted-label (in-distribution) second reference in Check D, and H4. Heimersheim & Mendel's "sensitive directions" claim is cited only with its later caveat (Lee & Heimersheim 2024).
- Gao et al. and Yuan & Vlachos: late, output-sensitive components can carry the whole effect. Hence Check E2 (direct path) and the OUTPUT-BIAS flag.

## 6. Candidate directions (code verified by the candidate search; reproducible on open models ≤14B)

"Repo" is what the repository page showed when the search agent opened it. Unless marked otherwise, the authors used dataset or prompt-type labels. The table's "process-defined target" column is the target proposed for the battery.

| # | Direction | Paper / code | Built from | Claimed process | Paper's ground truth | Process-defined target for the battery | Main nuisance in calibration |
|---|---|---|---|---|---|---|---|
| 1 | **Refusal** (positive control) | Arditi et al., NeurIPS 2024, <https://arxiv.org/abs/2406.11717>; <https://github.com/andyrdt/refusal_direction> (Apache-2.0; pipeline for Qwen-Chat 1.8–14B, Gemma, Yi, Llama-2/3) | difference of means harmful − harmless at post-instruction template tokens; layer chosen by bypass/induce score plus a KL filter, excluding the top 20% of layers | refusal "mediated by a one-dimensional subspace" | refusal-phrase match plus Llama Guard 2 | actual refusal per completion, on XSTest/OR-Bench items where refusal and harmfulness come apart | phrasing and length of AdvBench-style vs Alpaca prompts; read on template tokens |
| 2 | **Truth** (static-concept control) | Marks & Tegmark, COLM 2024, <https://arxiv.org/abs/2310.06824>; <https://github.com/saprmarks/geometry-of-truth>. Bürger et al., NeurIPS 2024, <https://arxiv.org/abs/2407.12831>; <https://github.com/sciai-lab/Truth_is_Universal> | mass-mean (difference of means) on curated true/false statements | a linear "truth" direction that causally affects true/false outputs | dataset label; "likely" set as a probability control | the model's own true/false verdict on each statement, including the items it gets wrong | templates, negation, numerals; text probability |
| 3 | **Compute–retrieve axis** (known-answer case) | Ramdan 2026a, doi:10.5281/zenodo.19455724; <https://github.com/Somebodyhere101/compute-retrieve> (commit `ce0c61f` in `third_party/`) | difference of means between 30 GSM8K word problems and 30 MMLU questions; last token; best of 4 layers | compute vs retrieve "mode" | prompt type | logit-lens settling depth and shown working on format-matched multiplication (paper 4) | word problem vs multiple choice, numerals, length (compute prompts are longer) |
| 4 | **LiReFs, reasoning vs memorisation** | Hong et al., ACL 2025 Findings, <https://arxiv.org/abs/2503.23084>; <https://github.com/yihuaihong/Linear_Reasoning_Features> (notebooks, `dataset.zip`, intervention script; no vectors) | difference of means at the last user-turn token; GSM8K/MGSM/MMLU-Pro-R vs PopQA/C-Eval-H/MMLU-Pro-M; middle layer | features that "capture general reasoning capability" | dataset labels plus a GPT-4o reasoning score *of the question* | settling depth (paper 4 items) and per-item CoT necessity (accuracy with minus without CoT) | dataset identity, MC vs open answer, language, length |
| 5 | **Backtracking** | Ward et al., ICML 2025 workshop, <https://arxiv.org/abs/2507.12638>; <https://github.com/jnward/latent-backtracking> (**released** `base_steering_vectors.pt`, `ft_steering_vectors.pt`, annotated chains). Venhoff et al., <https://arxiv.org/abs/2506.18167>; <https://github.com/cvenhoff/steering-thinking-llms> | difference of means at offsets before backtracking sentences (Ward: Llama-3.1-8B base, layer 10, used to steer R1-Distill-Llama-8B); Venhoff: span vs overall mean, layer by attribution patching | behaviours "mediated by linear directions" (Venhoff); fine-tuning "repurposes pre-existing representations" (Ward) | sentences labelled by GPT-4o; keyword fractions ("Wait"/"Hmm") | judged backtrack in the *next* sentence of the model's own chain of thought; backtracks per 1,000 thinking tokens under a push | keyword tokens inside the span; contrast against the overall mean; position in the chain. Ward et al. note that Gaussian noise also raised "Wait" |
| 6 | **Reasoning strength** | Sheng et al., NeurIPS 2025, <https://arxiv.org/abs/2506.08390>; <https://github.com/AlphaLab-USTC/LRM-plans-CoT> (steering scripts; user supplies the vector) | difference of means between MATH difficulty levels 2–5 and level 1, at the `<think>`/last prompt token | LRMs "pre-plan the reasoning strengths … before generation" | difficulty labels (vector); token counts (separate probe) | actual thinking-token count, partialling out difficulty, question length and subject | difficulty ↔ question length ↔ LaTeX density |
| 7 | **Known-entity awareness** | Ferrando et al., ICLR 2025, <https://arxiv.org/abs/2411.14257>; <https://github.com/javiferran/sae_entities> | SAE latents in the paper (Gemma-2-2B/9B; Llama-3.1-8B in Appendix Q); a difference-of-means version from the same labels and position is straightforward | knowledge awareness with a causal effect on refusal and hallucination | **already behavioural**: known if ≥2 attributes are correct, unknown if all are wrong | attribute actually recalled (correctness; logit-lens rank at the entity token), including the discarded middle items | entity popularity, number of tokens, entity type |
| 8 | **Persona vector: sycophancy** | Chen et al. 2025, <https://arxiv.org/abs/2507.21509>; <https://github.com/safety-research/persona_vectors> (trait data, no vectors; judge is gpt-4.1-mini) | 5 contrastive system-prompt pairs, rollouts filtered by judge, difference of means averaged over response tokens | prompt-token projections "correlate strongly with trait expression" (weaker within a prompt type, per the authors) | judge scores of the responses | answer actually flips to the user's stated wrong belief on known-answer items (exact match, no judge) | instruction present in context; response-averaged activations read the response text |
| 9 | **True thinking** | Zhao, Sun, Shi, Song 2025, <https://arxiv.org/abs/2510.24941>; <https://github.com/andotalao24/Identify_true_decorative_thinking> (example directions in `data/pt`) | difference of means, steps with True Thinking Score ≥ 0.9 minus ≤ 0; layers 15–22; R1-Distill-Qwen-7B and Llama-8B | models "often verbalize reasoning steps without actually performing them internally" | **already causal**: each step's measured effect on the answer | True Thinking Score per step on matched steps (tests (b)–(e)) | only numeric steps are scored; position; verification phrasing |
| 10 | **Test awareness** (secondary: high cost) | Abdelnabi & Salem, NeurIPS 2025, <https://arxiv.org/abs/2505.14617>; <https://github.com/microsoft/Test_Awareness_Steering> (probes released for 32B only) | probe on hidden states averaged over CoT spans flagged as evaluation-aware; QwQ-32B, Qwen3-32B, R1-Distill-Qwen-32B; needs a port to ≤14B | test awareness changes compliance | judge-flagged spans | later verbalised test awareness, and the compliance gap between real and hypothetical versions | spans literally say "this may be a test"; ".mock" and placeholder cues |

**Alternates (code verified by the search agent):**
- Sentiment: Tigges et al. 2024, <https://arxiv.org/abs/2310.15154>; <https://github.com/curt-tigges/eliciting-latent-sentiment>.
- ThinkEdit: <https://arxiv.org/abs/2503.22048>; <https://github.com/Trustworthy-ML-Lab/ThinkEdit>.
- SEAL: <https://arxiv.org/abs/2504.07986>; <https://github.com/VITA-Group/SEAL>.
- In-advance correctness: Cencerrado et al., <https://arxiv.org/abs/2509.10625>; <https://github.com/ivanvmoreno/correctness-model-internals>.
- Thinking speed: Lin et al., <https://arxiv.org/abs/2507.03704>; <https://github.com/D2I-ai/thinking-speed-control>.
- Lying: Huan et al., <https://arxiv.org/abs/2509.03518>; <https://github.com/yzhhr/llm-liar>.
- CAA: Rimsky et al., <https://arxiv.org/abs/2312.06681>; <https://github.com/nrimsky/CAA> (vectors released, Llama 2 only; the A/B-letter format is a known confound per Tan et al.).
- ITI (<https://github.com/likenneth/honest_llama>): not recommended, because it needs a fine-tuned GPT judge.
- Dropped:
  - Lindsey's 2025 introspection work: no official code.
  - Apollo deception probes: main model 70B, and small models rarely deceive.

**Recommended order of work:**
1. Refusal and the CR axis first: they validate the battery code against known answers.
2. LiReFs next: it shares paper 4's items and targets, so it gives a direct head-to-head with the CR axis.
3. Then backtracking (released vectors) and reasoning strength: the most testable process claims.
4. Then known-entity awareness, sycophancy and true thinking.
5. Test awareness last, if at all.

## 7. This series (for citation in paper 6)

- **Paper 1.** Ramdan 2026a, compute–retrieve axis. doi:10.5281/zenodo.19455724; code <https://github.com/Somebodyhere101/compute-retrieve>.
- **Paper 2.** Ramdan 2026b, "A Compounding Fingerprint: Telling Abstraction from Exemplar Reuse in Context". doi:10.5281/zenodo.23318679. Experiments 5 and 5b give the CR-axis reversal under matched controls.
- **Paper 3.** Injected-thought reports in Qwen3 1.7B–14B run through a content-free yes-bias direction; matched-KL controls; stage analysis. Results page in the session folder `effort_jspace/`. Not yet on Zenodo as far as this folder shows.
- **Paper 4.** Preregistered causal test of the CR axis (`cr4/`). The CR axis does not track settling depth. The within-class length direction from the same calibration prompts tracks it better and matches the CR axis at toggling reply format. This is the worked example for checks (a)–(d).
- **Paper 5.** Chosen effort in the Ouro looped model (`loop5/`). Its H4 and H5 apply the CR axis to a model-defined effort target, a further process-defined target.

Inline links throughout this file serve as the reference list. Before submission, every ~ item must be opened and checked, and the post-cutoff (26xx) items read in full, not just their abstracts.
