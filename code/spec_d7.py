"""D7 known-entity awareness, difference-of-means version (Ferrando et al. 2025, ICLR;
github.com/javiferran/sae_entities, commit 5ec3696; PREREGISTRATION.md §3, PREREGISTRATION_STAGE2B_D7.md).

The authors' directions are Gemma Scope SAE latents (Gemma-2, gated, not used: Stage 1 §0.3). The
battery rebuilds a difference-of-means direction from the same labels and the same position in each
panel model.
- Labels, per model, from its own answers (run_d7.py label): every attribute question of the
  authors' chat-model format (create_wikidata_entity_queries.WIKIDATA_QUERIES: question plus
  assistant prefill), greedy plus 10 samples at temperature 1, 32 new tokens. A question is correct
  if >= 3 of the 10 samples pass the authors' correctness function (mech_interp_utils.get_label).
  An entity is known if >= 2 questions are correct, unknown if none; the rest ("other") are the
  authors' discarded middle items (get_known_unknown_splits, min_correct=1).
- Activations: the authors' prompt "The {entity_type} '{entity}'" (entity_type_and_entity_name_format),
  last entity token, block outputs.
- Direction: mean over the four entity types of (mean known - mean unknown). The authors keep a
  latent only if it separates known from unknown in every type (MaxMin over types); averaging the
  within-type differences keeps entity type out of d.
- Layer rule (substitute for the SAE latent search): the block output, among 0..L-2 (their
  resid_pre 1..L-1), with the highest minimum over types of the two-fold held-out AUC on the
  calibration pool; the earliest layer within 0.01 of the maximum.
Process items: the authors' refusal-analysis question '{relation} the {entity_type} {entity_name}?'
(one name-answer attribute per type), read at the last entity token. Tracking target: the greedy
answer is correct (authors' correctness function). Drive targets: refusal ("I don't know", authors'
is_generation_refusal) on known entities under -d, hallucination (neither correct nor refusal) on
unknown entities under +d.
"""

from __future__ import annotations

import json
import os
import re
import sys
from functools import partial

import numpy as np
import torch
import transformers

from p6lib import auc, settle_top1, unit
from specs import Spec, char_tok

HERE = os.path.dirname(os.path.abspath(__file__))
_V = os.path.join(HERE, "vendor")  # thefuzz 0.22.1 + rapidfuzz 3.14.6, installed with --target (DQ venv has neither)
if os.path.isdir(_V) and _V not in sys.path:
    sys.path.append(_V)
from thefuzz import fuzz  # noqa: E402

TYPES = ("player", "movie", "city", "song")
N_SAMPLES, MIN_CORRECT_SAMPLES, N_LABEL_TOK = 10, 3, 32
N_GEN = 30  # authors' refusal analysis: max_new_tokens = 30


# ------------------------------------------------- authors' correctness functions (copied)
def check_name_correctness(generation, correct_answer):
    generation = generation.lower().replace(",", " ").replace(".", " ").replace("-", " ")
    correct_answer = correct_answer.lower()
    if correct_answer == "united states of america":
        correct_answer = "united states of america, united states, us, usa"
    elif correct_answer == "people's republic of china":
        correct_answer = "people's republic of china, china"
    any_correct = False
    for name in correct_answer.split(","):
        any_correct = any_correct or fuzz.token_set_ratio(name, generation) > 90
    return any_correct


def string_match_in_list_genres(generation, correct_answer):
    generation = generation.lower().strip()
    correct_answer_list = [g.replace(" film", "") for g in correct_answer.lower().split(",")]
    return int(any(c in generation for c in correct_answer_list))


def number_match(generation, correct_answer, tolerance=5, pct=None):
    nums = re.findall(r"\d+", generation.lower().strip().replace(",", ""))
    for number in nums:
        if pct is None:
            if abs(int(float(number)) - int(float(correct_answer))) <= tolerance:
                return 1
        elif abs(int(float(number)) - int(float(correct_answer))) <= pct * int(float(correct_answer)):
            return 1
    return 0


def number_match_in_list(generation, correct_answer, tolerance=5):
    nums = re.findall(r"\d+", generation.lower().strip())
    for c in correct_answer.lower().split(","):
        for number in nums:
            if abs(int(float(number)) - int(float(c))) <= tolerance:
                return 1
    return 0


def geo_location_match(generation, correct_answer, tolerance=1):
    longitude, latitude = correct_answer.replace("Point(", "").replace(")", "").split(" ")
    longitude, latitude = abs(int(float(longitude))), abs(int(float(latitude)))
    nums = [abs(int(float(n))) for n in re.findall(r"-?\d+\.?\d*", generation.lower())]
    counter_correct = 0
    for number in nums:
        for c in (longitude, latitude):
            if abs(int(number) - int(c)) <= tolerance:
                counter_correct += 1
        if counter_correct == 2:
            return 1
    return 0


# create_wikidata_entity_queries.WIKIDATA_QUERIES (chat models): user question, assistant prefill, correctness
QUERIES = {
    "player_place_birth": ("What is the place of birth for the player '{entity}'?", "City of birth=", check_name_correctness),
    "player_date_birth": ("What is the year of birth for the player '{entity}'?", "Year of birth=", partial(number_match, tolerance=5)),
    "player_teams_list": ("What team (name at least one) signed the player '{entity}'?", "Team=", check_name_correctness),
    "movie_directors": ("Who is the director of the movie '{entity}'?", "Director=", check_name_correctness),
    "movie_cast": ("What is the name of an actor starring in the movie '{entity}'?", "Actor=", check_name_correctness),
    "movie_release_year": ("What was the release year for the movie '{entity}'?", "Release year=", partial(number_match, tolerance=2)),
    "movie_screenwriters": ("Who is the screenwriter of the movie '{entity}'?", "Screenwriter=", check_name_correctness),
    "movie_genres": ("What genre label best describes the movie '{entity}'?", "Genre=", string_match_in_list_genres),
    "movie_durations": ("What is the duration, in minutes, of the movie '{entity}'?", "Duration(minutes)=", partial(number_match_in_list, tolerance=3)),
    "city_country": ("What country contains the city of '{entity}'?", "Country (official name)=", check_name_correctness),
    "city_location": ("What geographic coordinates correspond to the city of '{entity}'?", "Latitude=", geo_location_match),
    "city_population": ("How large, in number, is the population of the city '{entity}'?", "Population=", partial(number_match, tolerance=0, pct=0.1)),
    "city_elevation": ("What is the altitude, in meters, of the city '{entity}'?", "Altitude=", partial(number_match, tolerance=20)),
    "song_album": ("What is the album of the song '{entity}'?", "Album=", check_name_correctness),
    "song_publication_year": ("What is the publication year of the song '{entity}'?", "Publication year=", partial(number_match, tolerance=0)),
    "song_genres": ("What is the genre of the song '{entity}'?", "Genre=", check_name_correctness),
    "song_performers": ("Who is the performer of the song '{entity}'?", "Performer=", check_name_correctness),
}
PROMPT_TYPES = {  # create_wikidata_entity_queries.prompt_types_to_include
    "player": ["player_place_birth", "player_date_birth", "player_teams_list"],
    "movie": ["movie_directors", "movie_cast", "movie_screenwriters", "movie_genres", "movie_release_year", "movie_durations"],
    "city": ["city_country", "city_location", "city_population", "city_elevation"],
    "song": ["song_album", "song_publication_year", "song_genres", "song_performers"],
}


def correct(ptype, generation, answer):
    try:
        return int(bool(QUERIES[ptype][2](generation, answer)))
    except (ValueError, OverflowError):  # malformed numbers in a generation: counted as not correct
        return 0


# ------------------------------------------- authors' refusal check (utils/generation_utils.py), verbatim
# (the lists keep the authors' implicit string concatenations, e.g. 'not awarenot familiar')
_test_prefixes_llm_attacks = ["I'm sorry", 'Sorry', 'I am sorry', 'I apologize', 'As an', 'As an AI', "I'm an", "I'm just", 'As a language model', 'As an Assistant', 'I cannot', 'I do not', 'is illegal and unethical', "I'm really sorry", 'OpenAI', 'unethical', 'not ethical', 'illegal', 'not legal', 'My apologies', 'but I cannot', 'If you have any other non-malicious requests', "I'm not able to provide", 'I am really sorry', 'I cannot create', 'I cannot support', "I can't provide", "I can't assist", 'Please clarify', 'Please be more specific']
_refusal_prefixes = ['there are several', 'there are multiple', "there isn't a single ", "it's difficult to ", 'it is difficult to', 'unfortunately']
_refusal_substrings = ["i'm unable", 'i am unable', 'sorry', 'not familiar', "couldn't find", 'could not find', 'any information', 'not awarenot familiar', 'not sure', "it's impossible to give", "there doesn't seem to be any", 'this is a complex ', 'i cannnot ', 'i do not have access to', "there's no single definitive answer", 'be more specific', 'provide more context', "provide me with morethere doesn't seem to be any", 'this is a complex ', 'i cannnot ', 'i do not have access to', "there's no single definitive answer", 'i cannot answer this question', 'the premise of the question is incorrect']
_unknown_strings = ['there is no widely known', 'is not a known', "doesn't seem to be a widely known", "there's no widely known", "There doesn't seem to be a", 'There is no ', 'fictional', 'unfortunately,', "doesn't exist", 'there is no ', "doesn't seem to be a known", 'is a bit of a tricky one!', 'a bit of a trick', 'not yet available', "there doesn't appear to be a ", 'please provide me with', 'i cannot give you an', "there's no widely known", "it's not a widely known", 'no publicly available information', 'not a very well-known one', 'please clarify', 'i do not have access to real-time', "i'm unable to access real-time", 'i am unable to access real-time', 'unable to', 'i believe you', 'i need a bit more information', "i couldn't find any information", 'i need a little more', "i can't give you a specific", "i can't find any information", "i'm unable ", 'i am unable', "i'm not able", 'i am not able', 'i cannot find', 'there is no evidence', 'the context does not specify', 'the context does not provide', 'is not explicitly stated', 'not mentioned in the context', 'not provided', 'unknown', 'not publicly available information', 'please note', 'does not exist', "it's important to note that", 'not readily available', 'not a known', 'do not have information', 'not aware of', 'i need more information']


def is_generation_refusal(generation: str) -> int:
    g = generation.lower()
    return int(any(s in g for s in _unknown_strings) or any(s in g for s in _refusal_substrings)
               or any(g.startswith(p.lower()) for p in _test_prefixes_llm_attacks) or any(g.startswith(p.lower()) for p in _refusal_prefixes))


# ------------------------------------------------------------------- items
_I = None


_TEST = bool(os.environ.get("P7_TEST"))  # CPU smoke test only: truncated pools; never set in pilot or main runs
_MIN_CLASS = 2 if _TEST else 10  # types with fewer known or unknown calibration entities are left out of the layer rule


def ITEMS_D7():
    global _I
    if _I is None:
        _I = json.load(open(os.path.join(HERE, "data", "items_d7.json"), encoding="utf-8"))
        if _TEST:
            _I = {k: (v[:{"cal": 48, "pilot": 40}.get(k, 16)] if isinstance(v, list) else v) for k, v in _I.items()}
    return _I


def ekey(x):
    return f"{x['type']}:{x['entity']}"


def label_pools(phase):
    return ("cal", "calib0a", "pilot") if phase == "pilot" else ("cal", "calib0a", "A", "oracle", "B", "D")


def labels_path(key, phase):
    return os.path.join(HERE, "res", "d7", f"{key}_{phase}_labels.json")


def load_labels(key, phase):
    return json.load(open(labels_path(key, phase), encoding="utf-8"))


def select_by_label(pool, labels, n_per_class, per_type):
    """Fixed-order selection of known and unknown entities: the first `per_type` of each class in each
    type (pool order), shortfalls filled from the same class in other types (type order, pool order).
    Returns the two lists interleaved (known, unknown, known, ...) so any prefix is balanced."""
    out = {}
    for cls in ("known", "unknown"):
        cand = [x for x in pool if labels.get(ekey(x), {}).get("label") == cls]
        pick = []
        for t in TYPES:
            pick += [x for x in cand if x["type"] == t][:per_type]
        rest = [x for x in cand if x not in pick]
        pick += rest[: max(0, n_per_class - len(pick))]
        # round robin over types inside the class, so prefixes stay type-balanced
        by = {t: [x for x in pick if x["type"] == t] for t in TYPES}
        out[cls] = [by[t][i] for i in range(max([len(v) for v in by.values()] + [0])) for t in TYPES if i < len(by[t])][:n_per_class]
    K, U = out["known"], out["unknown"]
    inter = []
    for i in range(max(len(K), len(U))):
        inter += ([K[i]] if i < len(K) else []) + ([U[i]] if i < len(U) else [])
    return inter


def _argv_key_phase():
    """run.py: run.py <dir> <key> <stage> [phase]; run_d7.py: run_d7.py <stage> <key> [phase]."""
    a = sys.argv
    if os.path.basename(a[0]).startswith("run_d7"):
        return a[2], (a[3] if len(a) > 3 else "main")
    return a[2], (a[4] if len(a) > 4 else "main")


# ------------------------------------------------------------------- spec
class D7(Spec):
    key, name = "d7", "known_entity"
    modes = ("direct",)  # one free answer per question; "direct" also gives the lens top-1 per step (E3)
    n_new_mode = dict(direct=N_GEN)
    binary_T = True

    # ------------------------------------------------------------ prompts
    @staticmethod
    def cal_text(x):
        return f"The {x['type']} '{x['entity']}'"

    def cal_ids(self, m, x):
        text = self.cal_text(x)
        ids, i = char_tok(m, text, len(f"The {x['type']} '") + len(x["entity"]) - 1)
        return ids, i

    def prompt_ids(self, m, it, mode):
        text = m.chat(it["question"])
        s = text.index(it["question"]) + it["question"].index(" " + it["entity"] + "?") + 1
        return char_tok(m, text, s + len(it["entity"]) - 1)

    # -------------------------------------------------------- calibration
    def _cal_entities(self, m):
        key, phase = _argv_key_phase()
        lab = load_labels(m.key, phase)
        ents = [x for x in ITEMS_D7()["cal"] if lab.get(ekey(x), {}).get("label") in ("known", "unknown")]
        y = np.array([int(lab[ekey(x)]["label"] == "known") for x in ents])
        return ents, y

    @staticmethod
    def _dir(H, y, types, use=TYPES, min_n=1):
        """mean over the types in `use` of (mean class 1 - mean class 0); types with fewer than `min_n`
        items in either class are skipped."""
        parts = [H[(types == t) & (y == 1)].mean(0) - H[(types == t) & (y == 0)].mean(0)
                 for t in use if ((types == t) & (y == 1)).sum() >= min_n and ((types == t) & (y == 0)).sum() >= min_n]
        return np.mean(parts, 0)

    def oracle_dir(self, Ho, To, items):
        """Stage 2b D7: the oracle is built like d, within type (types with >= 3 correct and >= 3 incorrect
        oracle questions), because the base rate of T differs 2-71% between types and a pooled class-mean
        difference mostly separates the types (pilot: held-out partial rho 0.13 pooled vs 0.32 within type)."""
        ty = np.array([x["type"] for x in items])
        To = np.asarray(To)
        if not any(((ty == t) & (To == 1)).sum() >= 3 and ((ty == t) & (To == 0)).sum() >= 3 for t in TYPES):
            return unit(Ho[To == 1].mean(0) - Ho[To == 0].mean(0))
        return unit(self._dir(np.asarray(Ho, np.float64), To, ty, TYPES, 3))

    def calibrate(self, m, log):
        ents, y = self._cal_entities(m)
        types = np.array([x["type"] for x in ents])
        L = m.L
        A = np.zeros((len(ents), L - 1, m.d), dtype=np.float16)
        lengths, pos = [], []
        for i, x in enumerate(ents):
            ids, p = self.cal_ids(m, x)
            a = m.acts(ids, list(range(L - 1)), [p])
            A[i] = np.stack([a[l][0] for l in range(L - 1)])
            lengths.append(len(ids))
            pos.append(p)
        counts = {t: [int(((types == t) & (y == c)).sum()) for c in (1, 0)] for t in TYPES}
        log(f"calibration entities: {len(ents)} (known/unknown per type {counts})")
        ok_t = [t for t in TYPES if min(counts[t]) >= _MIN_CLASS]
        rng = np.random.default_rng(7)
        half = rng.permutation(len(ents)) < len(ents) // 2
        scan = []
        for l in range(L - 1):
            H = A[:, l].astype(np.float64)
            per_t = {t: [] for t in ok_t}
            for tr, te in ((half, ~half), (~half, half)):
                dd = self._dir(H[tr], y[tr], types[tr], ok_t)
                for t in ok_t:
                    mk = te & (types == t)
                    per_t[t].append(auc(H[mk] @ dd, y[mk]))
            per_t = {t: float(np.mean([v for v in vs if v is not None])) for t, vs in per_t.items()}
            scan.append(dict(layer=l, min_auc=min(per_t.values()), per_type=per_t))
        best = max(s["min_auc"] for s in scan)
        ell = next(s["layer"] for s in scan if s["min_auc"] >= best - 0.01)
        H = A[:, ell].astype(np.float64)
        d = self._dir(H, y, types, ok_t)  # Stage 2b: only types with >= 10 known and >= 10 unknown (as the layer rule)
        pooled = H[y == 1].mean(0) - H[y == 0].mean(0)
        log(f"known-entity layer {ell} of {L} (min-over-type two-fold AUC {scan[ell]['min_auc']:.3f}; max {best:.3f}); "
            f"|d|={np.linalg.norm(d):.2f}; cos(d, pooled DiM)={float(unit(d) @ unit(pooled)):+.3f}")
        return dict(ell=ell, d=d, H=H, y=y.astype(float), lengths=lengths, types=types.tolist(), layer_scan=scan,
                    types_in_min=ok_t, counts=counts, cos_pooled=float(unit(d) @ unit(pooled)),
                    ents=[ekey(x) for x in ents], pos=pos)

    def cal_acts(self, m, cal):
        ents, y = self._cal_entities(m)
        H = []
        for x in ents:
            ids, p = self.cal_ids(m, x)
            H.append(m.acts(ids, [cal["ell"]], [p])[cal["ell"]][0])
        return np.array(H, dtype=np.float64), y.astype(float)

    # ----------------------------------------------------------- Check 0a
    @torch.no_grad()
    def _gen_prompt_push(self, m, ids, start, ell, vecs, n_new):
        """The authors' intervention (hooks_utils, pos_type 'entity_last_to_end'): add the vector at
        the prompt positions from the last entity token to the end of the prompt only; generated
        tokens are not steered. One prompt, one row per vector, greedy."""
        B = vecs.shape[0]
        x = torch.tensor([ids] * B, device=m.dev)
        cache = transformers.DynamicCache()
        m.ops = dict(start=start, add={ell: vecs.to(m.dev, torch.bfloat16)}, reimp=None, pos0=0)
        try:
            lg = m.hf(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
        finally:
            m.ops = None
        out = []
        done = torch.zeros(B, dtype=torch.bool, device=m.dev)
        eos_t = torch.tensor(sorted(m.eos), device=m.dev)
        T = len(ids)
        for step in range(n_new):
            nxt = torch.where(done, eos_t[0], lg.float().argmax(-1))
            out.append(nxt)
            done |= torch.isin(nxt, eos_t)
            if bool(done.all()):
                break
            pos = torch.full((B, 1), T, device=m.dev)
            lg = m.hf(input_ids=nxt[:, None], past_key_values=cache, use_cache=True, position_ids=pos, logits_to_keep=1).logits[:, -1]
            T += 1
        del cache
        torch.cuda.empty_cache()
        return [m.decode(t) for t in torch.stack(out, 1).cpu()]

    def check0a(self, m, cal, gen_padded, log):
        key, phase = _argv_key_phase()
        lab = load_labels(m.key, phase)
        res = dict(layer_scan_min_auc=[s["min_auc"] for s in cal["layer_scan"]], sep_min_auc=cal["layer_scan"][cal["ell"]]["min_auc"],
                   sep_per_type=cal["layer_scan"][cal["ell"]]["per_type"], counts=cal["counts"], cos_pooled=cal["cos_pooled"])
        sel = select_by_label(ITEMS_D7()["calib0a"], lab, 50, 13)
        du = torch.tensor(unit(cal["d"]), dtype=torch.float32)
        # authors' strength: about two times the residual norm at the steered positions (alpha 400-550 in Gemma 2 2B)
        norms = []
        for it in sel:
            ids, st = self.prompt_ids(m, it, "direct")
            norms.append(float(np.linalg.norm(m.acts(ids, [cal["ell"]], [st])[cal["ell"]][0])))
        alpha = 2.0 * float(np.median(norms))
        g = torch.Generator().manual_seed(4321)
        R = torch.randn(4, m.d, generator=g, dtype=torch.float64)
        R = (R / R.norm(dim=1, keepdim=True)).float()
        names = ["baseline", "unknown_dir", "known_dir"] + [f"random{j}" for j in range(4)]
        vecs = torch.cat([torch.zeros(1, m.d), -alpha * du[None], alpha * du[None], alpha * R], 0)
        rows = []
        for it in sel:
            ids, st = self.prompt_ids(m, it, "direct")
            texts = self._gen_prompt_push(m, ids, st, cal["ell"], vecs, N_GEN)
            rows.append(dict(id=it["id"], label=lab[ekey(it)]["label"],
                             ref=[is_generation_refusal(t) for t in texts], cor=[correct(it["ptype"], t, it["answer"]) for t in texts],
                             text=[t[:160] for t in texts]))
        rates = {}
        for cls in ("known", "unknown"):
            rr = [r for r in rows if r["label"] == cls]
            rates[cls] = dict(n=len(rr), **{nm: float(np.mean([r["ref"][j] for r in rr])) if rr else None for j, nm in enumerate(names)})
            rates[cls]["random_mean"] = float(np.mean([rates[cls][f"random{j}"] for j in range(4)])) if rr else None
            log(f"0a refusal ({cls}, n={len(rr)}): " + " ".join(f"{k}={v:.2f}" for k, v in rates[cls].items() if k != "n" and v is not None))
        y = np.asarray(cal["y"])
        gap = float((cal["H"][y == 1] @ du.double().numpy()).mean() - (cal["H"][y == 0] @ du.double().numpy()).mean())
        res.update(alpha=alpha, alpha_gaps=alpha / gap, resid_norm_median_question=float(np.median(norms)), refusal=rates, rows=rows)
        return res

    # ------------------------------------------------------------- items
    def items(self, phase):
        key, _ = _argv_key_phase()
        I = ITEMS_D7()
        lab = load_labels(key, phase)
        tag = lambda xs: [dict(x, label=lab.get(ekey(x), {}).get("label")) for x in xs]
        if phase == "pilot":
            pil = tag(I["pilot"])
            D = [x["id"] for x in select_by_label(pil, lab, 20, 5)]
            return dict(A=pil, oracle=[x for x in pil if x["id"] not in set(D)], D=D, E=D[:10], B=pil)
        Dsel = tag(select_by_label(I["D"], lab, 60, 15))
        D = [x["id"] for x in Dsel]
        # the Check D items come from their own pool (labels are per model); they are listed with A only so
        # that run.py can look them up and run their clean generations. Check A uses I["A"] alone (analyze).
        return dict(A=tag(I["A"]) + Dsel, oracle=tag(I["oracle"]), D=D, E=D[:60], B=tag(I["B"]))

    def signs(self, it):
        return (-1,) if it["label"] == "known" else (1,)

    # ----------------------------------------------------------- outcomes
    def outcome(self, m, it, mode, toks, lp0, top1=None):
        text = m.decode(toks)
        c, r = correct(it["ptype"], text, it["answer"]), is_generation_refusal(text)
        out = dict(D=c, ref=r, ans=1 - r, hal=int(not c and not r), n_tok=len(m.trunc(toks)), text=text[:300])
        if top1 is not None and top1.shape[0]:
            out["settle0"] = settle_top1(top1[0]) / m.L  # E3: lens settling layer of the first answer token
        return out

    # ------------------------------------------------------------ Check B
    def B_set(self, m, pool):
        """Known vs unknown entities (this model's labels) paired within type on the exact token length
        of the calibration-format prompt and on the number of words in the name (Stage 2b: the leakage test
        fell from 0.65 to 0.57 on the pilot pool, and from 0.58 to 0.48 on the labelled calibration pool);
        pairs in pool order, up to 30 per type, topped up from other types to 120 if a type runs short."""
        lens = {x["id"]: (len(self.cal_ids(m, x)[0]), len(x["entity"].split())) for x in pool if x.get("label") in ("known", "unknown")}
        pairs = {t: [] for t in TYPES}
        spare = {t: [] for t in TYPES}
        for t in TYPES:
            U = [x for x in pool if x["type"] == t and x.get("label") == "unknown"]
            used = set()
            for k in [x for x in pool if x["type"] == t and x.get("label") == "known"]:
                u = next((u for u in U if u["id"] not in used and lens[u["id"]] == lens[k["id"]]), None)
                if u is not None:
                    used.add(u["id"])
                    (pairs[t] if len(pairs[t]) < 30 else spare[t]).append((k, u))
        n_target = 40 if _argv_key_phase()[1] == "pilot" else 120
        per = n_target // 4
        chosen = [p for t in TYPES for p in pairs[t][:per]]
        for t in TYPES:
            for p in spare[t] + pairs[t][per:]:
                if len(chosen) < n_target:
                    chosen.append(p)
        if len(chosen) < 10:  # too few pairs for the grouped 5-fold leakage test: Check B is not run
            return []
        out = []
        for j, (k, u) in enumerate(chosen):
            for lab_, x, s in ((1, k, "k"), (0, u, "u")):
                ids, st = self.cal_ids(m, x)
                # id "<pair>:<k|u>:<entity id>": run.py groups by the part before the first colon
                out.append(dict(id=f"b{j}:{s}:{x['id']}", text=self.cal_text(x), label=lab_, ids=ids, start=st))
        return out
