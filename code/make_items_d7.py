"""D7 item pools (PREREGISTRATION_STAGE2B_D7.md §3): Ferrando et al.'s Wikidata entities, split into
disjoint pools with fixed seeds. Known/unknown labels are NOT fixed here: they are computed per
panel model from its own answers (run_d7.py label), as the authors do.

python make_items_d7.py <sae_entities dir> [--no-fetch]   ->  data/items_d7.json (+ data/sitelinks_d7.json)

- Entities: dataset/processed/entities/{player,movie,city,song}.json (the authors' processed lists),
  deduplicated by name within type, names without newlines or double quotes.
- Question per type (one template and answer format per type, all with name answers): the authors'
  chat-model relations (mech_interp_utils.entity_type_to_it_relations) in their refusal-analysis
  template '{relation} the {entity_type} {entity_name}?'.
- Entity token count: tokens covering the entity in the question (Qwen3 tokenizer, shared by Qwen2.5).
  A and oracle are stratified by it: 2..6 tokens, equal numbers per count within each type.
- Popularity covariate: Wikidata sitelink count of the entity's item (QID from the authors' raw files;
  max over items when a name maps to several). Fetched once and frozen in data/sitelinks_d7.json.
"""

import json
import os
import random
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TYPES = ("player", "movie", "city", "song")
QUESTION = {  # authors' it relation for the chosen attribute (one per type)
    "player": ("player_teams_list", "teams_list", "In which teams have played"),
    "movie": ("movie_directors", "directors", "Who directed"),
    "city": ("city_country", "country", "In which country is"),
    "song": ("song_performers", "performers", "Who is the singer of"),
}
RAW = {"player": ("human", "humanLabel"), "movie": ("film", "filmLabel"), "city": ("city", "cityLabel"), "song": ("song", "songLabel")}
# pool sizes per type, carved in this order from each type's shuffled list
STRAT = (("A", 60), ("oracle", 30))          # stratified on entity token count 2..6
PLAIN = (("calib0a", 75), ("D", 250), ("B", 400), ("pilot", 100), ("cal", 600))
SEED = 70
NTOK_RANGE = (2, 6)
MAX_NTOK = 10


def question_text(t, e):
    return f"{QUESTION[t][2]} the {t} {e}?"


def entity_ntok(tok, t, e):
    q = question_text(t, e)
    s = q.index(" " + e + "?") + 1
    enc = tok(q, add_special_tokens=False, return_offsets_mapping=True)
    return sum(1 for a, b in enc["offset_mapping"] if a < s + len(e) and b > s)


def fetch_sitelinks(qids, cache_path):
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    todo = sorted({q for q in qids if q not in cache})
    for i in range(0, len(todo), 50):
        ids = "|".join(todo[i:i + 50])
        url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={ids}&props=sitelinks&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "paper6-d7-items/1.0 (academic research script)"})
        for attempt in range(10):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=60))
                break
            except Exception as ex:  # noqa: BLE001  (429: wait as asked, else back off)
                wait = int(getattr(ex, "headers", {}).get("Retry-After", 0) or 0) if hasattr(ex, "headers") else 0
                print("retry", attempt, repr(ex)[:120], flush=True)
                time.sleep(max(wait, 10 * (attempt + 1)))
        else:
            raise RuntimeError("sitelink fetch failed")
        for q in todo[i:i + 50]:
            ent = r.get("entities", {}).get(q, {})
            cache[q] = len(ent.get("sitelinks", {})) if "missing" not in ent else None
        time.sleep(1.5)
        if (i // 50) % 20 == 0:
            print(f"sitelinks {i + 50}/{len(todo)}", flush=True)
            json.dump(cache, open(cache_path, "w", encoding="utf-8"))
    json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=0, sort_keys=True)
    return cache


def main():
    root = sys.argv[1]
    fetch = "--no-fetch" not in sys.argv
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    pools = {k: [] for k, _ in STRAT + PLAIN}
    qmap_all = {}
    for t in TYPES:
        ents = json.load(open(os.path.join(root, "dataset", "processed", "entities", f"{t}.json"), encoding="utf-8"))
        raw = json.load(open(os.path.join(root, "dataset", "raw", "wikidata", f"{t}.json"), encoding="utf-8"))
        qmap = {}
        for r in raw:
            qmap.setdefault(r[RAW[t][1]], []).append(r[RAW[t][0]].rsplit("/", 1)[-1])
        seen, rows = set(), []
        for i, x in enumerate(ents):
            e = x["entity"]
            attrs = {a["attribute_type"]: a["attribute_value"] for a in x["attributes"] if str(a["attribute_value"]).strip()}
            if e in seen or not e.strip() or "\n" in e or '"' in e or QUESTION[t][1] not in attrs:
                continue
            seen.add(e)
            n = entity_ntok(tok, t, e)
            if n < 1 or n > MAX_NTOK:
                continue
            q = sorted(set(qmap.get(e, [])))
            rows.append(dict(id=f"{t[0]}{i}", type=t, entity=e, attrs=attrs, ptype=QUESTION[t][0],
                             answer=attrs[QUESTION[t][1]], question=question_text(t, e), ntok=n, qids=q))
            for qq in q:
                qmap_all[qq] = 1
        random.Random(SEED + TYPES.index(t)).shuffle(rows)
        # plain pools first (simple random samples), then the stratified pools from the remainder, so that
        # stratifying does not deplete any token count from the front of the shuffled list
        o = 0
        for name, n in PLAIN:
            assert o + n <= len(rows), (t, name)
            pools[name] += rows[o:o + n]
            o += n
        rest, used = rows[o:], set()
        for name, n in STRAT:  # equal numbers per entity token count (2..6) within the type
            per = n // (NTOK_RANGE[1] - NTOK_RANGE[0] + 1)
            for c in range(NTOK_RANGE[0], NTOK_RANGE[1] + 1):
                pick = [r for r in rest if r["ntok"] == c and r["id"] not in used][:per]
                assert len(pick) == per, (t, name, c, len(pick))
                used |= {r["id"] for r in pick}
                pools[name] += pick
        print(t, "usable", len(rows), "unused", len(rest) - len(used), flush=True)
    # within each pool: a fixed interleaving of types (round robin) so that ordered selections stay type-balanced
    for name in pools:
        by = {t: [r for r in pools[name] if r["type"] == t] for t in TYPES}
        pools[name] = [by[t][i] for i in range(max(len(v) for v in by.values())) for t in TYPES if i < len(by[t])]
    sl = {}
    cache = os.path.join(HERE, "data", "sitelinks_d7.json")
    if fetch:
        need = sorted({q for v in pools.values() for r in v for q in r["qids"]})
        sl = fetch_sitelinks(need, cache)
    elif os.path.exists(cache):
        sl = json.load(open(cache, encoding="utf-8"))
    for v in pools.values():
        for r in v:
            vals = [sl.get(q) for q in r["qids"] if sl.get(q) is not None]
            r["sitelinks"] = max(vals) if vals else None
            r["n_qids"] = len(r["qids"])
    meta = dict(seed=SEED, question=QUESTION, template="{relation} the {entity_type} {entity_name}?",
                calibration_prompt="The {entity_type} '{entity}'", ntok_tokenizer="Qwen/Qwen3-8B",
                strat_ntok=list(NTOK_RANGE), sizes_per_type={k: n for k, n in STRAT + PLAIN})
    out = dict(meta=meta, **pools)
    json.dump(out, open(os.path.join(HERE, "data", "items_d7.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    print({k: len(v) for k, v in pools.items()})
    print("sitelinks missing:", {k: sum(r["sitelinks"] is None for r in v) for k, v in pools.items()})


if __name__ == "__main__":
    main()
