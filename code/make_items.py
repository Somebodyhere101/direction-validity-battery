"""Builds the frozen, model-independent item pools for D1 (refusal), D2 (truth) and D3 (CR axis)
with fixed seeds, and writes data/items.json plus the calibration data the direction recipes use.
Run once on the laptop before the pilot; the output is hashed in PREREG_SHA256SUMS (Stage 2).

python -I make_items.py <third_party dir> <out dir>
"""

import csv
import importlib.util
import json
import os
import random
import sys

TP, OUT = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

import transformers  # noqa: E402

TOK = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")  # Qwen2.5/Qwen3 share the vocabulary
ntok = lambda s: len(TOK(s, add_special_tokens=False)["input_ids"])


def rd_csv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


# ------------------------------------------------------------------ D1 refusal
xs = rd_csv(os.path.join(TP, "xstest", "xstest_prompts.csv"))
PAIR = {"homonyms": "contrast_homonyms", "figurative_language": "contrast_figurative_language",
        "safe_targets": "contrast_safe_targets", "safe_contexts": "contrast_safe_contexts",
        "definitions": "contrast_definitions", "historical_events": "contrast_historical_events",
        "nons_group_real_discr": "contrast_discr", "privacy_public": "contrast_privacy"}
by_type = {}
for r in xs:
    by_type.setdefault(r["type"], []).append(r)
pairs = []
for s, c in PAIR.items():
    for a, b in zip(by_type[s], by_type[c]):  # XSTest's contrast prompts follow the safe prompts' order
        if abs(ntok(a["prompt"]) - ntok(b["prompt"])) <= 1:
            pairs.append((a, b))
rng = random.Random(6)
rng.shuffle(pairs)
b_pairs, pilot_b_pairs = pairs[:120], pairs[120:140]
used = {r["id"] for p in pairs[:140] for r in p}
xs_rest = [r for r in xs if r["id"] not in used]
rng.shuffle(xs_rest)


def xs_item(r, tag):
    return dict(id=f"xs{r['id']}", src="xstest", type=r["type"], label=int(r["label"] == "unsafe"),
                prompt=r["prompt"], set=tag)


hard = [r["prompt"] for r in rd_csv(os.path.join(TP, "or-bench", "or-bench-hard-1k.csv"))]
tox = [r["prompt"] for r in rd_csv(os.path.join(TP, "or-bench", "or-bench-toxic.csv"))]
random.Random(7).shuffle(hard)
random.Random(8).shuffle(tox)
orb = lambda p, i, lab, tag: dict(id=f"or{'t' if lab else 'h'}{i}", src="or-bench-" + ("toxic" if lab else "hard"),
                                  type="or", label=lab, prompt=p, set=tag)
d1 = dict(
    pilot=[orb(p, i, 0, "pilot") for i, p in enumerate(hard[:40])] + [orb(p, i, 1, "pilot") for i, p in enumerate(tox[:40])],
    oracle=[orb(p, i + 40, 0, "oracle") for i, p in enumerate(hard[40:100])] + [orb(p, i + 40, 1, "oracle") for i, p in enumerate(tox[40:100])],
    A=[xs_item(r, "A") for r in xs_rest] + [orb(p, i + 100, 0, "A") for i, p in enumerate(hard[100:180])]
    + [orb(p, i + 100, 1, "A") for i, p in enumerate(tox[100:180])],
    B=[dict(id=f"B{k}", safe=a["prompt"], unsafe=b["prompt"], type=a["type"]) for k, (a, b) in enumerate(b_pairs)],
    pilot_B=[dict(id=f"pB{k}", safe=a["prompt"], unsafe=b["prompt"], type=a["type"]) for k, (a, b) in enumerate(pilot_b_pairs)],
)
rd = random.Random(9)
d1["D"] = sorted(rd.sample([x["id"] for x in d1["A"]], 120), key=[x["id"] for x in d1["A"]].index)
d1["E"] = d1["D"][:60]
d1["pilot_D"] = [x["id"] for x in d1["pilot"]][::2][:40]

# Arditi et al.'s calibration data, exactly as their pipeline samples it (random.seed(42)).
SP = os.path.join(TP, "refusal_direction", "dataset")
ld = lambda f: json.load(open(os.path.join(SP, f), encoding="utf-8"))
random.seed(42)
cal1 = dict(
    harmful_train=[x["instruction"] for x in random.sample(ld("splits/harmful_train.json"), 128)],
    harmless_train=[x["instruction"] for x in random.sample(ld("splits/harmless_train.json"), 128)],
    harmful_val=[x["instruction"] for x in random.sample(ld("splits/harmful_val.json"), 32)],
    harmless_val=[x["instruction"] for x in random.sample(ld("splits/harmless_val.json"), 32)],
)
cal1["jailbreakbench"] = [x["instruction"] for x in ld("processed/jailbreakbench.json")]
random.seed(42)
cal1["harmless_test"] = [x["instruction"] for x in random.sample(ld("splits/harmless_test.json"), 100)]

# ------------------------------------------------------------------- D2 truth
GT = os.path.join(TP, "geometry-of-truth", "datasets")
rows = {n: rd_csv(os.path.join(GT, n + ".csv")) for n in
        ("cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "larger_than", "smaller_than")}
cal2 = dict(statements=[r["statement"] for n in ("cities", "neg_cities") for r in rows[n]],
            labels=[int(r["label"]) for n in ("cities", "neg_cities") for r in rows[n]])
st = lambda r, n, i, tag: dict(id=f"{n}:{i}", src=n, statement=r["statement"], label=int(r["label"]), set=tag)
d2 = dict(A=[], oracle=[], pilot=[], heldout={})
for k, n in enumerate(("sp_en_trans", "neg_sp_en_trans", "smaller_than")):
    idx = list(range(len(rows[n])))
    random.Random(20 + k).shuffle(idx)
    d2["A"] += [st(rows[n][i], n, i, "A") for i in idx[:80]]
    d2["oracle"] += [st(rows[n][i], n, i, "oracle") for i in idx[80:120]]
    d2["pilot"] += [st(rows[n][i], n, i, "pilot") for i in idx[120:140]]
    d2["heldout"][n] = [st(rows[n][i], n, i, "heldout") for i in idx[140:340]]
lt, sm = rows["larger_than"], rows["smaller_than"]
# Matched set (Stage 2 construction, after the pilot showed that order-swapped larger_than pairs leak
# through capitalisation): the same two numbers in the same order and case, with "larger" vs
# "smaller"; exactly one of the two is true.
used_sm = {int(x["id"].split(":")[1]) for k in ("A", "oracle", "pilot") for x in d2[k] if x["src"] == "smaller_than"}
used_sm |= {int(x["id"].split(":")[1]) for x in d2["heldout"]["smaller_than"]}
key = {(r["n1"], r["n2"]): i for i, r in enumerate(lt)}
cand = [(key[(r["n1"], r["n2"])], j) for j, r in enumerate(sm) if j not in used_sm]
cand = [(i, j) for i, j in cand if abs(ntok(lt[i]["statement"]) - ntok(sm[j]["statement"])) <= 1]
random.Random(24).shuffle(cand)
tf = lambda i, j: (lt[i]["statement"], sm[j]["statement"]) if lt[i]["label"] == "1" else (sm[j]["statement"], lt[i]["statement"])
d2["B"] = [dict(id=f"B{k}", true=tf(i, j)[0], false=tf(i, j)[1]) for k, (i, j) in enumerate(cand[:120])]
d2["pilot_B"] = [dict(id=f"pB{k}", true=tf(i, j)[0], false=tf(i, j)[1]) for k, (i, j) in enumerate(cand[120:140])]
usedB = {i for i, _ in cand[:140]}
rest = [i for i in range(len(lt)) if i not in usedB]
random.Random(25).shuffle(rest)
d2["heldout"]["larger_than"] = [st(lt[i], "larger_than", i, "heldout") for i in rest[:200]]
rd = random.Random(26)
d2["D"] = sorted(rd.sample([x["id"] for x in d2["A"]], 120), key=[x["id"] for x in d2["A"]].index)
d2["E"] = d2["D"][:60]
d2["pilot_D"] = [x["id"] for x in d2["pilot"]][::2][:30]

# ------------------------------------------------------------------ D3 CR axis
spec = importlib.util.spec_from_file_location("cr4items", os.path.join(HERE, "..", "..", "cr4", "items.py"))
cr4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr4)
S = cr4.build()
M = [x for x in S["main"] if x["set"] == "M"]
usedp = {(x["a"], x["b"]) for x in S["main"] + S["pilot"]}
allp = [(a, b) for a in range(11, 100) for b in range(a, 100) if (a, b) not in usedp]
orc = random.Random(30).sample(allp, 120)
d3 = dict(A=M, oracle=[cr4.item(a, b, "O", "familiar" if cr4.familiar(a, b) else "novel" if cr4.novel(a, b) else "other")
                       for a, b in orc],
          pilot=[x for x in S["pilot"] if x["set"] == "M"])
rd = random.Random(31)
d3["D"] = sorted(rd.sample([x["id"] for x in d3["A"]], 120), key=[x["id"] for x in d3["A"]].index)
d3["E"] = d3["D"][:60]
d3["pilot_D"] = [x["id"] for x in d3["pilot"]][:30]

json.dump(dict(d1=d1, d2=d2, d3=d3), open(os.path.join(OUT, "items.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
json.dump(dict(d1=cal1, d2=cal2), open(os.path.join(OUT, "calibration.json"), "w", encoding="utf-8"), indent=0, ensure_ascii=False)
print("D1", {k: len(v) for k, v in d1.items()}, "xstest pairs matched:", len(pairs),
      "A labels:", sum(x["label"] for x in d1["A"]), "/", len(d1["A"]))
print("D2", {k: (len(v) if not isinstance(v, dict) else {a: len(b) for a, b in v.items()}) for k, v in d2.items()},
      "larger_than pairs:", len(cand))
print("D3", {k: len(v) for k, v in d3.items()})
print("cal1", {k: len(v) for k, v in cal1.items()}, "cal2", len(cal2["statements"]))
