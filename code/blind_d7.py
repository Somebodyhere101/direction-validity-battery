"""Blind check of D7's refusal measure (Stage 1 §0.4 procedure applied to the authors' phrase list).

python blind_d7.py sample <push.jsonl> <clean.jsonl> <out.jsonl>
    50 pilot outputs, stratified over row kinds, written WITHOUT any automatic label, for the
    assistant to label first (1 = refuses / says it does not know / declines, 0 = gives an answer).
python blind_d7.py score <out.jsonl> <labels.json>
    Cohen's kappa of the phrase list (spec_d7.is_generation_refusal) against those labels.
"""
import json
import random
import sys

if sys.argv[1] == "sample":
    rows = []
    for l in open(sys.argv[2], encoding="utf-8"):
        r = json.loads(l)
        for j, row in enumerate(r["rows"]):
            if j and "direct" in row:
                rows.append((row["kind"], f"{r['id']}|{j}", row["direct"]["text"]))
    for l in open(sys.argv[3], encoding="utf-8"):
        r = json.loads(l)
        rows.append(("clean", f"{r['id']}|0", r["direct"]["text"]))
    rng = random.Random(77)
    by = {}
    for x in rows:
        by.setdefault(x[0], []).append(x)
    quota = {"clean": 14, "d": 12, "o": 4, "Nlen": 4, "Npc1": 4, "rand": 6, "perm": 4, "E2": 2}
    pick = []
    for k, n in quota.items():
        pick += rng.sample(by.get(k, []), min(n, len(by.get(k, []))))
    rest = [x for x in rows if x not in pick]
    pick += rng.sample(rest, max(0, 50 - len(pick)))
    rng.shuffle(pick)
    with open(sys.argv[4], "w", encoding="utf-8") as f:
        for kind, key, text in pick[:50]:
            f.write(json.dumps(dict(key=key, response=text), ensure_ascii=False) + "\n")
    print(len(pick[:50]), "written")
else:
    from spec_d7 import is_generation_refusal
    items = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]
    lab = json.load(open(sys.argv[3], encoding="utf-8"))
    a = [int(lab[x["key"]]) for x in items]
    b = [is_generation_refusal(x["response"]) for x in items]
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    out = dict(n=n, agreement=po, kappa=kappa, rate_assistant=pa, rate_phrase=pb,
               disagreements=[dict(key=x["key"], assistant=p, phrase=q, response=x["response"][:200]) for x, p, q in zip(items, a, b) if p != q])
    json.dump(out, open(sys.argv[2].replace(".jsonl", "_score.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}))
