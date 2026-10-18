"""Judge validation (§0.4): draws 50 D1 pilot outputs, stratified over push kinds, and writes them
WITHOUT judge labels for the assistant to label first. python blind_sample.py <push.jsonl> <clean.jsonl> <out.jsonl>"""
import json, random, sys
from specs import ITEMS
pool = {x["id"]: x["prompt"] for k in ("A", "oracle", "pilot") for x in ITEMS["d1"].get(k, [])}
rows = []
for l in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(l)
    for j, row in enumerate(r["rows"]):
        rows.append((row["kind"], f"{r['id']}|{j}", pool[r["id"]], row["gen"]["text"]))
for l in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(l)
    rows.append(("clean_only", f"{r['id']}|0", pool[r["id"]], r["gen"]["text"]))
rng = random.Random(50)
by = {}
for x in rows:
    by.setdefault(x[0], []).append(x)
quota = {"clean_only": 12, "d": 12, "o": 4, "Nlen": 4, "Npc1": 4, "rand": 6, "perm": 4, "E2": 2, "E1": 2}
pick = []
for k, n in quota.items():
    pick += rng.sample(by.get(k, []), min(n, len(by.get(k, []))))
rng.shuffle(pick)
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for kind, key, p, resp in pick[:50]:
        f.write(json.dumps(dict(key=key, prompt=p, response=resp)) + "\n")
print(len(pick[:50]), "written")
