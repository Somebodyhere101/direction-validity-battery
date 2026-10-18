"""D4G items: GSM8K test problems not used by D5 (openai/grade-school-math test.jsonl, the same shuffle as
make_items_d5.py, taking the problems after D5's 420). python make_items_d4g.py <test.jsonl>"""
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
rows = [dict(id=f"g{i}", question=r["question"], answer=r["answer"].split("####")[-1].strip()) for i, r in enumerate(rows)]
random.Random(50).shuffle(rows)          # identical to make_items_d5.py
rest = rows[420:]                        # D5 used rows[:420]
I = dict(pilot=rest[:40], oracle=rest[40:160], A=rest[160:400])
I["D"] = sorted(random.Random(61).sample([x["id"] for x in I["A"]], 120), key=[x["id"] for x in I["A"]].index)
I["E"] = I["D"][:60]
d5 = json.load(open("data/items_d5.json", encoding="utf-8"))
used = {x["id"] for k in ("calib0a", "pilot", "oracle", "A") for x in d5[k]}
assert not used & {x["id"] for k in ("pilot", "oracle", "A") for x in I[k]}
json.dump(I, open("data/items_d4g.json", "w", encoding="utf-8"), indent=0)
print({k: len(v) for k, v in I.items()})
