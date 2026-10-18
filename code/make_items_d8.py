"""D8 items: ARC (allenai/ai2_arc test, Challenge + Easy, via the HF dataset viewer) with exactly four
options A-D. Each item gets one wrong option X (uniform among the wrong options, seeded), which the user
states as their belief in the belief version of the prompt. python make_items_d8.py <arc_test.json>"""
import json, random, sys
src = json.load(open(sys.argv[1], encoding="utf-8"))
pool = []
for cfg in ("ARC-Challenge", "ARC-Easy"):
    for r in src[cfg]:
        if r["choices"]["label"] != ["A", "B", "C", "D"] or r["answerKey"] not in "ABCD":
            continue
        pool.append(dict(id=f"{cfg[4]}{r['id']}", question=r["question"], options=r["choices"]["text"], answer=r["answerKey"], src=cfg))
rng = random.Random(80)
rng.shuffle(pool)
for x in pool:
    x["belief"] = rng.choice([c for c in "ABCD" if c != x["answer"]])
I = dict(pilot=pool[:80], oracle=pool[80:300], A=pool[300:1100])
I["D"] = [x["id"] for x in I["A"][:200]]   # fixed; the analysis keeps the items each model answers correctly without the belief
I["E"] = I["D"][:80]
# Check B (Stage 2b-D8): the same belief sentence naming the correct option (label 0) or the item's wrong
# option (label 1); disjoint from every other split. 120 pairs main, 40 pilot.
I["B"], I["B_pilot"] = pool[1100:1220], pool[1220:1260]
# Check A extended (Stage 2b-D8) to every remaining item: the target is defined only on items answered correctly
# without the belief, and Stage 1 §8 asks for >= 100 in the smaller outcome class where the source allows it.
# D and E keep their items (the first 200 / 80 of the original 800).
I["A"] = I["A"] + pool[1260:]
json.dump(I, open("data/items_d8.json", "w", encoding="utf-8"), indent=0, ensure_ascii=False)
print(len(pool), {k: len(v) for k, v in I.items()})
import collections; print("belief letters", collections.Counter(x["belief"] for x in I["A"]), "answers", collections.Counter(x["answer"] for x in I["A"]))
