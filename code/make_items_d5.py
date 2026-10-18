"""D5 item pools: GSM8K test problems (openai/grade-school-math test.jsonl), disjoint from Ward et
al.'s Claude-generated calibration tasks. python make_items_d5.py <test.jsonl>"""
import json, random, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
rows = [dict(id=f"g{i}", question=r["question"], answer=r["answer"].split("####")[-1].strip()) for i, r in enumerate(rows)]
random.Random(50).shuffle(rows)
I = dict(calib0a=rows[:20], pilot=rows[20:60], oracle=rows[60:180], A=rows[180:420])
I["D"] = sorted(random.Random(51).sample([x["id"] for x in I["A"]], 120), key=[x["id"] for x in I["A"]].index)
I["E"] = I["D"][:60]
json.dump(I, open("data/items_d5.json", "w", encoding="utf-8"), indent=0)
print({k: len(v) for k, v in I.items()})
