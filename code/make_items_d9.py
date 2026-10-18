"""D9 item pools (PREREGISTRATION_STAGE2B_D9.md §3). python -I make_items_d9.py <d9_data dir>

- cal: the 83 AMC 2022-23 problems of AI-MO/aimo-validation-amc, Zhao et al.'s in-domain set (their
  released direction is "amc"). The direction is rebuilt from each model's own chains on them.
- Every other split: MATH test problems (EleutherAI/hendrycks_math, Zhao et al.'s out-of-domain set),
  Levels 1-3, integer reference answer (last \\boxed{} of the solution), a digit in the problem.
  Seed 90; splits are disjoint.
- D: 120 of the Check A problems (seed 91), each with a unit preference that alternates "dt", "et":
  the step that is perturbed is chosen per model from its own chain (spec_d9.choose_unit).
"""
import json
import os
import random
import re
import sys


def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j = k = i + 7
    depth = 1
    while k < len(s) and depth:
        depth += (s[k] == "{") - (s[k] == "}")
        k += 1
    return s[j:k - 1]


def int_answer(sol):
    a = boxed(sol)
    if a is None:
        return None
    a = a.replace("$", "").replace(",", "").replace("\\!", "").replace(" ", "")
    return a if re.fullmatch(r"-?\d+", a) else None


D = sys.argv[1]
amc = json.load(open(os.path.join(D, "aimo_validation_amc.json"), encoding="utf-8"))
cal = [dict(id=f"amc{r['id']}", problem=r["problem"], answer=str(int(float(r["answer"]))) if float(r["answer"]).is_integer()
            else str(r["answer"]), url=r.get("url")) for r in amc]
M = json.load(open(os.path.join(D, "hendrycks_math_test.json"), encoding="utf-8"))
pool = []
for cfg in sorted(M):
    for i, r in enumerate(M[cfg]):
        a = int_answer(r["solution"])
        if a is None or r["level"] not in ("Level 1", "Level 2", "Level 3") or not re.search(r"\d", r["problem"]):
            continue
        pool.append(dict(id=f"m_{cfg}_{i}", problem=r["problem"], answer=a, level=r["level"], type=r["type"]))
random.Random(90).shuffle(pool)
sizes = [("calib0a", 40), ("pilot", 40), ("pilot_B", 30), ("B", 120), ("oracle", 120), ("A", 240)]
I, o = dict(cal=cal), 0
for k, n in sizes:
    I[k] = pool[o:o + n]
    o += n
aid = [x["id"] for x in I["A"]]
I["D"] = sorted(random.Random(91).sample(aid, 120), key=aid.index)
I["E"] = I["D"][:60]
I["pilot_D"] = [x["id"] for x in I["pilot"]][:20]
I["pref"] = {i: ("dt" if j % 2 == 0 else "et") for lst in (I["D"], I["pilot_D"]) for j, i in enumerate(lst)}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "items_d9.json")
json.dump(I, open(out, "w", encoding="utf-8"), indent=0)
print("eligible MATH problems:", len(pool), {k: len(v) for k, v in I.items()})
