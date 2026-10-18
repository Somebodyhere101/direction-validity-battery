"""D6 (reasoning strength, Sheng et al. 2025) calibration set and item pools for paper 6.

python make_items_d6.py <LRM-plans-CoT MATH.json> <hendrycks_math test.json> <MATH-500 test.jsonl>

- Calibration (data/calibration_d6.json): the authors' own question file, Data/Questions/MATH.json of
  github.com/AlphaLab-USTC/LRM-plans-CoT (git blob d5d8a78, the 7,500 MATH training problems).
  "Level ?" is relabelled "Level 3", as their AnalyzeOneModel.py does.
- Items (data/items_d6.json): MATH test problems (EleutherAI/hendrycks_math, revision 21a5633; the 5,000
  test rows as JSON, third_party/hendrycks_math/test.json), so they
  are disjoint from calibration by construction; test problems whose text also occurs in the training
  file are dropped, and so are "Level ?" problems.
  - calib0a: 20 MATH-500 problems (4 per level), the authors' steering evaluation set.
  - Everything else comes from the 4,500 test problems outside MATH-500, stratified by level:
    pilot 40 (8 per level), oracle 120 (24), A 240 (48); D = 120 of A (24 per level), ordered so that
    every prefix cycles through the levels (the first 80 hold 16 per level, the first 60 hold 12);
    E = the first 60 of D. Pilot D = 15 pilot problems (3 per level); pilot E = the first 5.
  - B pools for Check B (pairs are matched per model at run time on token length): the remaining
    test problems, split into a pilot pool (60 level-1 problems and 300 of levels 2-5) and the main pool.
Seeds 60-66.
"""
import json
import os
import random
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
cal_path, test_path, m500_path = sys.argv[1:4]

cal = json.load(open(cal_path, encoding="utf-8"))
cal = [dict(id=f"c{i}", problem=x["problem"], level=int((x["level"] if x["level"] != "Level ?" else "Level 3").split()[-1]),
            subject=x["type"]) for i, x in enumerate(cal)]
train_text = {x["problem"].strip() for x in cal}

rows = [dict(problem=r["problem"], level=r["level"], subject=r["type"], solution=r["solution"])
        for r in json.load(open(test_path, encoding="utf-8"))]
m500 = [json.loads(l) for l in open(m500_path, encoding="utf-8")]
m500_text = {x["problem"].strip() for x in m500}


def boxed(sol):
    """Last \\boxed{...} of a MATH solution (balanced braces)."""
    i = sol.rfind("\\boxed")
    if i < 0:
        return None
    j = sol.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(sol):
        depth += {"{": 1, "}": -1}.get(sol[k], 0)
        if depth == 0:
            return sol[j + 1:k]
        k += 1
    return None


n0 = len(rows)
rows = [r for r in rows if r["level"] != "Level ?" and r["problem"].strip() not in train_text]
for i, r in enumerate(rows):
    r["level"] = int(r["level"].split()[-1])
    r["answer"] = boxed(r["solution"])
    r["id"] = f"t{i}"
    del r["solution"]
pool = [r for r in rows if r["problem"].strip() not in m500_text]
in500 = [r for r in rows if r["problem"].strip() in m500_text]
print(f"test rows {n0}; kept {len(rows)}; outside MATH-500 {len(pool)}; inside {len(in500)}")


def by_level(xs, seed):
    out = {}
    for lv in range(1, 6):
        g = [x for x in xs if x["level"] == lv]
        random.Random(seed * 10 + lv).shuffle(g)
        out[lv] = g
    return out


def take(bl, n_per):
    got = []
    for lv in range(1, 6):
        got += bl[lv][:n_per]
        bl[lv] = bl[lv][n_per:]
    return got


def interleave(xs, seed):
    """Order so that every prefix cycles through the levels (shuffled within level)."""
    bl = by_level(xs, seed)
    out = []
    for i in range(max(len(v) for v in bl.values())):
        for lv in range(1, 6):
            if i < len(bl[lv]):
                out.append(bl[lv][i])
    return out


b500 = by_level(in500, 60)
calib0a = take(b500, 4)
bl = by_level(pool, 61)
pilot = take(bl, 8)
oracle = take(bl, 24)
A = take(bl, 48)
D = [x["id"] for x in interleave(A, 62)]
Dsel = []
cnt = {lv: 0 for lv in range(1, 6)}
lev = {x["id"]: x["level"] for x in A}
for i in D:
    if cnt[lev[i]] < 24:
        Dsel.append(i)
        cnt[lev[i]] += 1
pilot_D = [x["id"] for x in interleave(pilot, 63)]
pcnt = {lv: 0 for lv in range(1, 6)}
plev = {x["id"]: x["level"] for x in pilot}
pD = []
for i in pilot_D:
    if pcnt[plev[i]] < 3:
        pD.append(i)
        pcnt[plev[i]] += 1
rest = [x for lv in range(1, 6) for x in bl[lv]]
random.Random(64).shuffle(rest)
r1 = [x for x in rest if x["level"] == 1]
rk = [x for x in rest if x["level"] > 1]
pilot_B = dict(lv1=r1[:60], lvk=rk[:300])
B = dict(lv1=r1[60:], lvk=rk[300:])

I = dict(calib0a=calib0a, pilot=pilot, pilot_D=pD, pilot_E=pD[:5], oracle=oracle, A=A, D=Dsel, E=Dsel[:60],
         pilot_B=pilot_B, B=B)
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(I, open(os.path.join(HERE, "data", "items_d6.json"), "w", encoding="utf-8"), indent=0)
json.dump(cal, open(os.path.join(HERE, "data", "calibration_d6.json"), "w", encoding="utf-8"), indent=0)
summ = {k: (len(v) if isinstance(v, list) else {kk: len(vv) for kk, vv in v.items()}) for k, v in I.items()}
print(summ)
print("calibration", len(cal), {lv: sum(x["level"] == lv for x in cal) for lv in range(1, 6)})
print("D levels (first 80):", {lv: sum(lev[i] == lv for i in Dsel[:80]) for lv in range(1, 6)})
