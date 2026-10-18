"""EXPLORATORY (Stage 1 §9, not confirmatory): matched-KL effect of pushes at every strength already in a
run's push file, for the oracle o and the direction d, on the primary drive target.

It runs the analysis script of the folder it sits in (analyze.py) up to its Check D machinery, unchanged,
then calls check_D for each (kind, strength, sign). Nothing in the preregistered summaries changes.

python explore_oracle_k.py <dir> <model key> [main] [osweep]  ->  res/<dir>/<key>_<phase>_explore_k[_osweep].json
With "osweep", the rows come from oracle_sweep.py (strengths 4 and 8 gaps) instead of the push file.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIR, KEY = sys.argv[1], sys.argv[2]
PHASE = sys.argv[3] if len(sys.argv) > 3 else "main"
SWEEP = len(sys.argv) > 4 and sys.argv[4] == "osweep"

src = open(os.path.join(HERE, "analyze.py"), encoding="utf-8").read()
cut = src.index("\nsigns = sorted(")  # everything before the preregistered verdict computation
ns = {"__name__": "explore", "__file__": os.path.join(HERE, "analyze.py")}
sys.argv = ["analyze.py", DIR, KEY, PHASE]
exec(compile(src[:cut], "analyze.py", "exec"), ns)

if SWEEP:  # check_D reads the module-level `push`; point it at the sweep rows
    ns["push"] = [json.loads(l) for l in open(os.path.join(HERE, "res", DIR, f"{KEY}_{PHASE}_osweep.jsonl"), encoding="utf-8")]
push, DRIVE, check_D = ns["push"], ns["DRIVE"], ns["check_D"]
target, sesoi = DRIVE[0]
kinds = sorted({(r["kind"], abs(r["k"])) for rec in push for r in rec["rows"] if r["kind"] in ("d", "o") and "k" in r})
signs = sorted({int(r["k"] > 0) * 2 - 1 for rec in push for r in rec["rows"] if r["kind"] == "d" and "k" in r})
out = dict(direction=DIR, model=KEY, phase=PHASE, target=target, sesoi=sesoi, exploratory=True, results=[])
for kind, k in kinds:
    for s in signs:
        r = check_D(target, sesoi, s, push_kind=kind, k=k)
        if r is None:
            continue
        out["results"].append(dict(kind=kind, k=k, sign=s, n=r["n_items"], kl_median=r.get("kl_median"),
                                   rand=r.get("rand"), perm=r.get("perm"), verdict=r.get("verdict"),
                                   verdict_rand=r.get("verdict_rand"), verdict_perm=r.get("verdict_perm")))
p = os.path.join(HERE, "res", DIR, f"{KEY}_{PHASE}_explore_k{'_osweep' if SWEEP else ''}.json")
json.dump(out, open(p, "w"), indent=1, default=str)
for x in out["results"]:
    e = (x.get("rand") or {})
    print(f"{DIR} {KEY} {x['kind']} k={x['k']:<4} s={x['sign']:+d} n={x['n']:<4} kl~{x['kl_median']:.2f} "
          f"eff_rand={e.get('effect', float('nan')):+.4f} ci={[round(c, 4) for c in e.get('ci', [])]} sesoi={sesoi:.4f} -> {x['verdict']}")
