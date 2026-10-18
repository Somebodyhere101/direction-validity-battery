"""D9 clean stage (replaces run.py's clean stage for D9). python d9_clean.py <model key> <pilot|main>

For every Check A and oracle problem: the model's own greedy chain (4,096 new tokens at most), the
True Thinking Score of every numeric step (spec_d9.score_chain), the block-ell output at the step's
last token in Zhao et al.'s extraction format, and its projections on d, N_len, N_pc1, the label-
permuted directions and (R1 models) the released vector. Check D problems also get their unit (the
perturbed step pushed in run.py's push stage). Needs run.py's cal stage. Resumable.

Writes res/d9/<key>_<phase>_clean.jsonl (one line per problem; per-step records under "sentences",
the field name run.py's per-sentence oracle and analyze.py read) and _clean_h.npz ("<id>:<j>").
"""
import json
import os
import sys
import time

import numpy as np
import torch

from p6lib import Model, unit
from spec_d9 import CAP_TEST, D9, ITEMS_D9, choose_unit, process_problems

KEY, PHASE = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "res", "d9", f"{KEY}_{PHASE}")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = D9()
m = Model(KEY, thinking=True)
cal = torch.load(PRE + "_cal.pt", weights_only=False)
ell = cal["ell"]
log(f"{m.name} L={m.L} ell={ell} phase={PHASE}")
Pm = np.array([unit(cal["d"]), cal["Nlen"], cal["Npc1"]] + list(cal["perm"]))
rel = unit(cal["released"]) if cal.get("released") is not None else None
S = spec.items(PHASE)
probs = list({x["id"]: x for x in S["A"] + S["oracle"]}.values())
recs, H = process_problems(m, probs, CAP_TEST, PRE + "_chains.jsonl", log, read_ell=ell, npz=PRE + "_clean_h.npz")
Dset, pref = set(S["D"]), ITEMS_D9()["pref"]
setof = {x["id"]: "A" for x in S["A"]}
setof.update({x["id"]: ("A+oracle" if x["id"] in setof else "oracle") for x in S["oracle"]})
n_units = {}
with open(PRE + "_clean.jsonl", "w", encoding="utf-8") as f:
    for p in probs:
        r = recs[p["id"]]
        sents = []
        for st in r.get("sentences", []):
            h = H[f"{p['id']}:{st['j']}"].astype(np.float64)
            pr = Pm @ h
            sents.append(dict(st, P=float(pr[0]), P_Nlen=float(pr[1]), P_Npc1=float(pr[2]), P_perm=pr[3:].tolist(),
                              P_rel=float(h @ rel) if rel is not None else None))
        rec = dict(id=p["id"], set=setof[p["id"]], n_tok=r["n_tok"], finished=r["finished"], boxed=r["boxed"],
                   correct=r["correct"], ystar=r["ystar"], n_steps=r.get("n_steps"), n_num=r.get("n_num"), sentences=sents)
        if p["id"] in Dset:
            rec["unit"] = choose_unit(r, pref[p["id"]], p["id"])
            n_units[rec["unit"]["kind"]] = n_units.get(rec["unit"]["kind"], 0) + 1
        f.write(json.dumps(rec) + "\n")
allT = np.array([st["T"] for p in probs for st in recs[p["id"]].get("sentences", [])])
log(f"clean: {len(probs)} problems, {len(allT)} numeric steps, TTS mean {allT.mean():.4f}, >0.9 {(allT > 0.9).mean():.4f}, "
    f"<=0.005 {(allT <= 0.005).mean():.3f}; correct {np.mean([bool(recs[p['id']]['correct']) for p in probs]):.2f}, "
    f"finished {np.mean([recs[p['id']]['finished'] for p in probs]):.2f}; Check D units {n_units}")
log("DONE")
