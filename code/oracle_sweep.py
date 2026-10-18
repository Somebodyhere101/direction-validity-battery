"""EXPLORATORY (Stage 1 §9, not confirmatory): strength sweep for the oracle and the direction beyond the
preregistered grid. For each of the first N Check D items, one batch holds the clean row, o and d at
k in {±4, ±8} gaps, and the run's 12 Gaussian reference directions at c in {2, 3, 4, 6, 8, 12, 16, 24} gaps,
so that every push is bracketed for KL matching. Uses the oracle, Gaussian directions and calibration
saved by the main run (res/<dir>/<key>_main_dirs.npz, _cal.pt); nothing preregistered is rerun or changed.

python oracle_sweep.py <dir> <model key> [n_items]   ->  res/<dir>/<key>_main_osweep.jsonl
"""

import json
import os
import sys
import time

import numpy as np
import torch

from p6lib import Model, rows_to_ops, unit
from specs import SPECS
from spec_d4 import D4, D4G
from spec_d8 import D8

SPECS["d4"], SPECS["d4g"], SPECS["d8"] = D4, D4G, D8
DIR, KEY = sys.argv[1], sys.argv[2]
N_ITEMS = int(sys.argv[3]) if len(sys.argv) > 3 else 80
os.environ["P6_PHASE"] = "main"
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "res", DIR, f"{KEY}_main")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "96"))
K_PUSH, C_REF = [4, 8], [2, 3, 4, 6, 8, 12, 16, 24]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = SPECS[DIR]()
m = Model(KEY, thinking=getattr(spec, "thinking", False))
n_new = lambda mode: spec.n_new_mode[mode] if hasattr(spec, "n_new_mode") else spec.n_new
cal = torch.load(PRE + "_cal.pt", weights_only=False)
ELL, G, du = cal["ell"], cal["g"], unit(cal["d"])
dirs = np.load(PRE + "_dirs.npz")
o, gauss = dirs["o"], dirs["gauss"]
S = spec.items("main")
byid = {x["id"]: x for x in S["A"] + S["oracle"]}
items = S["D"][:N_ITEMS]
log(f"{m.name} L={m.L} ell={ELL} g={G:.2f}; {len(items)} items; k={K_PUSH} c={C_REF}")

path = PRE + "_osweep.jsonl"
done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
kl_mode = "direct" if "direct" in spec.modes else spec.modes[0]
with open(path, "a", encoding="utf-8") as f:
    for n, iid in enumerate([i for i in items if i not in done]):
        it = byid[iid]
        rows = [(dict(kind="clean"), ELL, np.zeros_like(du))]
        for s in spec.signs(it):
            for k in K_PUSH:
                rows.append((dict(kind="d", k=s * k), ELL, s * k * G * du))
                rows.append((dict(kind="o", k=s * k), ELL, s * k * G * o))
        for j in range(gauss.shape[0]):
            for c in C_REF:
                rows.append((dict(kind="rand", j=j, c=c), ELL, c * G * gauss[j]))
        out_rows = [dict(r[0]) for r in rows]
        step = MAX_ROWS - 1
        body = rows[1:]
        for ci, c0 in enumerate(range(0, len(body), step)):
            ch = [rows[0]] + body[c0:c0 + step]
            add, _ = rows_to_ops(ch, m.d)
            tgt = [0] + list(range(1 + c0, 1 + c0 + len(ch) - 1))
            for mode in spec.modes:
                ids, start = spec.prompt_ids(m, it, mode)
                r = m.generate(ids, start, add, n_new(mode), lens=(mode == "direct"))
                for b, ti in enumerate(tgt):
                    if ci > 0 and b == 0:
                        continue
                    oc = spec.outcome(m, it, mode, r["tokens"][b], r["lp0"][b], r["top1"][b] if "top1" in r else None)
                    oc.pop("text", None)
                    out_rows[ti][mode] = oc
                if mode == kl_mode:
                    K = m.tf_kl(ids, start, add, m.trunc(r["tokens"][0]))
                    for b, ti in enumerate(tgt):
                        if ci > 0 and b == 0:
                            continue
                        out_rows[ti]["kl"] = K["kl_all"][b]
        f.write(json.dumps(dict(id=iid, n_rows=len(rows), rows=out_rows)) + "\n")
        f.flush()
        if n % 10 == 0:
            log(f"sweep {n}/{len(items)} {iid} rows={len(rows)} | o(+8) kl=" +
                " ".join(f"{x['kl']:.1f}" for x in out_rows if x["kind"] == "o" and x["k"] == 8))
log("DONE")
