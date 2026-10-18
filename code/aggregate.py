"""Paper 6: cross-model verdicts and hypotheses H1-H7 (Stage 1 §6-7) from the per-model summaries.
H5 and H6 count process directions (D4-D9; D10 was not run); D4 is one direction scored on two item sets
(D4: paper 4's items, D4G: GSM8K), and counts as valid if either set gives a valid overall verdict.

python aggregate.py [main]  ->  res/aggregate_<phase>.json
"""

import json
import os
import sys

import numpy as np

PHASE = sys.argv[1] if len(sys.argv) > 1 else "main"
HERE = os.path.dirname(os.path.abspath(__file__))
CHAT, THINK = ["q25-7b", "q3-8b", "olmo-7b"], ["r1-qwen-7b", "r1-llama-8b", "q3-8b"]
PANEL = {"d1": CHAT, "d2": CHAT, "d3": CHAT, "d4": CHAT, "d4g": CHAT, "d5": ["r1-llama-8b", "r1-qwen-7b", "q3-8b"],
         "d6": THINK, "d7": CHAT, "d8": CHAT, "d9": THINK}
PROCESS = {"D4": ("d4", "d4g"), "D5": ("d5",), "D6": ("d6",), "D7": ("d7",), "D8": ("d8",), "D9": ("d9",)}
VALID = ("VALID TRACKER", "VALID DRIVER")
# primary drive target and SESOI when a summary predates the drive_target / drive_sesoi fields
PRIM = {"d1": ("sub", 0.10), "d2": ("V", 0.10), "d4g": ("correct", 0.10), "d7": ("drv", 0.10), "d8": ("flip", 0.10), "d9": ("F", 0.10)}
S = {}
for d, models in PANEL.items():
    for m in models:
        p = os.path.join(HERE, "res", d, f"{m}_{PHASE}_summary.json")
        if os.path.exists(p):
            S[(d, m)] = json.load(open(p, encoding="utf-8"))

out = dict(phase=PHASE, n_summaries=len(S), per_model={f"{d}/{m}": dict(verdict=s["verdict"], label=s.get("label_verdict"), statuses=s.get("statuses"),
                                                                         reproduced=s["reproduced"]) for (d, m), s in S.items()})


def overall(d):
    vs = [S[(d, m)]["verdict"] for m in PANEL[d] if (d, m) in S]
    for v in set(vs):  # settled once two of the three panel models agree (Stage 1 §6)
        if vs.count(v) >= 2:
            return v, vs
    return ("MIXED" if len(vs) == 3 else None), vs


out["overall"] = {d: dict(zip(("verdict", "per_model"), overall(d))) for d in PANEL}

# H1: refusal receives VALID DRIVER in at least two of three panel models
v1 = [S[("d1", m)]["verdict"] for m in PANEL["d1"] if ("d1", m) in S]
out["H1"] = dict(per_model=v1, supported=(v1.count("VALID DRIVER") >= 2) if len(v1) == 3 else None,
                 failing_checks={m: {k: v for k, v in S[("d1", m)]["statuses"].items() if v not in ("pass", None)}
                                 for m in PANEL["d1"] if ("d1", m) in S})

# H2: N_perm directions fail A and D (effect below SESOI) in every model
h2 = {}
for (d, m), s in S.items():
    A = (s.get("checkA") or {}).get("perm")
    if not A:
        h2[f"{d}/{m}"] = dict(missing=True)
        continue
    nA = sum(v == "pass" for v in A["verdicts"])
    nA_low = sum(v == "pass" for v, c in zip(A["verdicts"], A["cos"]) if abs(c) < 0.2)
    n_low = sum(abs(c) < 0.2 for c in A["cos"])
    dr = s.get("H2_drive", {})
    if dr and all(isinstance(v, dict) and "reaches_sesoi" in v for v in dr.values()):  # analyze_d6 keys one target by permutation
        dr = {s.get("drive_target") or "T": dr}
    nD = {t: sum(x["reaches_sesoi"] for x in v.values()) for t, v in dr.items()}
    nD_low = {t: sum(x["reaches_sesoi"] for x in v.values() if abs(x["cos"]) < 0.2) for t, v in dr.items()}
    h2[f"{d}/{m}"] = dict(A_pass=nA, A_n=len(A["verdicts"]), A_pass_lowcos=nA_low, n_lowcos=n_low,
                          D_reaching_sesoi=nD, D_reaching_sesoi_lowcos=nD_low,
                          max_abs_cos=float(np.max(np.abs(A["cos"]))))
h2v = [v for v in h2.values() if not v.get("missing")]
out["H2"] = dict(per_model=h2, supported=all(v["A_pass"] == 0 and all(x == 0 for x in v["D_reaching_sesoi"].values()) for v in h2v) if h2v else None,
                 supported_lowcos=all(v["A_pass_lowcos"] == 0 and all(x == 0 for x in v["D_reaching_sesoi_lowcos"].values()) for v in h2v) if h2v else None)

# H3: for at least half of the directions, the weight-randomised recipe gives AUC >= 0.80
# (a direction counts if this holds in at least two of its three panel models)
h3 = {}
for d in PANEL:
    if d == "d4g":  # same calibration and Check 0b as D4: one direction, counted once
        continue
    a = [S[(d, m)]["check0b"]["random_twofold_auc"] for m in PANEL[d] if (d, m) in S and S[(d, m)].get("check0b")]
    h3[d] = dict(random_auc=a, real_auc=[S[(d, m)]["check0b"]["real_twofold_auc"] for m in PANEL[d] if (d, m) in S and S[(d, m)].get("check0b")],
                 counts=sum(x >= 0.80 for x in a) >= 2 if len(a) == 3 else None)
n3 = [v["counts"] for v in h3.values() if v["counts"] is not None]
out["H3"] = dict(per_direction=h3, n_directions=len(n3), n_counting=sum(n3),
                 supported=(sum(n3) >= len(n3) / 2) if len(n3) == len(PANEL) - 1 else None,
                 note="D4G shares D4's calibration and is not counted separately; D10 was not run")

# H4: norm-matched vs KL-matched effects, standardised by SESOI, over (direction, model) pairs passing Check 0
pairs = []
for (d, m), s in S.items():
    if not s["reproduced"]:
        continue
    if s.get("drive_target") is not None and s.get("drive_sesoi") is not None:
        prim = (s["drive_target"], s["drive_sesoi"])
    elif d in ("d3", "d4"):
        prim = ("D", 1.0 / s["L"])
    else:
        prim = PRIM[d]
    diffs = []
    for key, r in s["checkD"].items():
        if not key.startswith(prim[0] + ":"):
            continue
        sign = 1 if key.endswith("+") else -1
        nm, kl = (r.get("norm_matched") or {}).get("effect"), (r.get("rand") or {}).get("effect")
        if nm is None or kl is None:
            continue
        diffs.append(sign * (nm - kl) / prim[1])
    if diffs:
        pairs.append(dict(pair=f"{d}/{m}", diff=float(np.mean(diffs))))
if pairs:
    x = np.array([p["diff"] for p in pairs])
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    out["H4"] = dict(pairs=pairs, mean=float(x.mean()), ci=ci, supported=ci[0] > 0)

# H7: retrodiction of paper 4 for the CR axis in Qwen2.5-7B-Instruct and Qwen3-8B
h7 = {}
for m in ("q25-7b", "q3-8b"):
    s = S.get(("d3", m))
    if not s:
        continue
    A_fail = s["checkA"]["d"]["verdict"] == "fail"
    D_keys = [k for k in s["checkD"] if k.startswith("D:")]
    D_below = all(s["checkD"][k]["verdict"] in ("fail", "reversal") for k in D_keys)
    W_keys = [k for k in s["checkC_steering"] if k.startswith("W:")]
    W_not_beat = all(not (s["checkC_steering"][k]["diffs"].get("Nlen", {}).get("ci", [None])[0] or 0) > 0 for k in W_keys)
    h7[m] = dict(A_fails=A_fail, A=s["checkA"]["d"]["est"], A_ci=s["checkA"]["d"]["ci"], D_below_sesoi=D_below,
                 verdict=s["verdict"], verdict_ok=s["verdict"] in ("NUISANCE", "NOT SHOWN"), W_not_beyond_Nlen=W_not_beat,
                 retrodicted=A_fail and D_below and s["verdict"] in ("NUISANCE", "NOT SHOWN") and W_not_beat)
out["H7"] = h7
# H5: fewer than half of the process directions receive an overall VALID TRACKER or VALID DRIVER verdict
h5 = {}
for D, keys in PROCESS.items():
    ov = {k: out["overall"][k]["verdict"] for k in keys}
    h5[D] = dict(overall=ov, valid=any(v in VALID for v in ov.values()) if all(v is not None for v in ov.values()) else None)
done5 = [v["valid"] for v in h5.values() if v["valid"] is not None]
out["H5"] = dict(per_direction=h5, n_complete=len(done5), n_valid=sum(done5),
                 supported=(sum(done5) < len(PROCESS) / 2) if len(done5) == len(PROCESS) else None,
                 decided_early=(sum(done5) >= len(PROCESS) / 2) or (sum(done5) + len(PROCESS) - len(done5) < len(PROCESS) / 2))
# H6: at least two process directions receive NUISANCE in at least one panel model
h6 = {D: [f"{k}/{m}" for k in keys for m in PANEL[k] if (k, m) in S and S[(k, m)]["verdict"] == "NUISANCE"] for D, keys in PROCESS.items()}
n6 = sum(bool(v) for v in h6.values())
complete6 = all((k, m) in S for keys in PROCESS.values() for k in keys for m in PANEL[k])
out["H6"] = dict(nuisance=h6, n_directions=n6, supported=True if n6 >= 2 else (False if complete6 else None))
json.dump(out, open(os.path.join(HERE, "res", f"aggregate_{PHASE}.json"), "w"), indent=1, default=str)
print(json.dumps({k: out[k] for k in ("overall", "H1", "H7")}, indent=1, default=str)[:3000])
print("H3", out["H3"]["n_counting"], "/", out["H3"]["n_directions"], out["H3"]["supported"], "| H5", out["H5"]["n_valid"], "/", out["H5"]["n_complete"], out["H5"]["supported"], "| H6", out["H6"]["n_directions"], out["H6"]["supported"])
print("H2", out["H2"]["supported"], out["H2"]["supported_lowcos"], "H4", {k: out.get("H4", {}).get(k) for k in ("mean", "ci", "supported")})
