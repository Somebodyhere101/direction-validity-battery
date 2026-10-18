"""Compact text report of one D7 run for the Stage 2b write-up. python report_d7.py <model key> [pilot|main]
Reads res/d7/<key>_<phase>_{labels,cal,summary}.json and the push file (bracketing, KL)."""
import collections
import json
import os
import sys

import numpy as np

KEY = sys.argv[1]
PHASE = sys.argv[2] if len(sys.argv) > 2 else "main"
HERE = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(HERE, "res", "d7", f"{KEY}_{PHASE}")
sys.path.insert(0, HERE)
import spec_d7 as S  # noqa: E402

I = S.ITEMS_D7()
lab = json.load(open(PRE + "_labels.json", encoding="utf-8"))
print("== labels per pool and type (known/unknown/other)")
for pool in S.label_pools(PHASE):
    c = {t: collections.Counter() for t in S.TYPES}
    for x in I[pool]:
        c[x["type"]][lab.get(S.ekey(x), {}).get("label")] += 1
    print(f"  {pool:8s}", {t: f"{v['known']}/{v['unknown']}/{v['other']}" for t, v in c.items()})
info = json.load(open(PRE + "_cal.json", encoding="utf-8"))
c0 = info["check0a"]
print("== calibration: ell", info["ell"], "of L", info["L"], "g %.2f" % info["g"], "|h| median %.1f" % info["resid_norm_median"],
      "cal AUC", {k: round(v, 3) for k, v in info["cal_auc"].items()}, "cos", {k: round(v, 3) for k, v in info["cos"].items() if k != "d_perm"})
print("   |cos(d, perm)| max %.2f median %.2f" % (np.max(np.abs(info["cos"]["d_perm"])), np.median(np.abs(info["cos"]["d_perm"]))))
print("   layer scan min-over-type AUC:", " ".join(f"{l}:{a:.2f}" for l, a in enumerate(c0["layer_scan_min_auc"])))
print("   at ell: min %.3f per type %s; cos(d, pooled DiM) %+.3f; counts %s" % (c0["sep_min_auc"], {k: round(v, 3) for k, v in c0["sep_per_type"].items()}, c0["cos_pooled"], c0["counts"]))
print("== Check 0a: alpha %.1f (= %.1f gaps; 2 x median question |h| %.1f)" % (c0["alpha"], c0["alpha_gaps"], c0["resid_norm_median_question"]))
for cls, r in c0["refusal"].items():
    print("  ", cls, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
print("   Check 0b:", info.get("check0b"))
if info.get("checkB"):
    print("== Check B:", {k: v for k, v in info["checkB"].items() if k != "proj"})
sm = json.load(open(PRE + "_summary.json", encoding="utf-8"))
print("== verdict:", sm["verdict"], "| label:", sm["label_verdict"], "| reproduced:", sm["reproduced"], "| statuses:", sm["statuses"])
A = sm["checkA"]
print("== Check A: n", A["n"], "T mean %.3f" % A["T_mean"], "primary", A["primary"], "rho(T, prompt len) %.3f" % A["rho_T_promptlen"])
for k in ("d", "o", "Nlen", "Npc1", "d_perp"):
    if k in A:
        a = A[k]
        print(f"   {k:6s} est {a['est']:+.3f} ci [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] {a['verdict']:12s} auc {a['auc']:.3f} rho {a['rho']:+.3f} partial {a['partial_rho']:+.3f}")
print("   perm verdicts:", collections.Counter(A["perm"]["verdicts"]))
print("   Check C tracking:", sm["checkC_tracking"]["status"], {k: (round(v["est"], 3), [round(c, 3) for c in v["ci"] if c is not None]) for k, v in sm["checkC_tracking"]["diffs"].items()})
d7 = sm.get("d7") or {}
for k in ("A_labels", "A_middle", "A_auc_by_type", "B_popularity_auc", "B_pairs_by_type"):
    if k in d7:
        print(f"   {k}:", json.dumps(d7[k], default=lambda x: round(x, 3))[:600])
if "A_lens" in d7:
    L_ = d7["A_lens"]
    print("   lens target: n", L_["n"], "rho(lens, correct) %.3f" % L_["rho_lens_correct"],
          {k: (round(v["rho"], 3), round(v["partial_rho"], 3)) for k, v in L_.items() if isinstance(v, dict)})
print("== Check D")
for key, v in sm["checkD"].items():
    print(f"   {key}: n {v['n_items']} kl median {v['kl_median']:.2f} verdict {v['verdict']}",
          {f: (round(v[f]["effect"], 3) if v.get(f, {}).get("effect") is not None else None, [round(c, 3) for c in v.get(f, {}).get("ci", [])],
               "excl", v.get(f, {}).get("n_excluded"), "rawdT", round(v.get(f, {}).get("raw_dT", float("nan")), 3)) for f in ("rand", "perm")},
          "norm-matched", round(v["norm_matched"]["effect"], 3) if v["norm_matched"]["effect"] is not None else None)
    for kk in ("exploratory_k0.5", "exploratory_k2.0"):
        if kk in v:
            print(f"      {kk}:", {f: round(v[kk][f]["effect"], 3) if v[kk].get(f, {}).get("effect") is not None else None for f in ("rand", "perm")}, "kl", round(v[kk]["kl_median"], 2))
    for N in ("control_Nlen", "control_Npc1"):
        if N in v:
            print(f"      {N}: verdict {v[N]['verdict']}", {f: round(v[N][f]["effect"], 3) if v[N].get(f, {}).get("effect") is not None else None for f in ("rand", "perm")})
print("   oracle D:", {k: (v["verdict"] if v else None) for k, v in sm["oracle_D"].items()})
print("   C steering:", {k: v["status"] for k, v in sm["checkC_steering"].items()})
print("   E:", {k: (v["handoff_layer"], round(v["e2_fraction"], 2) if v and v.get("e2_fraction") is not None else None, v["verdict"]) for k, v in sm["checkE"].items() if v})
# clean rates on the D items, and the KL range of every push kind vs the random grid
push = [json.loads(l) for l in open(PRE + "_push.jsonl", encoding="utf-8")]
by = collections.defaultdict(list)
for rec in push:
    s = next(int(np.sign(r["k"])) for r in rec["rows"] if r["kind"] == "d")
    c = rec["rows"][0]["direct"]
    by[s].append((c["D"], c["ref"], c["hal"]))
for s, v in by.items():
    v = np.array(v)
    print(f"   clean D items sign {s:+d}: n {len(v)} correct {v[:, 0].mean():.2f} refusal {v[:, 1].mean():.2f} halluc {v[:, 2].mean():.2f}")
kl = collections.defaultdict(list)
for rec in push:
    for r in rec["rows"][1:]:
        kl[(r["kind"], r.get("k", r.get("c")))].append(r.get("kl"))
for k in sorted(kl, key=str):
    if k[0] in ("d", "o", "Nlen", "Npc1") or (k[0] == "rand" and k[1] in (0.25, 4)):
        print("   KL", k, "median %.2f max %.2f" % (np.median(kl[k]), np.max(kl[k])))
