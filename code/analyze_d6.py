"""Paper 6 analysis for D6 (reasoning strength): Checks 0, A-E and the §6 verdicts for one model.

python analyze_d6.py <model key> [pilot|main]  ->  res/d6/<key>_<phase>_summary.json

A copy of analyze.py's procedures (same bootstrap, interpolation, verdict rules) with D6's target:
- T = thinking tokens, mean over 4 samples, capped chains counted at the cap (Stage 2b);
- Check A's primary statistic is the partial Spearman rho controlling for MATH level and subject
  (dummy-coded) and question token length (Stage 1 §3 D6); the plain rho is secondary;
- Check D's per-row change dT is computed in run_d6.py against the clean chains of the same samples
  (common random numbers); SESOI = 10% of the mean clean T on the Check D items;
- Check E reads the pushE file if it exists.
All CIs: item bootstrap, 2,000 resamples. Check D's primary CI is two-level (items, directions).
"""

import json
import math
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

KEY = sys.argv[1]
PHASE = sys.argv[2] if len(sys.argv) > 2 else "main"
DIR = "d6"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "res", DIR)
PRE = os.path.join(RES, f"{KEY}_{PHASE}")
NB = 2000
STAGE2 = json.load(open(os.path.join(HERE, "stage2_settings_d6.json"), encoding="utf-8"))
CFG = json.load(open(os.path.join(HERE, f"config_d6_{PHASE}.json"), encoding="utf-8"))
if "design_by_model" in CFG:  # main: Stage 1 §8 full design or the long-generation reduction, per model
    CFG = dict(CFG["common"], **CFG[CFG["design_by_model"][KEY]], design=CFG["design_by_model"][KEY])

info = json.load(open(PRE + "_cal.json", encoding="utf-8"))
L, ELL = info["L"], info["ell"]
from spec_d6 import gate_d6  # noqa: E402

if not os.path.exists(PRE + "_clean.jsonl"):  # the gate failed and the later stages were not run
    g_ = STAGE2["check0a"]["d6"]
    rep_ = gate_d6(info["check0a"], g_)
    summary = dict(direction=DIR, model=KEY, phase=PHASE, L=L, ell=ELL, g=info["g"], reproduced=rep_, check0a=info["check0a"],
                   check0b=info.get("check0b"), cal=info["cal_auc"], cal_rho_level=info.get("cal_rho_level"), cos=info["cos"], rk=info.get("rk"),
                   checkB=info.get("checkB"), verdict="NOT REPRODUCED" if not rep_ else "INCOMPLETE (no clean stage)")
    if summary["checkB"]:
        summary["checkB"].pop("proj", None)
    json.dump(summary, open(PRE + "_summary.json", "w"), indent=1, default=float)
    print(json.dumps(dict(direction=DIR, model=KEY, verdict=summary["verdict"], ratios=[info["check0a"]["ratio_minus"], info["check0a"]["ratio_plus"]])))
    sys.exit(0)
clean = [json.loads(l) for l in open(PRE + "_clean.jsonl", encoding="utf-8")]
push = [json.loads(l) for l in open(PRE + "_push.jsonl", encoding="utf-8")] if os.path.exists(PRE + "_push.jsonl") else []
pushE = [json.loads(l) for l in open(PRE + "_pushE.jsonl", encoding="utf-8")] if os.path.exists(PRE + "_pushE.jsonl") else []
dirs = np.load(PRE + "_dirs.npz") if os.path.exists(PRE + "_dirs.npz") else None
H = dict(np.load(PRE + "_clean_h.npz"))

from spec_d6 import ITEMS_D6  # noqa: E402

sets = ITEMS_D6()
A_ids = [x["id"] for x in (sets["pilot"] if PHASE == "pilot" else sets["A"])]
SUBJ = sorted({x["subject"] for x in sets["A"] + sets["pilot"] + sets["oracle"]})
SESOI_RHO = 0.20


# ---------------------------------------------------------------- statistics (as analyze.py)
def partial_rho(x, y, z):
    z = np.asarray(z, float)
    z = z[:, None] if z.ndim == 1 else z
    rx, ry = rankdata(x), rankdata(y)
    rz = np.column_stack([rankdata(c) for c in z.T])

    def res(a, b):
        b1 = np.c_[np.ones(len(a)), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = res(rx, rz), res(ry, rz)
    d = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d > 0 else float("nan")


def boot(fn, n, seed=0):
    rng = np.random.default_rng(seed)
    out = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    out = np.array([o for o in out if o is not None and not (isinstance(o, float) and math.isnan(o))])
    if len(out) < NB * 0.5:
        return [None, None]
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def verdict_pos(est, ci, sesoi, null=0.0):
    if est is None or ci[0] is None:
        return "inconclusive"
    if ci[0] > null and est >= sesoi:
        return "pass"
    if ci[1] < sesoi:
        return "fail"
    return "inconclusive"


def covariates(recs):
    """Level dummies (2..5), subject dummies (all but the first) and question token length. Ranking a
    dummy column inside partial_rho is an affine map of it, so dummies stay dummies."""
    Z = []
    for r in recs:
        z = [float(r["level"] == lv) for lv in range(2, 6)] + [float(r["subject"] == s) for s in SUBJ[1:]] + [float(r["n_problem"])]
        Z.append(z)
    Z = np.array(Z)
    keep = [j for j in range(Z.shape[1]) if len(set(Z[:, j])) > 1]  # drop constant columns (small pilot sets)
    return Z[:, keep]


# =================================================================== Check A
byid = {r["id"]: r for r in clean}
A = [byid[i] for i in A_ids if i in byid]
T = np.array([r["T"] for r in A], dtype=float)
Z = covariates(A)
Zlev = Z[:, :4] if Z.shape[1] >= 4 else Z
proj = {"d": np.array([r["P"] for r in A]), "Nlen": np.array([r["P_Nlen"] for r in A]), "Npc1": np.array([r["P_Npc1"] for r in A])}
HA = np.array([H[r["id"]] for r in A], dtype=np.float64)
if dirs is not None:
    proj["o"] = HA @ dirs["o"]
    du, Nl, Np = dirs["d"], dirs["Nlen"], dirs["Npc1"]
    Q, _ = np.linalg.qr(np.c_[Nl, Np])
    dperp = du - Q @ (Q.T @ du)
    proj["d_perp"] = HA @ (dperp / np.linalg.norm(dperp))
perm_P = np.array([r["P_perm"] for r in A])


def stat(P, idx=None, Zm=None):
    idx = np.arange(len(T)) if idx is None else idx
    Zm = Z if Zm is None else Zm
    t = T[idx]
    if len(set(t.tolist())) < 2:
        return None
    return partial_rho(P[idx], t, Zm[idx])


capped_any = np.array([r["n_capped"] > 0 for r in A])
checkA = dict(n=len(T), T_mean=float(T.mean()), T_median=float(np.median(T)), capped_chain_frac=float(np.mean([c for r in A for c in r["capped_s"]])),
              items_with_capped=int(capped_any.sum()), primary="partial_rho(level, subject, question length)",
              rho_T_level=float(spearmanr(T, [r["level"] for r in A])[0]), rho_T_qlen=float(spearmanr(T, [r["n_problem"] for r in A])[0]),
              T_by_level={lv: float(np.mean([r["T"] for r in A if r["level"] == lv])) for lv in range(1, 6) if any(r["level"] == lv for r in A)},
              accuracy=float(np.mean([c for r in A for c in r["correct_s"]])))
for k, P in proj.items():
    est = stat(P)
    ci = boot(lambda i: stat(P, i), len(T))
    sec = dict(rho=float(spearmanr(P, T)[0]), partial_rho_level_only=partial_rho(P, T, Zlev),
               rho_P_level=float(spearmanr(P, [r["level"] for r in A])[0]))
    if (~capped_any).sum() > 10:  # sensitivity: items none of whose clean chains hit the cap
        ii = np.where(~capped_any)[0]
        sec["partial_rho_uncapped_items"] = stat(P, ii)
    checkA[k] = dict(est=est, ci=ci, verdict=verdict_pos(est, ci, SESOI_RHO), **sec)
if "d_perp" in checkA and checkA["d"]["est"]:
    checkA["d_perp"]["frac_retained"] = checkA["d_perp"]["est"] / checkA["d"]["est"] if checkA["d"]["est"] else None
pv = [stat(perm_P[:, j]) for j in range(perm_P.shape[1])]
cos_perm = info["cos"]["d_perm"]
checkA["perm"] = dict(est=pv, verdicts=[verdict_pos(v, boot(lambda i, j=j: stat(perm_P[:, j], i), len(T)), SESOI_RHO) for j, v in enumerate(pv)],
                      cos=cos_perm)
lo_cos = [j for j, c in enumerate(cos_perm) if abs(c) < 0.2]
checkA["perm"]["n_pass_lowcos"] = sum(checkA["perm"]["verdicts"][j] == "pass" for j in lo_cos)

checkC_track = {}
for N in ("Nlen", "Npc1"):
    diff = lambda i, N=N: (stat(proj["d"], i) - stat(proj[N], i)) if stat(proj["d"], i) is not None else None
    checkC_track[N] = dict(est=stat(proj["d"]) - stat(proj[N]), ci=boot(diff, len(T)))


def c_status(diffs, NA):
    beats = all(diffs[N]["ci"][0] is not None and diffs[N]["ci"][0] > 0 for N in diffs)
    if beats:
        return "pass", False
    flag = any(NA[N] == "pass" and not (diffs[N]["ci"][0] is not None and diffs[N]["ci"][0] > 0) for N in diffs)
    return ("nuisance" if flag else "inconclusive"), flag


C_track, nuis_track = c_status(checkC_track, {N: checkA[N]["verdict"] for N in ("Nlen", "Npc1")})


# =================================================================== Check D
def interp_ref(kl_push, pts):
    pts = sorted((max(k, 1e-12), v) for k, v in pts if k is not None and v is not None)
    if not pts or kl_push is None:
        return None
    lk = math.log(max(kl_push, 1e-12))
    xs = [math.log(k) for k, _ in pts]
    if lk < xs[0] or lk > xs[-1]:
        return None
    for (x0, (k0, v0)), (x1, (k1, v1)) in zip(zip(xs, pts), zip(xs[1:], pts[1:])):
        if x0 <= lk <= x1:
            return v0 if x1 == x0 else v0 + (v1 - v0) * (lk - x0) / (x1 - x0)
    return None


def item_table(rec, sign):
    out = dict(push={}, rand={}, perm={})
    for r in rec["rows"]:
        kind = r["kind"]
        if kind in ("d", "Nlen", "Npc1", "o", "d_auth") and np.sign(r["k"]) == sign:
            out["push"][(kind, abs(r["k"]))] = (r["kl"], r["dT"])
        elif kind in ("rand", "perm"):
            out[kind].setdefault(r["j"], []).append((r["kl"], r["dT"], r["c"]))
    return out


def check_D(sesoi, sign, push_kind="d", k=1.0):
    items = []
    for rec in push:
        tab = item_table(rec, sign)
        if (push_kind, k) not in tab["push"]:
            continue
        kl, dt = tab["push"][(push_kind, k)]
        refs = {fam: {j: interp_ref(kl, [(a, b) for a, b, _ in v]) for j, v in tab[fam].items()} for fam in ("rand", "perm")}
        norm_ref = np.mean([b for v in tab["rand"].values() for a, b, c in v if c == k] or [np.nan])
        items.append(dict(id=rec["id"], kl=kl, dt=dt, refs=refs, norm_ref=norm_ref, tab=tab, T_clean=rec["T_clean"]))
    if not items:
        return None
    out = dict(n_items=len(items), sign=sign, kind=push_kind, k=k)
    for fam in ("rand", "perm"):
        R = sorted({j for x in items for j in x["refs"][fam]})
        M = np.array([[x["refs"][fam].get(j) if x["refs"][fam].get(j) is not None else np.nan for j in R] for x in items], float)
        dts = np.array([x["dt"] for x in items], float)
        ok = ~np.all(np.isnan(M), 1)
        n_ex = int((~ok).sum())
        dts, M = dts[ok], M[ok]
        if len(dts) == 0:
            out[fam] = dict(n_excluded=n_ex, n=0)
            continue
        est = float(np.mean(dts - np.nanmean(M, 1)))
        rng = np.random.default_rng(1)
        two, one = [], []
        for _ in range(NB):
            ii = rng.integers(0, len(dts), len(dts))
            jj = rng.integers(0, M.shape[1], M.shape[1])
            with np.errstate(all="ignore"):
                a = np.nanmean(M[np.ix_(ii, jj)], 1)
                b = np.nanmean(M[ii], 1)
            if not np.all(np.isnan(a)):
                two.append(np.nanmean(dts[ii] - a))
            one.append(np.nanmean(dts[ii] - b))
        out[fam] = dict(effect=est, ci=[float(np.quantile(two, 0.025)), float(np.quantile(two, 0.975))],
                        ci_item_only=[float(np.quantile(one, 0.025)), float(np.quantile(one, 0.975))], n=int(len(dts)), n_excluded=n_ex,
                        raw_dT=float(np.mean(dts)), mean_ref=float(np.nanmean(M)))
    out["kl_median"] = float(np.median([x["kl"] for x in items]))
    out["kl_range"] = [float(min(x["kl"] for x in items)), float(max(x["kl"] for x in items))]
    nr = np.array([x["dt"] - x["norm_ref"] for x in items if not np.isnan(x["norm_ref"])])
    out["norm_matched"] = dict(effect=float(nr.mean()) if len(nr) else None, ci=boot(lambda i: float(nr[i].mean()), len(nr)) if len(nr) else [None, None])

    def passes(fam):
        r = out.get(fam, {})
        if not r.get("n") or sesoi is None:
            return None
        e, (lo, hi) = r["effect"] * sign, sorted([c * sign for c in r["ci"]])
        if lo > 0 and e >= sesoi:
            return "pass"
        if hi < 0:
            return "reversal"
        if hi < sesoi:
            return "fail"
        return "inconclusive"
    out["verdict_rand"], out["verdict_perm"] = passes("rand"), passes("perm")
    v = (out["verdict_rand"], out["verdict_perm"])
    out["verdict"] = "pass" if v == ("pass", "pass") else ("reversal" if "reversal" in v else ("fail" if "fail" in v else "inconclusive"))
    out["_items"] = items
    return out


def check_E(Dres, sign):
    """E1 remaining fraction per later block and E2 at the primary push's KL, from the pushE file."""
    byE = {r["id"]: r for r in pushE}
    items = [x for x in Dres["_items"] if x["id"] in byE]
    if not items:
        return None
    full = float(np.mean([x["dt"] for x in items]))
    e1, e2 = {}, []
    for x in items:
        rows = byE[x["id"]]["rows"]
        for r in rows:
            if r["kind"] == "E1" and r["k"] == sign:
                e1.setdefault(r["lp"], []).append(r["dT"])
        pts = [(r["kl"], r["dT"]) for r in rows if r["kind"] == "E2" and r["k"] == sign]
        v = interp_ref(x["kl"], pts)
        if v is not None:
            e2.append(v)
    lps = sorted(e1)
    frac = {lp: (float(np.mean(e1[lp])) / full if full != 0 else None) for lp in lps}
    hand = next((lp for lp in lps if frac[lp] is not None and frac[lp] >= 0.5), None)
    e2_frac = float(np.mean(e2) / full) if e2 and full != 0 else None
    limit = ELL + 0.75 * (L - 1 - ELL)
    ok = hand is not None and hand <= limit and e2_frac is not None and e2_frac < 0.5
    return dict(n=len(items), full_effect=full, remaining_fraction=frac, handoff_layer=hand, handoff_limit=limit,
                e2_fraction=e2_frac, e2_n=len(e2), verdict="pass" if ok else "output_bias")


SESOI_D = 0.10 * float(np.mean([rec["T_clean"] for rec in push])) if push else None
checkD, checkCs, checkE, oracleD = {}, {}, {}, {}
for s in (1, -1):
    key = f"T:{'+' if s > 0 else '-'}"
    r = check_D(SESOI_D, s)
    if r is None:
        continue
    ctrlD = {N: check_D(SESOI_D, s, N) for N in ("Nlen", "Npc1", "o")}
    oracleD[key] = {kk: vv for kk, vv in ctrlD["o"].items() if kk != "_items"} if ctrlD["o"] else None
    diffs = {}
    for N in ("Nlen", "Npc1"):
        if ctrlD[N] is None or not ctrlD[N].get("rand", {}).get("n"):
            continue
        a = {x["id"]: x["dt"] - np.nanmean([v for v in x["refs"]["rand"].values() if v is not None] or [np.nan]) for x in r["_items"]}
        b = {x["id"]: x["dt"] - np.nanmean([v for v in x["refs"]["rand"].values() if v is not None] or [np.nan]) for x in ctrlD[N]["_items"]}
        ids = [i for i in a if i in b and not np.isnan(a[i]) and not np.isnan(b[i])]
        dd = np.array([a[i] - b[i] for i in ids]) * s
        diffs[N] = dict(est=float(dd.mean()) if len(dd) else None, ci=boot(lambda i: float(dd[i].mean()), len(dd)) if len(dd) else [None, None],
                        n=len(ids), N_effect=ctrlD[N]["rand"].get("effect"), N_verdict=ctrlD[N]["verdict"])
    if len(diffs) == 2:
        st, flag = c_status(diffs, {N: diffs[N]["N_verdict"] for N in diffs})
    else:
        st, flag = "inconclusive", False
    checkCs[key] = dict(status=st, nuisance_flag=flag, diffs=diffs)
    if r["verdict"] == "pass":
        checkE[key] = check_E(r, s)
    checkD[key] = {kk: vv for kk, vv in r.items() if kk != "_items"}
    checkD[key]["capped_frac_pushed"] = float(np.mean([c for rec in push for row in rec["rows"] if row["kind"] == "d" and row["k"] == s
                                                       for c in row["capped_s"]]))
    for k2 in CFG["k_d"]:
        if k2 == 1:
            continue
        rr = check_D(SESOI_D, s, "d", k2)
        if rr:
            checkD[key][f"exploratory_k{k2}"] = {kk: rr.get(kk) for kk in ("rand", "perm", "kl_median", "kl_range")}
    rr = check_D(SESOI_D, s, "d_auth", 1)
    if rr:
        checkD[key]["authors_strength"] = {kk: vv for kk, vv in rr.items() if kk != "_items"}
    for N in ("Nlen", "Npc1"):
        if ctrlD[N]:
            checkD[key][f"control_{N}"] = {kk: vv for kk, vv in ctrlD[N].items() if kk != "_items"}

# H2: the 8 pushed label-permuted directions at one gap (c = 1) against the Gaussian reference at matched KL
H2_drive = {}
if push and SESOI_D:
    per = {}
    for rec in push:
        tab = item_table(rec, 1)
        for j, pts in tab["perm"].items():
            hit = [(a, b) for a, b, c in pts if c == 1]
            if not hit:
                continue
            refs = [interp_ref(hit[0][0], [(a, b) for a, b, _ in v]) for v in tab["rand"].values()]
            refs = [x for x in refs if x is not None]
            if refs:
                per.setdefault(j, []).append(hit[0][1] - float(np.mean(refs)))
    for j, v in per.items():
        v = np.array(v)
        ci = boot(lambda i, v=v: float(v[i].mean()), len(v))
        e = float(v.mean())
        H2_drive[int(j)] = dict(effect=e, ci=ci, n=len(v), cos=cos_perm[int(j)],
                                reaches_sesoi=bool(abs(e) >= SESOI_D and ci[0] is not None and (ci[0] > 0 or ci[1] < 0)))

# grid bracketing (Stage 2b: the random grid must bracket every push's KL)
bracket = {}
for rec in push:
    rk = [r["kl"] for r in rec["rows"] if r["kind"] == "rand"]
    lo, hi = (min(rk), max(rk)) if rk else (None, None)
    for r in rec["rows"]:
        if r["kind"] in ("d", "Nlen", "Npc1", "o", "d_auth"):
            kk = f"{r['kind']}{r['k']:+g}"
            bracket.setdefault(kk, [0, 0, 0])
            bracket[kk][0] += 1
            bracket[kk][1] += int(lo is not None and r["kl"] < lo)
            bracket[kk][2] += int(hi is not None and r["kl"] > hi)
bracket = {k: dict(n=v[0], below=v[1], above=v[2]) for k, v in bracket.items()}


# ================================================================== verdicts
def drive_status():
    keys = list(checkD)
    vs = [checkD[k]["verdict"] for k in keys]
    if not vs:
        return "not_run", None
    passing = [k for k in keys if checkD[k]["verdict"] == "pass"]
    other_rev = any(checkD[k]["verdict"] == "reversal" for k in keys if k not in passing)
    if passing and not other_rev:
        full = [k for k in passing if checkCs.get(k, {}).get("status") == "pass" and (checkE.get(k) or {}).get("verdict") == "pass"]
        return "pass", (full or passing)[0]
    if all(v == "fail" for v in vs):
        return "fail", None
    return ("reversal" if any(v == "reversal" for v in vs) else "inconclusive"), None


gate = STAGE2["check0a"]["d6"]
c0 = info["check0a"]
reproduced = gate_d6(c0, gate)

dstat, dkey = drive_status()
o_A = checkA.get("o", {}).get("verdict")
o_D = None
if dkey and oracleD.get(dkey):
    o_D = oracleD[dkey]["verdict"]
elif oracleD:
    o_D = "pass" if any(v and v["verdict"] == "pass" for v in oracleD.values()) else "fail"
C_steer = checkCs.get(dkey, {}).get("status") if dkey else None
nuis_steer = any(v["nuisance_flag"] for v in checkCs.values())
E_ok = (checkE.get(dkey) or {}).get("verdict") if dkey else None
A_v = checkA["d"]["verdict"]
if not reproduced:
    verdict = "NOT REPRODUCED"
elif o_A != "pass" or o_D == "fail":
    verdict = "TARGET NOT LINEAR"
elif nuis_track or nuis_steer:
    verdict = "NUISANCE"
elif dstat == "pass" and E_ok == "output_bias":
    verdict = "OUTPUT-BIAS"
elif A_v == "pass" and C_track == "pass" and dstat == "pass" and C_steer == "pass" and E_ok == "pass":
    verdict = "VALID DRIVER"
elif A_v == "pass" and C_track == "pass":
    verdict = "VALID TRACKER"
elif (A_v == "pass" and C_track == "inconclusive") or (dstat == "pass" and C_steer == "inconclusive"):
    verdict = "NOT SPECIFIC"
else:
    verdict = "NOT SHOWN"
if dstat == "pass" and E_ok is None:
    verdict += " (Check E pending)"

B = info.get("checkB")
if B:
    pj = B["proj"]
    P = np.array(pj["P"]); yb = np.array(pj["label"])
    grp = np.array([i.split(":")[0] for i in pj["id"]])
    ug = np.unique(grp)

    def bauc(ii):
        idx = np.concatenate([np.where(grp == x)[0] for x in ug[ii]])
        s_, y_ = P[idx], yb[idx]
        n1, n0 = y_.sum(), len(y_) - y_.sum()
        if n1 == 0 or n0 == 0:
            return None
        r_ = rankdata(s_)
        return float((r_[y_ == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    B["ci"] = boot(bauc, len(ug))
    B["accepted"] = B["text_only_auc"] is not None and B["text_only_auc"] <= 0.65
    if not B["accepted"]:
        label = "NOT EVALUABLE (matched set rejected: text-only AUC > 0.65)"
    elif B["auc_d"] >= 0.70 and B["ci"][0] is not None and B["ci"][0] > 0.5:
        label = "LABEL SURVIVES MATCHING"
    elif B["ci"][1] is not None and B["ci"][1] < 0.5:
        label = "REVERSED"
    else:
        label = "LABEL DOES NOT SURVIVE"
    B.pop("proj", None)
else:
    label = "NOT RUN"
if label == "LABEL SURVIVES MATCHING" and verdict == "NOT SHOWN":
    verdict = "LABEL-ONLY"

summary = dict(direction=DIR, model=KEY, phase=PHASE, L=L, ell=ELL, g=info["g"], reproduced=reproduced, check0a=c0,
               drive_target="T", drive_sesoi=SESOI_D, cap=CFG["cap"], design=CFG.get("design", PHASE),
               check0b=info.get("check0b"), cal=info["cal_auc"], cal_rho_level=info.get("cal_rho_level"), cos=info["cos"], rk=info.get("rk"),
               checkA=checkA, checkC_tracking=dict(status=C_track, diffs=checkC_track), checkD=checkD, checkC_steering=checkCs,
               checkE=checkE, oracle_D=oracleD, H2_drive=H2_drive, grid_bracket=bracket, checkB=B, label_verdict=label,
               run_seconds=dict(push_kl_mean=float(np.mean([r.get("kl_seconds", 0) for r in push])) if push else None,
                                pushE_mean=float(np.mean([r["seconds"] for r in pushE])) if pushE else None),
               statuses=dict(A=A_v, C_track=C_track, D=dstat, D_key=dkey, C_steer=C_steer, E=E_ok, o_A=o_A, o_D=o_D),
               verdict=verdict)
json.dump(summary, open(PRE + "_summary.json", "w"), indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else float(x))
print(json.dumps(dict(direction=DIR, model=KEY, verdict=verdict, label=label, statuses=summary["statuses"],
                      A_d=checkA["d"]["est"], A_ci=checkA["d"]["ci"], sesoi_D=SESOI_D), default=str))
for k, v in checkD.items():
    print(k, {f: (round(v[f]["effect"], 1) if v.get(f, {}).get("effect") is not None else None, [round(c, 1) for c in v.get(f, {}).get("ci", []) if c is not None])
              for f in ("rand", "perm")}, v["verdict"], "excluded:", {f: v.get(f, {}).get("n_excluded") for f in ("rand", "perm")})
