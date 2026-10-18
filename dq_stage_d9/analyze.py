"""Paper 6 analysis: Checks 0, A-E and the §6 verdicts for one (direction, model, phase).

python analyze.py <d1|d2|d3> <model key> [pilot|main]   ->  res/<dir>/<key>_<phase>_summary.json

All CIs: item bootstrap, 2,000 resamples, seed 0. Check D's primary CI is two-level (items, and
random directions within the reference); the item-only CI is reported alongside (§5).
"""

import json
import math
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

DIR, KEY = sys.argv[1], sys.argv[2]
PHASE = sys.argv[3] if len(sys.argv) > 3 else "main"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "res", DIR)
PRE = os.path.join(RES, f"{KEY}_{PHASE}")
NB = 2000
STAGE2 = json.load(open(os.path.join(HERE, "stage2_settings.json"), encoding="utf-8"))

info = json.load(open(PRE + "_cal.json", encoding="utf-8"))
L, ELL = info["L"], info["ell"]
clean = [json.loads(l) for l in open(PRE + "_clean.jsonl", encoding="utf-8")]
push = [json.loads(l) for l in open(PRE + "_push.jsonl", encoding="utf-8")] if os.path.exists(PRE + "_push.jsonl") else []
dirs = np.load(PRE + "_dirs.npz") if os.path.exists(PRE + "_dirs.npz") else None
H = dict(np.load(PRE + "_clean_h.npz"))

from specs import ITEMS  # noqa: E402
from spec_d5 import ITEMS_D5  # noqa: E402

ITEMS = dict(ITEMS, d5=ITEMS_D5(), d4=ITEMS["d3"],
             d4g=json.load(open(os.path.join(HERE, "data", "items_d4g.json"), encoding="utf-8")),
             d8=json.load(open(os.path.join(HERE, "data", "items_d8.json"), encoding="utf-8")),
             d9=json.load(open(os.path.join(HERE, "data", "items_d9.json"), encoding="utf-8")))
sets = ITEMS[DIR]
if PHASE == "pilot":
    A_ids = [x["id"] for x in sets["pilot"]]
else:
    A_ids = [x["id"] for x in sets["A"]]
meta = {x["id"]: x for k in ("A", "oracle", "pilot") for x in sets.get(k, [])}

judge = {}
for nm in ("_clean_judge.json", "_push_judge.json"):
    if os.path.exists(PRE + nm):
        judge[nm] = json.load(open(PRE + nm, encoding="utf-8"))


# ------------------------------------------------------------------ targets
def T_clean(rec, which="track"):
    if DIR == "d8":  # flip to the stated wrong option, only on items answered correctly without the belief
        return rec["belief"]["flip"] if rec["plain"]["correct"] else None
    if DIR == "d4g":  # CoT necessity: accuracy with chain of thought minus direct, 4 samples each
        return rec["T"]
    if DIR == "d1":  # Stage 2a: phrase matching primary (judge kappa 0.52 < 0.7); judge secondary
        if which == "judge":
            j = judge.get("_clean_judge.json", {}).get(f"{rec['id']}|0")
            return int(j >= 0.5) if j is not None else None
        return rec["gen"]["sub"]
    if DIR == "d2":
        return rec["gen"]["T"] if which == "track" else rec["gen"]["V"]
    if which == "W":
        return rec["free"]["W"]
    return rec["direct"]["D"]


def T_row(rec, j, row, target):
    if DIR == "d9":  # the early-exit answer follows the perturbed step (differs from y*)
        return row["gen"]["F"]
    if DIR == "d8":
        return row["belief"]["flip"] if rec["rows"][0]["plain"]["correct"] else None
    if DIR == "d4g":
        return row["direct"]["correct"]
    if DIR == "d1":
        if target == "sub":
            return row["gen"]["sub"]
        v = judge.get("_push_judge.json", {}).get(f"{rec['id']}|{j}")
        return int(v >= 0.5) if v is not None else None
    if DIR == "d2":
        return row["gen"]["V"] if target == "V" else row["gen"]["T"]
    if DIR == "d5":
        return row["gen"]["T"]
    if target == "W":
        return row["free"]["W"]
    return row["direct"]["D"]


# drive targets and SESOIs (§3); the first target is the primary one for the verdict
if DIR == "d1":
    DRIVE = [("sub", 0.10), ("judge", 0.10)]
    BINARY = True
elif DIR == "d2":
    DRIVE = [("V", 0.10), ("T", None)]
    BINARY = False
elif DIR == "d8":  # drive: flip rate, SESOI 10 points (§3)
    DRIVE = [("flip", 0.10)]
    BINARY = True
elif DIR == "d4g":  # drive: direct-answer accuracy, SESOI 10 points (§3)
    DRIVE = [("correct", 0.10)]
    BINARY = False
elif DIR == "d9":  # drive: rate at which the answer follows a perturbed step, SESOI 10 points (§0.5)
    DRIVE = [("F", 0.10)]
    BINARY = False  # tracking target (TTS) is continuous
elif DIR == "d5":  # SESOI: a 25% relative change of the clean rate (§3)
    base = [rec["rows"][0]["gen"]["T"] for rec in push] if push else [1.0]
    DRIVE = [("T", 0.25 * float(np.mean(base)))]
    BINARY = True
else:
    DRIVE = [("D", 1.0 / L), ("W", 0.10)]
    BINARY = False
SESOI_RHO, SESOI_AUC = 0.20, 0.60


# ---------------------------------------------------------------- statistics
def auc_fast(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


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


GROUPS = None  # D5: sentences are resampled by chain


def boot(fn, n, seed=0, groups=None):
    rng = np.random.default_rng(seed)
    out = []
    if groups is not None:
        ug = sorted(set(groups))
        members = {g: np.where(np.asarray(groups) == g)[0] for g in ug}
    for _ in range(NB):
        if groups is not None:
            pick = rng.integers(0, len(ug), len(ug))
            out.append(fn(np.concatenate([members[ug[k]] for k in pick])))
            continue
        out.append(fn(rng.integers(0, n, n)))
    out = np.array([o for o in out if o is not None and not (isinstance(o, float) and math.isnan(o))])
    if len(out) < NB * 0.5:
        return [None, None]
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def verdict_pos(est, ci, sesoi, null=0.0):
    """§5: pass if CI excludes null in the claimed (positive) sign and est >= sesoi; fail if the
    upper bound < sesoi; else inconclusive."""
    if est is None or ci[0] is None:
        return "inconclusive"
    if ci[0] > null and est >= sesoi:
        return "pass"
    if ci[1] < sesoi:
        return "fail"
    return "inconclusive"


# =================================================================== Check A
byid = {r["id"]: r for r in clean}
if DIR == "d5":  # one Check A unit per sentence of the model's own chain; covariates: position, previous length
    A = [dict(sj, id=f"{i}:{sj['j']}", chain=i) for i in A_ids if i in byid for sj in byid[i]["sentences"] if sj["prev_len"] is not None]
    T = np.array([a["T"] for a in A], dtype=float)
    nprompt = np.array([[a["pos"], a["prev_len"]] for a in A], dtype=float)
    GROUPS = [a["chain"] for a in A]
elif DIR == "d9":  # one Check A unit per numeric step of the model's own chain; covariates: position, step length
    A = [dict(sj, id=f"{i}:{sj['j']}", chain=i) for i in A_ids if i in byid for sj in byid[i]["sentences"]]
    T = np.array([a["T"] for a in A], dtype=float)
    nprompt = np.array([[a["pos"], a["step_len"]] for a in A], dtype=float)
    GROUPS = [a["chain"] for a in A]
else:
    A = [byid[i] for i in A_ids if i in byid and T_clean(byid[i]) is not None]
    T = np.array([T_clean(r) for r in A], dtype=float)
    nprompt = np.array([r["n_prompt"] for r in A], dtype=float)
proj = {"d": np.array([r["P"] for r in A]), "Nlen": np.array([r["P_Nlen"] for r in A]),
        "Npc1": np.array([r["P_Npc1"] for r in A])}
HA = np.array([H[r["id"]] for r in A], dtype=np.float64)
if dirs is not None:
    proj["o"] = HA @ dirs["o"]
    du, Nl, Np = dirs["d"], dirs["Nlen"], dirs["Npc1"]
    Q, _ = np.linalg.qr(np.c_[Nl, Np])
    dperp = du - Q @ (Q.T @ du)
    proj["d_perp"] = HA @ (dperp / np.linalg.norm(dperp))
perm_P = np.array([r["P_perm"] for r in A])
_len = nprompt[:, 0] if nprompt.ndim == 2 else nprompt
rho_T_len = float(spearmanr(T, _len)[0]) if len(set(T)) > 1 and len(set(_len)) > 1 else float("nan")
use_partial = abs(rho_T_len) > 0.1


def stat(P, idx=None):
    idx = np.arange(len(T)) if idx is None else idx
    t, p, z = T[idx], P[idx], nprompt[idx]
    if len(set(t.tolist())) < 2:
        return None
    if use_partial:
        return partial_rho(p, t, z)
    return auc_fast(p, t) if BINARY else float(spearmanr(p, t)[0])


SES_A = SESOI_RHO if (use_partial or not BINARY) else SESOI_AUC
NULL_A = 0.0 if (use_partial or not BINARY) else 0.5
checkA = dict(n=len(T), n_chains=len(set(GROUPS)) if GROUPS else None, T_mean=float(T.mean()), n_pos=int((T > 0.5).sum()) if BINARY else None,
              rho_T_promptlen=rho_T_len, primary="partial_rho" if use_partial else ("auc" if BINARY else "rho"))
for k, P in proj.items():
    est = stat(P)
    ci = boot(lambda i: stat(P, i), len(T), groups=GROUPS)
    sec = dict(rho=float(spearmanr(P, T)[0]) if len(set(T)) > 1 else None, partial_rho=partial_rho(P, T, nprompt),
               auc=auc_fast(P, T) if BINARY else None)
    checkA[k] = dict(est=est, ci=ci, verdict=verdict_pos(est, ci, SES_A, NULL_A), **sec)
if "d_perp" in checkA and checkA["d"]["est"]:
    checkA["d_perp"]["frac_retained"] = (checkA["d_perp"]["est"] - NULL_A) / (checkA["d"]["est"] - NULL_A) if checkA["d"]["est"] != NULL_A else None
pv = [stat(perm_P[:, j]) for j in range(perm_P.shape[1])]
cos_perm = info["cos"]["d_perm"]
if DIR == "d1":  # secondary: tracking against the (failed-validation) judge's labels
    tj = [T_clean(r, "judge") for r in A]
    ok = [i for i, v in enumerate(tj) if v is not None]
    if ok:
        checkA["d_vs_judge_T"] = dict(auc=auc_fast(proj["d"][ok], np.array(tj)[ok]), n=len(ok), T_mean=float(np.mean(np.array(tj)[ok])))
checkA["perm"] = dict(est=pv, verdicts=[verdict_pos(v, boot(lambda i, j=j: stat(perm_P[:, j], i), len(T), groups=GROUPS), SES_A, NULL_A)
                                        for j, v in enumerate(pv)], cos=cos_perm)

# Check C (tracking): paired differences d - N
checkC_track = {}
for N in ("Nlen", "Npc1"):
    diff = lambda i, N=N: (stat(proj["d"], i) - stat(proj[N], i)) if stat(proj["d"], i) is not None else None
    est = stat(proj["d"]) - stat(proj[N])
    checkC_track[N] = dict(est=est, ci=boot(diff, len(T), groups=GROUPS))


def c_status(dA, NA, diffs):
    """§5 Check C for one claimed kind."""
    beats = all(diffs[N]["ci"][0] is not None and diffs[N]["ci"][0] > 0 for N in diffs)
    if beats:
        return "pass", False
    flag = any(NA[N] == "pass" and not (diffs[N]["ci"][0] is not None and diffs[N]["ci"][0] > 0) for N in diffs)
    return ("nuisance" if flag else "inconclusive"), flag


C_track, nuis_track = c_status(checkA["d"]["verdict"], {N: checkA[N]["verdict"] for N in ("Nlen", "Npc1")}, checkC_track)


# =================================================================== Check D
def interp_ref(kl_push, pts):
    """Linear interpolation of dT in log KL between the two grid norms that bracket kl_push.
    pts: list of (kl, dT) for one random direction. Returns None if outside the bracket."""
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


def item_table(rec, target, sign):
    """Per item: dT and KL of every push of the given sign, and the per-direction reference curves."""
    rows = rec["rows"]
    t0 = T_row(rec, 0, rows[0], target)
    if t0 is None:
        return None
    out = dict(push={}, rand={}, perm={}, E1={}, E2=[])
    for j, r in enumerate(rows):
        t = T_row(rec, j, r, target)
        if t is None:
            continue
        dt = t - t0
        kind = r["kind"]
        if kind in ("d", "Nlen", "Npc1", "o", "d_auth") and np.sign(r["k"]) == sign:
            out["push"][(kind, abs(r["k"]))] = (r.get("kl"), dt)
        elif kind == "rand":
            out["rand"].setdefault(r["j"], []).append((r.get("kl"), dt, r["c"]))
        elif kind == "perm":
            out["perm"].setdefault(r["j"], []).append((r.get("kl"), dt, r["c"]))
        elif kind == "E1" and r["k"] == sign:
            out["E1"][r["lp"]] = dt
        elif kind == "E2" and r["k"] == sign:
            out["E2"].append((r.get("kl"), dt))
    return out


def check_D(target, sesoi, sign, push_kind="d", k=1.0):
    items = []
    for rec in push:
        it = meta[rec["id"]]
        if DIR == "d1" and (sign == 1) != (it["label"] == 0):
            continue
        if DIR == "d9" and (sign == 1) != (rec["rows"][0]["gen"]["F"] == 0):  # +d where the clean answer
            continue  # stays y* (engagement), -d where it follows the perturbed step (disengagement)
        tab = item_table(rec, target, sign)
        if tab is None or (push_kind, k) not in tab["push"]:
            continue
        kl, dt = tab["push"][(push_kind, k)]
        refs = {}
        for fam in ("rand", "perm"):
            refs[fam] = {j: interp_ref(kl, [(a, b) for a, b, _ in v]) for j, v in tab[fam].items()}
        norm_ref = np.mean([b for v in tab["rand"].values() for a, b, c in v if c == k] or [np.nan])
        items.append(dict(id=rec["id"], kl=kl, dt=dt, refs=refs, norm_ref=norm_ref, tab=tab))
    if not items:
        return None
    out = dict(n_items=len(items), target=target, sign=sign, kind=push_kind, k=k)
    claimed = sign  # every direction here claims a positive relation between projection and T
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
        eff = lambda ii, jj: float(np.mean(dts[ii] - np.nanmean(M[np.ix_(ii, jj)], 1))) if not np.all(np.isnan(M[np.ix_(ii, jj)])) else None
        allj = np.arange(M.shape[1])
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
        ci2 = [float(np.quantile(two, 0.025)), float(np.quantile(two, 0.975))]
        ci1 = [float(np.quantile(one, 0.025)), float(np.quantile(one, 0.975))]
        out[fam] = dict(effect=est, ci=ci2, ci_item_only=ci1, n=int(len(dts)), n_excluded=n_ex,
                        raw_dT=float(np.mean(dts)), mean_ref=float(np.nanmean(M)))
    out["kl_median"] = float(np.median([x["kl"] for x in items if x["kl"] is not None]))
    nr = np.array([x["dt"] - x["norm_ref"] for x in items if not np.isnan(x["norm_ref"])])
    out["norm_matched"] = dict(effect=float(nr.mean()) if len(nr) else None,
                               ci=boot(lambda i: float(nr[i].mean()), len(nr)) if len(nr) else [None, None])

    def passes(fam):
        r = out.get(fam, {})
        if not r.get("n") or sesoi is None:
            return None
        e, (lo, hi) = r["effect"] * claimed, sorted([c * claimed for c in r["ci"]])
        if lo > 0 and e >= sesoi:
            return "pass"
        if hi < 0:
            return "reversal"
        if hi < sesoi:
            return "fail"
        return "inconclusive"
    out["verdict_rand"], out["verdict_perm"] = passes("rand"), passes("perm")
    v = (out["verdict_rand"], out["verdict_perm"])
    out["verdict"] = "pass" if v == ("pass", "pass") else ("reversal" if "reversal" in v else
                                                          ("fail" if "fail" in v else "inconclusive"))
    out["_items"] = items
    return out


def check_E(Dres, target, sign):
    items = [x for x in Dres["_items"] if x["tab"]["E1"]]
    if not items:
        return None
    full = np.mean([x["dt"] for x in items])
    lps = sorted({lp for x in items for lp in x["tab"]["E1"]})
    frac = {lp: float(np.mean([x["tab"]["E1"][lp] for x in items if lp in x["tab"]["E1"]]) / full) if full != 0 else None for lp in lps}
    hand = next((lp for lp in lps if frac[lp] is not None and frac[lp] >= 0.5), None)
    e2 = [interp_ref(x["kl"], x["tab"]["E2"]) for x in items]
    e2 = [v for v in e2 if v is not None]
    e2_frac = float(np.mean(e2) / full) if e2 and full != 0 else None
    limit = ELL + 0.75 * (L - 1 - ELL)
    ok = hand is not None and hand <= limit and e2_frac is not None and e2_frac < 0.5
    return dict(n=len(items), full_effect=float(full), remaining_fraction=frac, handoff_layer=hand, handoff_limit=limit,
                e2_fraction=e2_frac, e2_n=len(e2), verdict="pass" if ok else "output_bias")


signs = sorted({int(np.sign(r["k"])) for rec in push for r in rec["rows"] if r["kind"] == "d"})
checkD, checkCs, checkE, oracleD = {}, {}, {}, {}
for target, ses in DRIVE:
    for s in signs:
        key = f"{target}:{'+' if s > 0 else '-'}"
        r = check_D(target, ses, s)
        if r is None:
            continue
        ctrlD = {N: check_D(target, ses, s, N) for N in ("Nlen", "Npc1", "o")}
        oracleD[key] = {kk: vv for kk, vv in (ctrlD["o"] or {}).items() if kk != "_items"} if ctrlD["o"] else None
        diffs = {}
        for N in ("Nlen", "Npc1"):
            if ctrlD[N] is None or "rand" not in ctrlD[N] or not ctrlD[N]["rand"].get("n"):
                continue
            a = {x["id"]: x["dt"] - np.nanmean([v for v in x["refs"]["rand"].values() if v is not None] or [np.nan]) for x in r["_items"]}
            b = {x["id"]: x["dt"] - np.nanmean([v for v in x["refs"]["rand"].values() if v is not None] or [np.nan]) for x in ctrlD[N]["_items"]}
            ids = [i for i in a if i in b and not np.isnan(a[i]) and not np.isnan(b[i])]
            dd = np.array([a[i] - b[i] for i in ids]) * s
            diffs[N] = dict(est=float(dd.mean()) if len(dd) else None, ci=boot(lambda i: float(dd[i].mean()), len(dd)) if len(dd) else [None, None],
                            n=len(ids), N_effect=ctrlD[N]["rand"].get("effect"), N_verdict=ctrlD[N]["verdict"])
        if len(diffs) == 2:
            st, flag = c_status(r["verdict"], {N: diffs[N]["N_verdict"] for N in diffs}, diffs)
        else:
            st, flag = "inconclusive", False
        checkCs[key] = dict(status=st, nuisance_flag=flag, diffs=diffs)
        if r["verdict"] == "pass":
            checkE[key] = check_E(r, target, s)
        checkD[key] = {kk: vv for kk, vv in r.items() if kk != "_items"}
        for k2 in (0.5, 2.0):
            rr = check_D(target, ses, s, "d", k2)
            if rr:
                checkD[key][f"exploratory_k{k2}"] = {kk: rr.get(kk) for kk in ("rand", "perm", "kl_median")}
        ka = sorted({abs(r["k"]) for rec in push for r in rec["rows"] if r["kind"] == "d_auth"})
        if ka:  # secondary: the authors' published strength
            rr = check_D(target, ses, s, "d_auth", ka[0])
            if rr:
                checkD[key]["authors_strength"] = {kk: vv for kk, vv in rr.items() if kk != "_items"}
        for N in ("Nlen", "Npc1"):
            if ctrlD[N]:
                checkD[key][f"control_{N}"] = {kk: vv for kk, vv in ctrlD[N].items() if kk != "_items"}

# H2: the 8 pushed label-permuted directions evaluated as drive claims (push c = 1, i.e. one gap),
# against the Gaussian reference at matched KL; their tracking verdicts are in checkA["perm"]
H2_drive = {}
for target, ses in DRIVE:
    if ses is None:
        continue
    per = {}
    for rec in push:
        tab = item_table(rec, target, 1)
        if tab is None:
            continue
        for j, pts in tab["perm"].items():
            hit = [(a, b) for a, b, c in pts if c == 1]
            if not hit or hit[0][0] is None:
                continue
            refs = [interp_ref(hit[0][0], [(a, b) for a, b, _ in v]) for v in tab["rand"].values()]
            refs = [r for r in refs if r is not None]
            if refs:
                per.setdefault(j, []).append(hit[0][1] - float(np.mean(refs)))
    H2_drive[target] = {}
    for j, v in per.items():
        v = np.array(v)
        ci = boot(lambda i, v=v: float(v[i].mean()), len(v))
        e = float(v.mean())
        big = abs(e) >= ses and ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
        H2_drive[target][int(j)] = dict(effect=e, ci=ci, n=len(v), cos=cos_perm[int(j)], reaches_sesoi=bool(big))


# ================================================================== verdicts
def drive_status(target):
    keys = [k for k in checkD if k.startswith(target + ":")]
    vs = [checkD[k]["verdict"] for k in keys]
    if not vs:
        return "not_run", None
    passing = [k for k in keys if checkD[k]["verdict"] == "pass"]
    other_rev = any(checkD[k]["verdict"] == "reversal" for k in keys if k not in passing)
    if passing and not other_rev:
        # Stage 2: if both signs pass D, C-steering and E are read on a sign where both pass, if any
        full = [k for k in passing if checkCs.get(k, {}).get("status") == "pass" and (checkE.get(k) or {}).get("verdict") == "pass"]
        return "pass", (full or passing)[0]
    if all(v == "fail" for v in vs):
        return "fail", None
    return ("reversal" if any(v == "reversal" for v in vs) else "inconclusive"), None


gate = STAGE2["check0a"][DIR]
c0 = info["check0a"]
if DIR == "d1":
    reproduced = (c0["jbb_baseline"] - c0["jbb_ablation"] >= gate["ablation_drop_rel_min"] * c0["jbb_baseline"]) and \
                 (c0["harmless_actadd"] - c0["harmless_baseline"] >= gate["actadd_rise_min"])
elif DIR == "d2":
    reproduced = c0["mean_heldout_acc"] >= gate["mean_heldout_acc_min"]
elif DIR == "d8":  # reproduction from d8_extract.py steer0a/judge0a
    c0 = json.load(open(PRE + "_0a.json")) if os.path.exists(PRE + "_0a.json") else c0
    reproduced = (not c0.get("pending")) and c0["trait_steered"] - c0["trait_baseline"] >= gate["trait_rise_min"]
elif DIR == "d9":  # Stage 2b D9 §5: released-vector cosine (R1 models) and the engagement / disengagement tests
    adv = [c0.get(t, {}).get("adv_d_vs_rand") for t in ("ET", "DT")]
    reproduced = (c0.get("cos_released") is None or c0["cos_released"] >= gate["cos_released_min"]) and \
                 any(a is not None and a >= gate["flip_adv_min"] for a in adv)
elif DIR == "d5":
    reproduced = (c0["kw_authors_strength"] >= gate["kw_rel_min"] * c0["kw_baseline"]) and                  (c0.get("cos_released_ft") is None or c0["cos_released_ft"] >= gate["cos_released_min"])
else:
    if DIR == "d3":
        reproduced = c0["cal_auc"] >= gate["cal_auc_min"]
    else:  # d4, d4g: LiReFs
        reproduced = (c0["twofold_auc"] >= gate["twofold_auc_min"]
                      and c0["heldout_reason_intervened"] - c0["heldout_reason_baseline"] >= gate["heldout_gain_min"])

prim = DRIVE[0][0]
dstat, dkey = drive_status(prim)
o_A = checkA.get("o", {}).get("verdict")
o_D = None
if dkey and oracleD.get(dkey):
    o_D = oracleD[dkey]["verdict"]
elif oracleD:
    o_D = "pass" if any(v and v["verdict"] == "pass" for k, v in oracleD.items() if k.startswith(prim + ":")) else "fail"
C_steer = checkCs.get(dkey, {}).get("status") if dkey else None
nuis_steer = any(v["nuisance_flag"] for k, v in checkCs.items() if k.startswith(prim + ":"))
E_ok = checkE.get(dkey, {}).get("verdict") if dkey else None

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

B = info.get("checkB")
if B:
    pj = B["proj"]
    P = np.array(pj["P"]); yb = np.array(pj["label"])
    grp = np.array([i.split(":")[0] for i in pj["id"]])
    ug = np.unique(grp)

    def bauc(ii):
        g = ug[ii]
        m_ = np.isin(grp, g)
        # resample pairs with replacement
        idx = np.concatenate([np.where(grp == x)[0] for x in g])
        return auc_fast(P[idx], yb[idx])
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
               drive_target=DRIVE[0][0], drive_sesoi=DRIVE[0][1],
               check0b=info.get("check0b"), cal=info["cal_auc"], cos=info["cos"], checkA=checkA,
               checkC_tracking=dict(status=C_track, diffs=checkC_track), checkD=checkD, checkC_steering=checkCs,
               checkE=checkE, oracle_D=oracleD, H2_drive=H2_drive, checkB=B, label_verdict=label,
               statuses=dict(A=A_v, C_track=C_track, D=dstat, D_key=dkey, C_steer=C_steer, E=E_ok, o_A=o_A, o_D=o_D),
               verdict=verdict)
json.dump(summary, open(PRE + "_summary.json", "w"), indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else float(x))
print(json.dumps(dict(direction=DIR, model=KEY, verdict=verdict, label=label, statuses=summary["statuses"],
                      A_d=checkA["d"]["est"], A_ci=checkA["d"]["ci"]), default=str))
for k, v in checkD.items():
    print(k, {f: (round(v[f]["effect"], 4) if v.get(f, {}).get("effect") is not None else None, [round(c, 4) for c in v.get(f, {}).get("ci", []) if c is not None])
              for f in ("rand", "perm")}, v["verdict"], "excluded:", {f: v.get(f, {}).get("n_excluded") for f in ("rand", "perm")})
