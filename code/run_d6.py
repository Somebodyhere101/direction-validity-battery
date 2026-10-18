"""Paper 6, D6 (reasoning strength) runs.
python run_d6.py <model key> <cal|clean|push|pushE|timing|lamsweep|selftest> [pilot|main] [extra]

A copy of run.py's stages for D6, whose target needs sampled long generations (p6lib_d6):
cal       rebuild the authors' vectors at every layer; d = the vector at their single-layer index;
          N_len, N_pc1, N_perm over the five levels; Check B; Check 0a (the authors' every-layer
          steering at their strength on 20 MATH-500 problems, 4 samples); Check 0b (weights
          re-initialised last).
clean     projections and the tracking target on the Check A and oracle items: thinking tokens of
          4 samples per item (cap from the config; answer_max tokens after </think> kept for accuracy).
push      Check C/D on the Check D items. Generation (every item's chains in one continuous batch):
          4 samples for d, the controls and the authors' strength, one sample per random and permuted
          direction and norm (sample index j mod 4), each against the same sample's clean chain.
          Then teacher-forced KL of every distinct push on the clean sample 0's chain.
pushE     Check E (E1 re-imposition, E2 last-block pushes) on the Check E items, 4 samples, lockstep
          batches with the clean twin in row 0; run only for the signs that pass Check D (extra = "+",
          "-" or "+-").
timing    run-time probe: clean-style generation of every second pilot problem (20, 4 per level) x 4
          samples; extra = cap. Keeps only timing, memory and thinking lengths.
lamsweep  pilot only: the authors' every-layer steering at other strengths (extra = "0.5,1").
selftest  pilot only: continuous batching (few slots, a forced out-of-memory eviction) against lockstep
          batches on the same rows and noise; the chains must agree.
Resumable per item. Env: P6_MEM_FRACTION; SLOTS (rows in the continuous batch; default from memory).
"""

import collections
import json
import os
import re
import sys
import time

# the KV buffers change size often; expandable segments keep the allocator from fragmenting (tested on DQ:
# 6.6 GB reserved-but-unused under the default allocator against 0.8 GB in a cache-like allocation pattern)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from p6lib import Model, auc, gaussian, oracle, twofold_auc, unit
from p6lib_d6 import install_fast_decode, kv_bytes_per_token, new_cache, rows_for, sample_generate, stream_generate, tf_kl, use_sdpa
from spec_d6 import D6, ITEMS_D6, gate_d6, n_len_levels, n_pc1_levels, n_perm_levels

KEY, STAGE = sys.argv[1], sys.argv[2]
PHASE = sys.argv[3] if len(sys.argv) > 3 else "main"
EXTRA = sys.argv[4] if len(sys.argv) > 4 else ""
DIR = "d6"
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, f"config_d6_{PHASE}.json"), encoding="utf-8"))
if "design_by_model" in CFG:  # main: Stage 1 §8 full design or the long-generation reduction, per model
    CFG = dict(CFG["common"], **CFG[CFG["design_by_model"][KEY]], design=CFG["design_by_model"][KEY])
RES = os.path.join(HERE, "res", DIR)
os.makedirs(RES, exist_ok=True)
PRE = os.path.join(RES, f"{KEY}_{PHASE}")
T0 = time.time()
NS = CFG["n_samples"]
CAP = int(EXTRA) if STAGE == "timing" and EXTRA else CFG["cap"]


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = D6()
m = Model(KEY, thinking=True)
install_fast_decode(m)  # single-token decoding: grouped-query attention without repeat_kv (p6lib_d6)
log(f"{m.name} L={m.L} d={m.d} stage={STAGE} phase={PHASE} cap={CAP}")
S = spec.items(PHASE)


def n_slots(n_prompt, answer_max=0):
    if os.environ.get("SLOTS"):
        return int(os.environ["SLOTS"])
    return rows_for(m, n_prompt + CAP + answer_max)


def text_only_auc(sets, labels, groups):
    """Leakage baseline (Check B), as run.py: bag-of-tokens logistic regression, 5-fold grouped by pair."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    X = CountVectorizer(token_pattern=r"\S+").fit_transform([" ".join(map(str, s)) for s in sets])
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, labels, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    return auc(p, labels)


def vec(v):
    return torch.tensor(np.asarray(v), dtype=torch.float32)


def authors_add(R_all, lam):
    """The authors' steering: lambda * r^(i) at the input of decoder layer i for i >= 1, i.e. at the
    output of block i - 1 (their r^(0), at the embedding output, is zero: the last prompt token is
    always the same token). R_all[j] is the vector at the output of block j."""
    return {j: vec(lam * R_all[j]) for j in range(m.L - 1)}


# ---------------------------------------------------------------- answers (secondary)
def boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    dep = 0
    for k in range(j, len(text)):
        dep += {"{": 1, "}": -1}.get(text[k], 0)
        if dep == 0:
            return text[j + 1:k]
    return None


def norm_ans(a):
    if a is None:
        return None
    a = a.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", a)
    a = re.sub(r"\s+", "", a).rstrip(".").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\$", "")
    return a


def answer_of(toks):
    te = m.tok.convert_tokens_to_ids("</think>")
    if te not in toks:
        return None
    return boxed(m.tok.decode(toks[toks.index(te) + 1:], skip_special_tokens=True))


def steer_rows(its, lams, R_all):
    """Check 0a / lamsweep rows: every item x lambda x sample, the authors' steering from position 0."""
    rows, lab = [], []
    for it in its:
        ids = spec.prompt_ids(m, it)[0]
        for lm in lams:
            add = authors_add(R_all, lm) if lm != 0 else None
            for s in range(NS):
                rows.append(dict(ids=ids, key=(it["id"], s), add=add, start="all"))
                lab.append((it["id"], lm))
    return rows, lab


# ======================================================================= cal
if STAGE == "cal":
    cal = spec.calibrate(m, log)
    du = unit(cal["d"])
    H, y, lev = cal["H"], np.asarray(cal["y"]), np.asarray(cal["levels"])
    g = float(np.linalg.norm(cal["d"]))  # gap along d with the authors' level-mean weighting (= |d|)
    g_pooled = float((H[y == 1] @ du).mean() - (H[y == 0] @ du).mean())
    Nl = n_len_levels(H, lev, cal["lengths"])
    Np = n_pc1_levels(H, lev, du)
    perms = n_perm_levels(H, lev, CFG["n_perm"], seed=6)
    from scipy.stats import spearmanr
    info = dict(ell=cal["ell"], g=g, g_pooled=g_pooled, L=m.L, model=m.name, d_norm=float(np.linalg.norm(cal["d"])),
                level_counts=cal["level_counts"], authors_lambda=spec.authors_lambda(m),
                cal_auc=dict(d=auc(H @ du, y), Nlen=auc(H @ Nl, y), Npc1=auc(H @ Np, y)),
                cal_rho_level=dict(d=float(spearmanr(H @ du, lev)[0]), Nlen=float(spearmanr(H @ Nl, lev)[0]),
                                   Npc1=float(spearmanr(H @ Np, lev)[0]), length=float(spearmanr(cal["lengths"], lev)[0])),
                cos=dict(d_Nlen=float(du @ Nl), d_Npc1=float(du @ Np), Nlen_Npc1=float(Nl @ Np), d_perm=[float(du @ p) for p in perms]),
                resid_norm_median=float(np.median(np.linalg.norm(H, axis=1))),
                rk=dict(norms=cal["rk_norms"], cos=cal["rk_cos"], cos_pairwise_by_layer=cal["cos_pairwise_by_layer"],
                        best_cos_block=int(np.argmax(cal["cos_pairwise_by_layer"][1:])) + 1))
    log(f"ell={cal['ell']} g={g:.2f} (pooled {g_pooled:.2f}) |h|~{info['resid_norm_median']:.0f} cal AUC d={info['cal_auc']['d']:.3f} "
        f"Nlen={info['cal_auc']['Nlen']:.3f} Npc1={info['cal_auc']['Npc1']:.3f} cos(d,Nlen)={info['cos']['d_Nlen']:+.3f} "
        f"cos(d,Npc1)={info['cos']['d_Npc1']:+.3f} rho(P,level)={info['cal_rho_level']['d']:+.3f}")
    save = {k: v for k, v in cal.items() if k != "H"}
    save.update(Nlen=Nl, Npc1=Np, perm=perms, info=info, g=g)
    torch.save(save, PRE + "_cal.pt")
    json.dump(info, open(PRE + "_cal.json", "w"), indent=1, default=float)
    # Check B: matched label set (evaluation only)
    Bs = spec.B_set(m, S["B"], S["B_pairs"])
    if Bs:
        hb = np.array([m.acts(b["ids"], [cal["ell"]], [b["start"]])[cal["ell"]][0] for b in Bs], dtype=np.float64)
        yb = np.array([b["label"] for b in Bs])
        grp = [b["id"].split(":")[0] for b in Bs]
        info["checkB"] = dict(n_pairs=len(Bs) // 2, auc_d=auc(hb @ du, yb), auc_Nlen=auc(hb @ Nl, yb), auc_Npc1=auc(hb @ Np, yb),
                              text_only_auc=text_only_auc([m.tok(b["text"], add_special_tokens=False)["input_ids"] for b in Bs], yb, grp),
                              length_auc=auc([len(b["ids"]) for b in Bs], yb),
                              proj=dict(id=[b["id"] for b in Bs], label=yb.tolist(), P=(hb @ du).tolist(), level=[b["level"] for b in Bs]))
        log("Check B", {k: v for k, v in info["checkB"].items() if k != "proj"})
    json.dump(info, open(PRE + "_cal.json", "w"), indent=1, default=float)
    # Check 0a: the authors' every-layer steering at their strength, 20 MATH-500 problems x 4 samples
    lam = spec.authors_lambda(m)
    its = S["calib0a"]
    rows, lab = steer_rows(its, (0.0, -lam, lam), cal["R_all"])
    t0 = time.time()
    res = stream_generate(m, rows, CAP, slots=n_slots(max(len(r["ids"]) for r in rows)), log=log)
    c0 = dict(lambda_=lam, cap=CAP, n_items=len(its), seconds=time.time() - t0, per_item={})
    for lm in (0.0, -lam, lam):
        c0[f"T_mean_{lm:+g}"] = float(np.mean([r["T"] for r, (_, l2) in zip(res, lab) if l2 == lm]))
        c0[f"capped_frac_{lm:+g}"] = float(np.mean([r["capped"] for r, (_, l2) in zip(res, lab) if l2 == lm]))
    for it in its:
        c0["per_item"][it["id"]] = {f"{lm:+g}": [r["T"] for r, (i2, l2) in zip(res, lab) if l2 == lm and i2 == it["id"]] for lm in (0.0, -lam, lam)}
    c0["ratio_minus"] = c0[f"T_mean_{-lam:+g}"] / c0["T_mean_+0"]
    c0["ratio_plus"] = c0[f"T_mean_{lam:+g}"] / c0["T_mean_+0"]
    info["check0a"] = c0
    log(f"0a authors' steering lambda={lam}: T(0)={c0['T_mean_+0']:.0f} T(-)={c0[f'T_mean_{-lam:+g}']:.0f} "
        f"T(+)={c0[f'T_mean_{lam:+g}']:.0f} (ratios {c0['ratio_minus']:.2f}, {c0['ratio_plus']:.2f}); {c0['seconds']:.0f}s")
    json.dump(info, open(PRE + "_cal.json", "w"), indent=1, default=float)
    # Check 0b: same recipe at the same layer and position, real vs weight-randomised model
    Hr, yr = spec.cal_acts(m, cal)
    info["check0b"] = dict(real_twofold_auc=twofold_auc(Hr, yr))
    m.randomize_weights(0)
    Hz, yz = spec.cal_acts(m, cal)
    info["check0b"]["random_twofold_auc"] = twofold_auc(Hz, yz)
    log("Check 0b", info["check0b"])
    json.dump(info, open(PRE + "_cal.json", "w"), indent=1, default=float)
    log("DONE")
    sys.exit(0)

# ==================================================================== timing
if STAGE == "timing":
    its = sorted(ITEMS_D6()["pilot"][::2], key=lambda x: (x["level"], len(x["problem"])))
    rows = [dict(ids=spec.prompt_ids(m, it)[0], key=(it["id"], s)) for it in its for s in range(NS)]
    sl = n_slots(max(len(r["ids"]) for r in rows))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = stream_generate(m, rows, CAP, slots=sl, log=log)
    dt = time.time() - t0
    ng = sum(len(x["toks"]) for x in res)
    out = dict(model=m.name, cap=CAP, rows=len(rows), slots=sl, kv_bytes_per_token=kv_bytes_per_token(m), seconds=dt, tokens=ng,
               tok_per_s=ng / dt, peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
               T=[x["T"] for x in res], capped=[x["capped"] for x in res], level=[its[i // NS]["level"] for i in range(len(rows))])
    log(f"timing: {len(rows)} chains, {sl} slots, {dt:.0f}s, {ng} tokens, {ng / dt:.0f} tok/s, peak {out['peak_gb']:.1f} GB")
    json.dump(out, open(PRE + f"_timing_cap{CAP}.json", "w"), indent=1)
    log("DONE")
    sys.exit(0)

cal = torch.load(PRE + "_cal.pt", weights_only=False)
ELL, G, du = cal["ell"], cal["g"], unit(cal["d"])
if PHASE != "pilot" and STAGE in ("clean", "push", "pushE") and not os.environ.get("P6_IGNORE_GATE"):
    _c0 = json.load(open(PRE + "_cal.json", encoding="utf-8"))["check0a"]
    _g = json.load(open(os.path.join(HERE, "stage2_settings_d6.json"), encoding="utf-8"))["check0a"]["d6"]
    if not gate_d6(_c0, _g):  # Stage 1 §5: not reproduced -> leaves the confirmatory analysis in this model
        log(f"Check 0a gate failed (ratios {_c0['ratio_minus']:.3f}, {_c0['ratio_plus']:.3f}); {STAGE} not run. "
            "Set P6_IGNORE_GATE=1 to run it as exploratory.")
        log("DONE")
        sys.exit(0)
byid = {x["id"]: x for x in S["A"] + S["oracle"]}


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8")}


# ================================================================== selftest
if STAGE == "selftest":
    its = ITEMS_D6()["pilot"][::8][:5]
    cap = 384
    # (0) decode attention: one decoding step with SDPA and with p6_decode_attention on the same prompts
    ids4 = [spec.prompt_ids(m, it)[0] for it in its[:4]]
    Lp = max(len(r) for r in ids4)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    x = torch.tensor([[pad] * (Lp - len(r)) + r for r in ids4], device=m.dev)
    am = torch.tensor([[0] * (Lp - len(r)) + [1] * len(r) for r in ids4], device=m.dev)
    pos = (am.cumsum(1) - 1).clamp_min(0)
    lg = {}
    with torch.no_grad():
        for name in ("sdpa", "p6decode"):
            (use_sdpa(m) if name == "sdpa" else install_fast_decode(m))
            c = new_cache()
            o = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=c, use_cache=True, logits_to_keep=1)
            nx = o.logits[:, -1].argmax(-1)[:, None]
            o2 = m.hf(input_ids=nx, attention_mask=torch.cat([am, torch.ones_like(am[:, :1])], 1), position_ids=pos[:, -1:] + 1,
                      past_key_values=c, use_cache=True, logits_to_keep=1)
            lg[name] = o2.logits[:, -1].float()
    install_fast_decode(m)
    dmax = float((lg["sdpa"] - lg["p6decode"]).abs().max())
    same = bool((lg["sdpa"].argmax(-1) == lg["p6decode"].argmax(-1)).all())
    log(f"decode attention vs SDPA: max |logit diff| {dmax:.4f} (logit scale {float(lg['sdpa'].abs().max()):.1f}), same argmax {same}")
    rows = []
    for it in its:
        ids, rp = spec.prompt_ids(m, it)
        for s in range(2):
            rows.append(dict(ids=ids, key=(it["id"], s)))
            rows.append(dict(ids=ids, key=(it["id"], s), add={ELL: vec(G * du)}, start="last"))
    rows.append(dict(ids=spec.prompt_ids(m, its[0])[0], key=(its[0]["id"], 0), add=authors_add(cal["R_all"], 0.2), start="all"))
    os.environ["P6_TEST_OOM_AT"] = "150"
    a = stream_generate(m, rows, cap, slots=5, log=log, admit_frac=0.4)
    os.environ["P6_TEST_OOM_AT"] = "0"
    b = []
    for r in rows:  # one lockstep batch per row
        add = {l: v[None] for l, v in r["add"].items()} if r.get("add") else None
        st = (len(r["ids"]) - 1) if r.get("start", "last") == "last" else 0
        b += sample_generate(m, [r["ids"]], [r["key"]], add=add, start=st, cap=cap)
    agree = []
    for x, y_, r in zip(a, b, rows):
        n = 0
        while n < min(len(x["toks"]), len(y_["toks"])) and x["toks"][n] == y_["toks"][n]:
            n += 1
        agree.append(dict(key=r["key"], pushed=bool(r.get("add")), T_stream=x["T"], T_lock=y_["T"], same_prefix=n, len_stream=len(x["toks"]), len_lock=len(y_["toks"])))
        log(agree[-1])
    clean_vs_push = [(agree[i]["T_stream"], agree[i + 1]["T_stream"]) for i in range(0, 20, 2)]
    log("clean vs one-gap push (stream):", clean_vs_push)
    log("text of row 0:", m.tok.decode(a[0]["toks"][:120]))
    json.dump(dict(agree=agree, attn_max_logit_diff=dmax, attn_same_argmax=same), open(PRE + "_selftest.json", "w"), indent=1)
    log("DONE")
    sys.exit(0)

# ================================================================== lamsweep
if STAGE == "lamsweep":
    lams = sorted({s * float(v) for v in EXTRA.split(",") for s in (1, -1)} | {0.0})
    its = S["calib0a"]
    rows, lab = steer_rows(its, lams, cal["R_all"])
    t0 = time.time()
    res = stream_generate(m, rows, CAP, slots=n_slots(max(len(r["ids"]) for r in rows)), log=log)
    out = dict(cap=CAP, seconds=time.time() - t0, lambdas={})
    for lm in lams:
        rr = [(i2, r) for r, (i2, l2) in zip(res, lab) if l2 == lm]
        out["lambdas"][f"{lm:+g}"] = dict(T_mean=float(np.mean([r["T"] for _, r in rr])), capped=float(np.mean([r["capped"] for _, r in rr])),
                                         eos_no_end=float(np.mean([r["eos_no_think_end"] for _, r in rr])),
                                         per_item={it["id"]: [r["T"] for i2, r in rr if i2 == it["id"]] for it in its},
                                         text0=[m.tok.decode(r["toks"][:200]) for _, r in rr[:12:4]])
        log(f"lambda {lm:+g}: T mean {out['lambdas'][f'{lm:+g}']['T_mean']:.0f}, capped {out['lambdas'][f'{lm:+g}']['capped']:.2f}")
    json.dump(out, open(PRE + "_lamsweep.json", "w"), indent=1)
    log("DONE")
    sys.exit(0)

# ===================================================================== clean
if STAGE == "clean":
    path = PRE + "_clean.jsonl"
    done = done_ids(path)
    items = [x for x in {x["id"]: x for x in S["A"] + S["oracle"]}.values() if x["id"] not in done]
    items.sort(key=lambda x: (x["level"], len(x["problem"])))
    log(f"clean: {len(items)} items to run, {NS} samples each")
    hpath = PRE + "_clean_h.npz"
    Hs = dict(np.load(hpath)) if os.path.exists(hpath) else {}
    Pm = np.array([du, cal["Nlen"], cal["Npc1"]] + list(cal["perm"]))
    am = CFG["answer_max_clean"]
    pids, pr = {}, {}
    for it in items:  # projections first (one prompt pass each)
        ids = spec.prompt_ids(m, it)[0]
        h = m.acts(ids, [ELL], [len(ids) - 1])[ELL][0].astype(np.float64)
        pids[it["id"]], pr[it["id"]] = ids, Pm @ h
        Hs[it["id"]] = h.astype(np.float32)
    np.savez(hpath, **Hs)
    rows = [dict(ids=pids[it["id"]], key=(it["id"], s)) for it in items for s in range(NS)]
    got = collections.defaultdict(dict)
    f = open(path, "a", encoding="utf-8")
    t0, ntok, nd = time.time(), [0], [0]

    def on_done(i, r):
        it = items[i // NS]
        got[it["id"]][i % NS] = r
        ntok[0] += len(r["toks"])
        if len(got[it["id"]]) < NS:
            return
        rr = [got[it["id"]][s] for s in range(NS)]
        ans = [norm_ans(answer_of(x["toks"])) for x in rr]
        gold = norm_ans(it.get("answer"))
        p = pr[it["id"]]
        rec = dict(id=it["id"], level=it["level"], subject=it["subject"], n_prompt=len(pids[it["id"]]),
                   n_problem=len(m.tok(it["problem"], add_special_tokens=False)["input_ids"]),
                   P=float(p[0]), P_Nlen=float(p[1]), P_Npc1=float(p[2]), P_perm=p[3:].tolist(),
                   T_s=[x["T"] for x in rr], capped_s=[x["capped"] for x in rr], eos_no_end_s=[x["eos_no_think_end"] for x in rr],
                   T=float(np.mean([x["T"] for x in rr])), n_capped=int(sum(x["capped"] for x in rr)),
                   correct_s=[int(a is not None and a == gold) for a in ans], ans_s=[a[:60] if a else None for a in ans],
                   n_gen_s=[len(x["toks"]) for x in rr])
        if PHASE == "pilot":
            rec["text0"] = m.tok.decode(rr[0]["toks"][:1500], skip_special_tokens=False)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        del got[it["id"]]
        nd[0] += 1
        if nd[0] % 10 == 0:
            dt = time.time() - t0
            log(f"clean {nd[0]}/{len(items)} items, {ntok[0] / dt:.0f} tok/s; last T={rec['T']:.0f} (L{it['level']}), capped {rec['n_capped']}")

    sl = n_slots(max(len(r["ids"]) for r in rows), am)
    log(f"{len(rows)} chains in {sl} slots")
    stream_generate(m, rows, CAP, answer_max=am, slots=sl, on_done=on_done, log=log)
    f.close()
    json.dump(dict(seconds=time.time() - t0, tokens=ntok[0], slots=sl), open(PRE + "_clean_timing.json", "w"))
    log(f"DONE ({time.time() - t0:.0f}s, {ntok[0]} tokens)")
    sys.exit(0)


# ====================================================================== push
def directions():
    clean = {json.loads(l)["id"]: json.loads(l) for l in open(PRE + "_clean.jsonl", encoding="utf-8")}
    Hs = dict(np.load(PRE + "_clean_h.npz"))
    orc = [x for x in S["oracle"] if x["id"] in clean]
    To = np.array([clean[x["id"]]["T"] for x in orc], dtype=np.float64)
    Ho = np.array([Hs[x["id"]] for x in orc], dtype=np.float64)
    o = oracle(Ho, To)
    log(f"oracle: {len(orc)} items, T mean {To.mean():.0f}; cos(o,d)={o @ du:+.3f}")
    gauss = gaussian(CFG["R"], m.d)
    np.savez(PRE + "_dirs.npz", d=du, Nlen=cal["Nlen"], Npc1=cal["Npc1"], o=o, perm=cal["perm"], gauss=gauss)
    return clean, o, gauss


def push_rows(ctrl, gauss, lam):
    """(meta, {block: vector}, start) for one Check D item; meta['s'] is the sample index."""
    rows = [(dict(kind="clean", s=s), None, "last") for s in range(NS)]
    for sg in (1, -1):
        for k in CFG["k_d"]:  # k = 1 is the primary push; the others are exploratory (Stage 1 §5)
            ss = range(NS) if k == 1 else range(CFG.get("explore_samples", NS))
            rows += [(dict(kind="d", k=sg * k, s=s), {ELL: sg * k * G * du}, "last") for s in ss]
        for name, v in ctrl.items():
            for k in CFG["k_ctrl"]:
                rows += [(dict(kind=name, k=sg * k, s=s), {ELL: sg * k * G * v}, "last") for s in range(NS)]
    for j in range(CFG["n_perm_push"]):
        rows += [(dict(kind="perm", j=j, c=c, s=j % NS), {ELL: c * G * cal["perm"][j]}, "last") for c in CFG["c"]]
    for j in range(CFG["R"]):
        rows += [(dict(kind="rand", j=j, c=c, s=j % NS), {ELL: c * G * gauss[j]}, "last") for c in CFG["c"]]
    for sg in (1, -1):  # the authors' every-layer steering, from position 0 (secondary)
        rows += [(dict(kind="d_auth", k=sg, lam=sg * lam, s=s), {j: sg * lam * cal["R_all"][j] for j in range(m.L - 1)}, "all")
                 for s in range(NS)]
    return rows


def kl_unique(ids, rp, rows, cont):
    """KL(clean || push) for each distinct push (it does not depend on the sample), teacher-forced on
    `cont`, grouped by push start; returns {push key: (kl_all, kl_first)}."""
    out = {}
    for mode in ("last", "all"):
        uniq = {}
        for meta, a, st in rows:
            if a is None or st != mode:
                continue
            uniq.setdefault(tuple((k, v) for k, v in meta.items() if k != "s"), a)
        if not uniq:
            continue
        keys = list(uniq)
        layers = sorted({l for a in uniq.values() for l in a})
        addk = {l: torch.zeros(len(keys) + 1, m.d) for l in layers}
        for b, k in enumerate(keys, 1):
            for l, v in uniq[k].items():
                addk[l][b] = vec(v)
        K = tf_kl(m, ids, rp if mode == "last" else 0, rp, addk, cont)
        out.update({k: (K["kl_all"][b], K["kl_first"][b]) for b, k in enumerate(keys, 1)})
    return out


if STAGE == "push":
    clean, o, gauss = directions()
    ctrl = {"Nlen": cal["Nlen"], "Npc1": cal["Npc1"], "o": o}
    lam = spec.authors_lambda(m)
    gpath, path = PRE + "_push_gen.jsonl", PRE + "_push.jsonl"
    D = S["D"][:CFG["D_items"]]
    template = push_rows(ctrl, gauss, lam)
    todo = [i for i in D if i not in done_ids(gpath)]
    log(f"push: {len(todo)} of {len(D)} items to generate, {len(template)} chains each; grid c={CFG['c']} R={CFG['R']} "
        f"perm={CFG['n_perm_push']}; authors' lambda {lam}")
    tadd = [{l: vec(v) for l, v in a.items()} if a else None for _, a, _ in template]  # shared by every item
    rows, owner = [], []
    for iid in todo:
        ids = spec.prompt_ids(m, byid[iid])[0]
        for (meta, _, st), a in zip(template, tadd):
            rows.append(dict(ids=ids, key=(iid, meta["s"]), add=a, start=st))
            owner.append((iid, meta))
    left = collections.Counter(i for i, _ in owner)
    got = collections.defaultdict(list)
    f = open(gpath, "a", encoding="utf-8")
    t0, ntok, nd = time.time(), [0], [0]

    def on_done(i, r):
        iid, meta = owner[i]
        got[iid].append((meta, r))
        ntok[0] += len(r["toks"])
        left[iid] -= 1
        if left[iid]:
            return
        Tc = {meta["s"]: r["T"] for meta, r in got[iid] if meta["kind"] == "clean"}
        c0 = [r for meta, r in got[iid] if meta["kind"] == "clean" and meta["s"] == 0][0]
        groups = {}
        for meta, r in got[iid]:
            key = tuple((k, v) for k, v in meta.items() if k != "s")
            g_ = groups.setdefault(key, dict({k: v for k, v in meta.items() if k != "s"}, s=[], T_s=[], capped_s=[]))
            g_["s"].append(meta["s"])
            g_["T_s"].append(r["T"])
            g_["capped_s"].append(r["capped"])
        recs = []
        for g_ in groups.values():
            g_["T"] = float(np.mean(g_["T_s"]))
            g_["dT"] = float(np.mean([t - Tc[s] for t, s in zip(g_["T_s"], g_["s"])]))  # against the same samples' clean chains
            recs.append(g_)
        f.write(json.dumps(dict(id=iid, level=byid[iid]["level"], T_clean=float(np.mean(list(Tc.values()))), T_clean_s=[Tc[s] for s in range(NS)],
                                cont=c0["toks"][:c0["T"] + 1], gen_tokens=int(sum(len(r["toks"]) for _, r in got[iid])),
                                t_done=time.time() - t0, rows=recs)) + "\n")
        f.flush()
        del got[iid]
        nd[0] += 1
        d1 = [x for x in recs if x["kind"] == "d" and abs(x["k"]) == 1] + [x for x in recs if x["kind"] == "d_auth"]
        log(f"push gen {nd[0]}/{len(todo)} {iid} L{byid[iid]['level']} ({ntok[0] / (time.time() - t0):.0f} tok/s): T clean {np.mean(list(Tc.values())):.0f}; "
            + " ".join(f"{x['kind']}{x['k']:+g} dT={x['dT']:+.0f}" for x in d1))

    if rows:
        sl = n_slots(max(len(r["ids"]) for r in rows))
        log(f"{len(rows)} chains in {sl} slots")
        stream_generate(m, rows, CAP, slots=sl, on_done=on_done, log=log)
    f.close()
    if todo:
        json.dump(dict(seconds=time.time() - t0, tokens=ntok[0], items=len(todo)), open(PRE + "_push_gen_timing.json", "w"))
    # KL of every distinct push on the clean sample 0's chain
    done = done_ids(path)
    gen = [json.loads(l) for l in open(gpath, encoding="utf-8")]
    with open(path, "a", encoding="utf-8") as fo:
        for rec in gen:
            if rec["id"] in done or rec["id"] not in D:
                continue
            t1 = time.time()
            ids, rp = spec.prompt_ids(m, byid[rec["id"]])
            KL = kl_unique(ids, rp, template, rec["cont"])
            for r in rec["rows"]:
                key = tuple((k, v) for k, v in r.items() if k not in ("s", "T_s", "capped_s", "T", "dT"))
                r["kl"], r["kl_first"] = KL.get(key, (0.0, 0.0))
            rec["n_cont"] = len(rec.pop("cont"))
            rec["kl_seconds"] = time.time() - t1
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            d1 = [x for x in rec["rows"] if x["kind"] in ("d", "d_auth") and abs(x["k"]) == 1]
            log(f"push KL {rec['id']} ({rec['kl_seconds']:.0f}s, {rec['n_cont']} tokens): " + " ".join(f"{x['kind']}{x['k']:+g} kl={x['kl']:.1f}" for x in d1))
    log("DONE")
    sys.exit(0)

# ===================================================================== pushE
if STAGE == "pushE":
    if EXTRA == "auto":  # the signs whose Check D verdict is "pass" in the analysis summary (Stage 1 §5: E only for those)
        sm = json.load(open(PRE + "_summary.json", encoding="utf-8"))
        signs = [1 if k.endswith("+") else -1 for k, v in sm["checkD"].items() if v.get("verdict") == "pass"]
        if not signs:
            log("no push sign passes Check D; Check E not run")
            log("DONE")
            sys.exit(0)
    else:
        signs = [s for s, ch in ((1, "+"), (-1, "-")) if ch in (EXTRA or "+-")]
    path = PRE + "_pushE.jsonl"
    done = done_ids(path)
    E = S["E"][:CFG["E_items"]]
    todo = [i for i in E if i not in done]
    later = list(range(ELL + 1, m.L))
    if m.L > 32:
        later = later[::2]
    dev_du = torch.tensor(du, dtype=torch.float32, device=m.dev)
    log(f"pushE: {len(todo)} of {len(E)} items; signs {signs}; E1 blocks {later}; E2 c={CFG['e2_c']}")
    with open(path, "a", encoding="utf-8") as f:
        for n, iid in enumerate(todo):
            it = byid[iid]
            ids, rp = spec.prompt_ids(m, it)
            t0 = time.time()
            body = []
            for sg in signs:
                body += [(dict(kind="E1", k=sg, lp=lp), {ELL: sg * G * du}, lp) for lp in later]
                body += [(dict(kind="E2", k=sg, c=c), {m.L - 1: sg * c * G * du}, -1) for c in CFG["e2_c"]]
            layers = sorted({l for _, a, _ in body for l in a} | {ELL})
            # lockstep batches: no compaction with re-imposition, every row stays to the end
            per = max(2, int(os.environ["SLOTS"]) if os.environ.get("SLOTS") else rows_for(m, len(ids) + CAP))
            out, Tc, cont = {}, {}, None
            for s in range(NS):
                for c0 in range(0, len(body), per - 1):
                    chunk = body[c0:c0 + per - 1]
                    B = len(chunk) + 1
                    add = {l: torch.zeros(B, m.d) for l in layers}
                    rl = torch.full((B,), -1, dtype=torch.long)
                    for b, (meta, a, lp) in enumerate(chunk, 1):
                        for l, v in a.items():
                            add[l][b] = vec(v)
                        rl[b] = lp
                    reimp = dict(dir=dev_du, layer=rl.to(m.dev))
                    res = sample_generate(m, [ids] * B, [(iid, s)] * B, add=add, start=rp, cap=CAP, reimp=reimp)
                    Tc.setdefault(s, res[0]["T"])
                    if s == 0 and cont is None:
                        cont = res[0]["toks"][:res[0]["T"] + 1]
                    for (meta, _, _), r in zip(chunk, res[1:]):
                        key = tuple(sorted(meta.items()))
                        out.setdefault(key, dict(meta, s=[], T_s=[], capped_s=[], clean_T_s=[]))
                        out[key]["s"].append(s)
                        out[key]["T_s"].append(r["T"])
                        out[key]["capped_s"].append(r["capped"])
                        out[key]["clean_T_s"].append(res[0]["T"])
            e2 = [(meta, a) for meta, a, _ in body if meta["kind"] == "E2"]
            addk = {m.L - 1: torch.cat([torch.zeros(1, m.d)] + [vec(a[m.L - 1])[None] for _, a in e2], 0)}
            K = tf_kl(m, ids, rp, rp, addk, cont)
            for b, (meta, _) in enumerate(e2, 1):
                out[tuple(sorted(meta.items()))]["kl"] = K["kl_all"][b]
            recs = []
            for r in out.values():
                r["T"] = float(np.mean(r["T_s"]))
                r["dT"] = float(np.mean([t - c for t, c in zip(r["T_s"], r["clean_T_s"])]))
                recs.append(r)
            f.write(json.dumps(dict(id=iid, rows=recs, T_clean_s=[Tc[s] for s in range(NS)], seconds=time.time() - t0)) + "\n")
            f.flush()
            log(f"pushE {n + 1}/{len(todo)} {iid} {time.time() - t0:.0f}s")
    log("DONE")
