"""Paper 6 runs. python run.py <d1|d2|d3> <model key> <cal|clean|push> [pilot|main]

cal    rebuild the direction with its authors' recipe; build N_len, N_pc1, N_perm; Check 0a;
       Check B projections and leakage baselines; Check 0b (weights are re-initialised last).
clean  projections and clean outcomes on the Check A and oracle items (single-row batches).
push   Check C/D/E rows on the Check D items: one batch per item holds every condition on the
       same prompt (row 0 clean), as in paper 4. Needs the oracle targets (and judge labels for D1).
Resumable per item.
"""

import json
import os
import sys
import time

import numpy as np
import torch
import transformers

from p6lib import (Model, auc, build_conditions, gaussian, n_len, n_pc1, n_perm, oracle, rows_to_ops,
                   twofold_auc, unit)
from specs import SPECS
from spec_d5 import D5, kw

SPECS["d5"] = D5
from spec_d7 import D7; SPECS["d7"] = D7  # noqa: E402,E702  (Stage 2b D7)

DIR, KEY, STAGE = sys.argv[1], sys.argv[2], sys.argv[3]
PHASE = sys.argv[4] if len(sys.argv) > 4 else "main"
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, f"config_{PHASE}.json"), encoding="utf-8"))
if os.environ.get("MAX_ROWS"):  # memory cap per batch (OLMo-2 has full multi-head attention)
    CFG["max_rows"] = int(os.environ["MAX_ROWS"])
RES = os.path.join(HERE, "res", DIR)
os.makedirs(RES, exist_ok=True)
PRE = os.path.join(RES, f"{KEY}_{PHASE}")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = SPECS[DIR]()
m = Model(KEY, thinking=getattr(spec, "thinking", False), gpu_layers=int(os.environ["GPU_LAYERS"]) if os.environ.get("GPU_LAYERS") else None)
log(f"{m.name} L={m.L} d={m.d} stage={STAGE} phase={PHASE}")
n_new = lambda mode: spec.n_new_mode[mode] if hasattr(spec, "n_new_mode") else spec.n_new


@torch.no_grad()
def gen_padded(m, batch_ids, ops_fn, n, bs=32, tokens=False):
    """Greedy generation for different prompts (left-padded); ops apply at every position."""
    texts = []
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    for i in range(0, len(batch_ids), bs):
        b = batch_ids[i:i + bs]
        L = max(len(r) for r in b)
        x = torch.tensor([[pad] * (L - len(r)) + r for r in b], device=m.dev)
        am = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in b], device=m.dev)
        pos = (am.cumsum(1) - 1).clamp_min(0)
        cache = transformers.DynamicCache()
        out = []
        m.ops = ops_fn(len(b)) if ops_fn else None
        if m.ops is not None:
            m.ops["add"] = {l: v.to(m.dev, torch.bfloat16) for l, v in m.ops["add"].items()}
        try:
            for _ in range(n):
                lg = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True).logits[:, -1]
                nxt = lg.argmax(-1)
                out.append(nxt)
                x = nxt[:, None]
                am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
                pos = pos[:, -1:] + 1
        finally:
            m.ops = None
        toks = torch.stack(out, 1).cpu()
        texts += [m.trunc(t) if tokens else m.decode(t) for t in toks]
    return texts


def text_only_auc(sets, labels, groups):
    """Leakage baseline (Check B): bag-of-tokens logistic regression, 5-fold grouped by pair."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    X = CountVectorizer(token_pattern=r"\S+").fit_transform([" ".join(map(str, s)) for s in sets])
    p = cross_val_predict(LogisticRegression(max_iter=2000), X, labels, groups=groups, cv=GroupKFold(5), method="predict_proba")[:, 1]
    return auc(p, labels)


def load_cal():
    cal = torch.load(PRE + "_cal.pt", weights_only=False)
    if DIR == "d1":
        spec._pos = cal["pos"]
    return cal


# ======================================================================= cal
if STAGE == "cal":
    cal = spec.calibrate(m, log)
    if DIR == "d1":
        spec._pos = cal["pos"]
    du = unit(cal["d"])
    H, y = cal["H"], np.asarray(cal["y"])
    g = float((H[y == 1] @ du).mean() - (H[y == 0] @ du).mean())
    Nl, Np = n_len(H, y, cal["lengths"]), None
    Np = n_pc1(H, y, du)
    perms = n_perm(H, y, CFG["n_perm"], seed=6)
    info = dict(ell=cal["ell"], g=g, L=m.L, model=m.name, d_norm=float(np.linalg.norm(cal["d"])),
                cal_auc=dict(d=auc(H @ du, y), Nlen=auc(H @ Nl, y), Npc1=auc(H @ Np, y)),
                cos=dict(d_Nlen=float(du @ Nl), d_Npc1=float(du @ Np), Nlen_Npc1=float(Nl @ Np),
                         d_perm=[float(du @ p) for p in perms]),
                resid_norm_median=float(np.median(np.linalg.norm(H, axis=1))))
    log(f"ell={cal['ell']} g={g:.2f} |h|~{info['resid_norm_median']:.0f} cal AUC d={info['cal_auc']['d']:.3f} "
        f"Nlen={info['cal_auc']['Nlen']:.3f} Npc1={info['cal_auc']['Npc1']:.3f} cos(d,Nlen)={info['cos']['d_Nlen']:+.3f} "
        f"cos(d,Npc1)={info['cos']['d_Npc1']:+.3f}")
    info["check0a"] = spec.check0a(m, cal, gen_padded, log)
    # Check B: matched label set (evaluation only)
    Bs = spec.B_set(m, spec.items(PHASE)["B"])
    if Bs:
        hb = np.array([m.acts(b["ids"], [cal["ell"]], [b["start"]])[cal["ell"]][0] for b in Bs], dtype=np.float64)
        yb = np.array([b["label"] for b in Bs])
        grp = [b["id"].split(":")[0] for b in Bs]
        info["checkB"] = dict(n_pairs=len(Bs) // 2, auc_d=auc(hb @ du, yb), auc_Nlen=auc(hb @ Nl, yb), auc_Npc1=auc(hb @ Np, yb),
                              text_only_auc=text_only_auc([m.tok(b["text"], add_special_tokens=False)["input_ids"] for b in Bs], yb, grp),
                              length_auc=auc([len(b["ids"]) for b in Bs], yb),
                              proj=dict(id=[b["id"] for b in Bs], label=yb.tolist(), P=(hb @ du).tolist()))
        log("Check B", {k: v for k, v in info["checkB"].items() if k != "proj"})
    save = dict(cal, Nlen=Nl, Npc1=Np, perm=perms, info=info, g=g)
    torch.save(save, PRE + "_cal.pt")
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

cal = load_cal()
ELL, G, du = cal["ell"], cal["g"], unit(cal["d"])
S = spec.items(PHASE)
byid = {x["id"]: x for x in S["A"] + S["oracle"]}


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8")}


# ============================================================ clean (D5)
if STAGE == "clean" and DIR == "d5":
    path = PRE + "_clean.jsonl"
    done = done_ids(path)
    items = [x for x in {x["id"]: x for x in S["A"] + S["oracle"]}.values() if x["id"] not in done]
    log(f"clean (chains of {spec.n_chain} tokens): {len(items)} items to run")
    hpath = PRE + "_clean_h.npz"
    Hs = dict(np.load(hpath)) if os.path.exists(hpath) else {}
    Pm = np.array([du, cal["Nlen"], cal["Npc1"]] + list(cal["perm"]))
    bs = int(os.environ.get("CHAIN_BS", "16"))
    with open(path, "a", encoding="utf-8") as f:
        for b0 in range(0, len(items), bs):
            batch = items[b0:b0 + bs]
            pids = [spec.prompt_ids(m, it, "gen")[0] for it in batch]
            gens = gen_padded(m, pids, None, spec.n_chain, bs=bs, tokens=True)
            for it, ids, toks in zip(batch, pids, gens):
                sents, n = spec.sentences(m, toks)
                full = ids + toks[:n]
                a = m.acts(full, [ELL], list(range(len(full))))[ELL]
                rec = dict(id=it["id"], set=it.get("set"), n_prompt=len(ids), n_tok=n, sentences=[],
                           all_sents=[x["text"][:300] for x in sents])
                prev_len = None
                for j, st in enumerate(sents):
                    s_abs = len(ids) + st["tok"]
                    nxt_tok = sents[j + 1]["tok"] if j + 1 < len(sents) else n
                    ok = st["complete"] and s_abs - 13 >= len(ids)
                    if ok:
                        h6 = a[s_abs - 13: s_abs - 7].astype(np.float64).mean(0)
                        pr = Pm @ h6
                        rec["sentences"].append(dict(j=j, pos=st["tok"], prev_len=prev_len, T=kw(st["text"]), P=float(pr[0]),
                                                     P_Nlen=float(pr[1]), P_Npc1=float(pr[2]), P_perm=pr[3:].tolist(), text=st["text"][:200]))
                        Hs[f"{it['id']}:{j}"] = h6.astype(np.float32)
                    prev_len = nxt_tok - st["tok"]
                f.write(json.dumps(rec) + "\n")
                f.flush()
            np.savez(hpath, **Hs)
            log(f"clean chains {b0 + len(batch)}/{len(items)}; last: {rec['n_tok']} tokens, {len(rec['sentences'])} sentences, "
                f"{sum(x['T'] for x in rec['sentences'])} backtracks")
    log("DONE")
    sys.exit(0)

# ===================================================================== clean
if STAGE == "clean":
    path = PRE + "_clean.jsonl"
    done = done_ids(path)
    items = list({x["id"]: x for x in S["A"] + S["oracle"]}.values())
    todo = [x for x in items if x["id"] not in done]
    log(f"clean: {len(todo)} of {len(items)} items")
    hpath = PRE + "_clean_h.npz"
    Hs = dict(np.load(hpath)) if os.path.exists(hpath) else {}
    Pm = np.array([du, cal["Nlen"], cal["Npc1"]] + list(cal["perm"]))
    with open(path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            rec = dict(id=it["id"], set=it.get("set"))
            for j, mode in enumerate(spec.modes):
                ids, start = spec.prompt_ids(m, it, mode)
                if j == 0:
                    h = m.acts(ids, [ELL], [start])[ELL][0].astype(np.float64)
                    pr = Pm @ h
                    rec.update(P=float(pr[0]), P_Nlen=float(pr[1]), P_Npc1=float(pr[2]), P_perm=pr[3:].tolist(),
                               n_prompt=len(ids))
                    Hs[it["id"]] = h.astype(np.float32)
                z = {ELL: torch.zeros(1, m.d)}
                r = m.generate(ids, start, z, n_new(mode), lens=(mode == "direct"))
                rec[mode] = spec.outcome(m, it, mode, r["tokens"][0], r["lp0"][0], r["top1"][0] if "top1" in r else None)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 25 == 0:
                log(f"clean {n}/{len(todo)} {it['id']} " + json.dumps({k: v for k, v in rec[spec.modes[0]].items() if k != "text"})[:150])
                np.savez(hpath, **Hs)
    np.savez(hpath, **Hs)
    log("DONE")
    sys.exit(0)


# ====================================================================== push
def oracle_T(rec, judged):
    if DIR == "d1":  # Stage 2a: the judge failed validation (kappa 0.52 < 0.7); phrase matching is primary
        return rec["gen"]["sub"]
    if DIR == "d2":
        return rec["gen"]["T"]
    return rec["direct"]["D"]


if STAGE == "push":
    clean = {json.loads(l)["id"]: json.loads(l) for l in open(PRE + "_clean.jsonl", encoding="utf-8")}
    Hs = dict(np.load(PRE + "_clean_h.npz"))
    judged = None
    jp = PRE + "_clean_judge.json"
    if DIR == "d5":  # oracle from the oracle chains' sentences: next sentence backtracks or not
        orc = [(x["id"], sj) for x in S["oracle"] if x["id"] in clean for sj in clean[x["id"]]["sentences"]]
        To = np.array([sj["T"] for _, sj in orc], dtype=np.float64)
        Ho = np.array([Hs[f"{i}:{sj['j']}"] for i, sj in orc], dtype=np.float64)
    else:
        orc = [x for x in S["oracle"] if x["id"] in clean and oracle_T(clean[x["id"]], judged) is not None]
        To = np.array([oracle_T(clean[x["id"]], judged) for x in orc], dtype=np.float64)
        Ho = np.array([Hs[x["id"]] for x in orc], dtype=np.float64)
    o = spec.oracle_dir(Ho, To, orc) if hasattr(spec, "oracle_dir") else oracle(Ho, To, binary=spec.binary_T or DIR == "d5")
    log(f"oracle: {len(orc)} items, T mean {To.mean():.3f}; cos(o,d)={o @ du:+.3f}")
    ctrl = {"Nlen": cal["Nlen"], "Npc1": cal["Npc1"], "o": o}
    gauss = gaussian(CFG["R"], m.d)
    if DIR == "d5":  # Ward et al.'s magnitude 8 on the raw vector, in gaps; random grid extended to bracket it
        from spec_d5 import AUTHORS_MAG
        CFG["k_d_extra"] = [round(AUTHORS_MAG * float(np.linalg.norm(cal["d"])) / G, 3)]
        CFG["c"] = sorted(set(CFG["c"]) | {6, 8, 12, 16, 24})  # Stage 2b: the pilot needed > 12 to bracket
        log(f"authors' strength = {CFG['k_d_extra'][0]} gaps; random grid c = {CFG['c']}")
    np.savez(PRE + "_dirs.npz", d=du, Nlen=cal["Nlen"], Npc1=cal["Npc1"], o=o, perm=cal["perm"], gauss=gauss)
    path = PRE + "_push.jsonl"
    done = done_ids(path)
    Eset = set(S["E"])
    todo = [i for i in S["D"] if i not in done]
    log(f"push: {len(todo)} of {len(S['D'])} items")
    kl_mode = "direct" if "direct" in spec.modes else spec.modes[0]
    dev_du = torch.tensor(du, dtype=torch.float32, device=m.dev)
    with open(path, "a", encoding="utf-8") as f:
        for n, iid in enumerate(todo):
            it = byid[iid]
            rows = build_conditions(du, G, spec.signs(it), ctrl, cal["perm"], gauss, CFG, m.L, ELL, e_items=iid in Eset)
            chunks, body = [], rows[1:]
            step = CFG["max_rows"] - 1
            for c0 in range(0, len(body), step):
                chunks.append([rows[0]] + body[c0:c0 + step])
            out_rows = [dict(r[0]) for r in rows]
            for ci, ch in enumerate(chunks):
                add, rl = rows_to_ops(ch, m.d)
                reimp = dict(dir=dev_du, layer=rl.to(m.dev)) if bool((rl >= 0).any()) else None
                tgt = [0] + list(range(1 + ci * step, 1 + ci * step + len(ch) - 1))
                for mode in spec.modes:
                    ids, start = spec.prompt_ids(m, it, mode)
                    r = m.generate(ids, start, add, n_new(mode), reimp=reimp, lens=(mode == "direct"))
                    for b, ti in enumerate(tgt):
                        if ci > 0 and b == 0:
                            continue
                        oc = spec.outcome(m, it, mode, r["tokens"][b], r["lp0"][b], r["top1"][b] if "top1" in r else None)
                        if DIR == "d2" and ti != 0:
                            oc.pop("text", None)
                        out_rows[ti][mode] = oc
                    if mode == kl_mode:
                        cont = m.trunc(r["tokens"][0])
                        K = m.tf_kl(ids, start, add, cont, reimp=reimp)
                        for b, ti in enumerate(tgt):
                            if ci > 0 and b == 0:
                                continue
                            out_rows[ti]["kl"] = K["kl_all"][b]
                            out_rows[ti]["kl_first"] = K["kl_first"][b]
            f.write(json.dumps(dict(id=iid, n_rows=len(rows), rows=out_rows)) + "\n")
            f.flush()
            if n % 10 == 0:
                c = out_rows[0]
                dr = [x for x in out_rows if x["kind"] == "d" and abs(x["k"]) == 1]
                log(f"push {n}/{len(todo)} {iid} rows={len(rows)} clean=" +
                    json.dumps({k: v for k, v in c[spec.modes[0]].items() if k != "text"})[:120] +
                    " | d(+-1): " + " ".join(f"kl={x['kl']:.2f}" for x in dr))
    log("DONE")
