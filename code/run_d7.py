"""Paper 6, D7 stages that run.py does not have. python run_d7.py <label|lens> <model key> [pilot|main]

label  per-model known/unknown labels (spec_d7 docstring): every attribute question of every entity
       in the D7 pools in the authors' chat format, greedy plus 10 samples at temperature 1 (no top-k,
       top-p or repetition penalty, as the authors' vLLM call), 32 new tokens. Resumable.
       -> res/d7/<key>_<phase>_labels_raw.jsonl and _labels.json
lens   secondary tracking target (§3): logit-lens rank of the correct attribute's first token at the
       last entity token of each Check A / oracle / Check D question, best over blocks ell*..L-1
       (needs the cal stage). -> res/d7/<key>_<phase>_lens.json
"""

import json
import os
import sys
import time

import numpy as np
import torch
import transformers

from p6lib import Model
import spec_d7 as S

STAGE, KEY = sys.argv[1], sys.argv[2]
PHASE = sys.argv[3] if len(sys.argv) > 3 else "main"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res", "d7")
os.makedirs(RES, exist_ok=True)
PRE = os.path.join(RES, f"{KEY}_{PHASE}")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


m = Model(KEY, thinking=False)
log(f"{m.name} L={m.L} stage={STAGE} phase={PHASE}")


@torch.no_grad()
def sample_batch(batch_ids, gen):
    """Row 0 of each prompt greedy, rows 1..10 sampled at temperature 1; left-padded batch."""
    R = 1 + S.N_SAMPLES
    rows = [r for r in batch_ids for _ in range(R)]
    n = max(len(r) for r in rows)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    x = torch.tensor([[pad] * (n - len(r)) + r for r in rows], device=m.dev)
    am = torch.tensor([[0] * (n - len(r)) + [1] * len(r) for r in rows], device=m.dev)
    pos = (am.cumsum(1) - 1).clamp_min(0)
    greedy = torch.tensor([j % R == 0 for j in range(len(rows))], device=m.dev)
    done = torch.zeros(len(rows), dtype=torch.bool, device=m.dev)
    eos_t = torch.tensor(sorted(m.eos), device=m.dev)
    cache = transformers.DynamicCache()
    out = []
    for _ in range(S.N_LABEL_TOK):
        lg = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True,
                  logits_to_keep=1).logits[:, -1].float()
        smp = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen)[:, 0]
        nxt = torch.where(greedy, lg.argmax(-1), smp)
        nxt = torch.where(done, eos_t[0], nxt)
        out.append(nxt)
        done |= torch.isin(nxt, eos_t)
        if bool(done.all()):
            break
        x = nxt[:, None]
        am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
        pos = pos[:, -1:] + 1
    del cache
    toks = torch.stack(out, 1).cpu()
    texts = [m.decode(t) for t in toks]
    torch.cuda.empty_cache()
    return [texts[i * R:(i + 1) * R] for i in range(len(batch_ids))]


if STAGE == "label":
    I = S.ITEMS_D7()
    ents = {}
    for pool in S.label_pools(PHASE):
        for x in I[pool]:
            ents[S.ekey(x)] = x
    queries = []
    for k, x in ents.items():
        for pt in S.PROMPT_TYPES[x["type"]]:
            attr = pt[len(x["type"]) + 1:]
            if attr not in x["attrs"]:
                continue
            q, prefill, _ = S.QUERIES[pt]
            queries.append((k, pt, x["attrs"][attr], m.chat(q.format(entity=x["entity"])) + prefill))
    path = PRE + "_labels_raw.jsonl"
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["e"], r["pt"]))
    todo = [q for q in queries if (q[0], q[1]) not in done]
    ids = [m.ids(q[3]) for q in todo]
    order = sorted(range(len(todo)), key=lambda i: len(ids[i]))
    QB = int(os.environ.get("LABEL_Q", "48"))
    gen = torch.Generator(device=m.dev).manual_seed(1234 + len(done))
    log(f"label: {len(ents)} entities, {len(queries)} questions, {len(todo)} to run, {QB} per batch")
    with open(path, "a", encoding="utf-8") as f:
        for b0 in range(0, len(order), QB):
            bi = order[b0:b0 + QB]
            for i, tx in zip(bi, sample_batch([ids[i] for i in bi], gen)):
                k, pt, ans, _ = todo[i]
                cor = [S.correct(pt, t, ans) for t in tx]
                f.write(json.dumps(dict(e=k, pt=pt, greedy=tx[0][:200], g_cor=cor[0], g_ref=S.is_generation_refusal(tx[0]),
                                        n_cor=int(sum(cor[1:])), s_cor=cor[1:], samples=[t[:100] for t in tx[1:]]),
                                   ensure_ascii=False) + "\n")
            f.flush()
            if (b0 // QB) % 25 == 0:
                log(f"label {b0 + len(bi)}/{len(order)}; last: {pt} greedy={tx[0][:60]!r} n_cor={sum(cor[1:])}")
    per = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        per.setdefault(r["e"], {})[r["pt"]] = r
    labels = {}
    for k, qs in per.items():
        nc = sum(int(r["n_cor"] >= S.MIN_CORRECT_SAMPLES) for r in qs.values())
        labels[k] = dict(label="known" if nc >= 2 else ("unknown" if nc == 0 else "other"), n_correct_q=nc, n_q=len(qs),
                         greedy_correct=int(sum(r["g_cor"] for r in qs.values())), greedy_refusal=int(sum(r["g_ref"] for r in qs.values())))
    json.dump(labels, open(PRE + "_labels.json", "w", encoding="utf-8"), indent=0, ensure_ascii=False)
    for pool in S.label_pools(PHASE):
        c = {t: {} for t in S.TYPES}
        for x in I[pool]:
            lb = labels.get(S.ekey(x), {}).get("label")
            c[x["type"]][lb] = c[x["type"]].get(lb, 0) + 1
        log(f"labels {pool}: {c}")
    log("DONE")
    sys.exit(0)


def candidates(it):
    """First tokens (with and without a leading space) of every accepted answer string."""
    vals = [v.strip() for v in it["answer"].split(",") if v.strip()]
    low = it["answer"].lower()
    if low == "united states of america":
        vals += ["United States", "USA", "US"]
    elif low == "people's republic of china":
        vals += ["China"]
    c = set()
    for v in vals:
        for s in (v, " " + v):
            t = m.tok.encode(s, add_special_tokens=False)
            if t:
                c.add(t[0])
    return sorted(c)


if STAGE == "lens":
    cal = torch.load(PRE + "_cal.pt", weights_only=False)
    ell = cal["ell"]
    spec = S.D7()
    sets = spec.items(PHASE)
    items = list({x["id"]: x for x in sets["A"] + sets["oracle"]}.values())
    layers = list(range(ell, m.L))
    out = {}
    with torch.no_grad():
        for n, it in enumerate(items):
            ids, st = spec.prompt_ids(m, it, "direct")
            a = m.acts(ids, layers, [st])
            h = torch.tensor(np.stack([a[l][0] for l in layers]), device=m.dev, dtype=torch.bfloat16)
            lg = m.hf.lm_head(m.norm(h[:, None, :]))[:, 0].float()  # [layers, V]
            cand = torch.tensor(candidates(it), device=m.dev)
            tgt = lg[:, cand]  # [layers, C]
            rank = (lg[:, None, :] > tgt[:, :, None]).sum(-1) + 1  # [layers, C]
            best_per_layer = rank.min(1).values
            j = int(best_per_layer.argmin())
            out[it["id"]] = dict(min_rank=int(best_per_layer[j]), layer=layers[j], rank_last=int(best_per_layer[-1]), n_cand=len(cand))
            if n % 100 == 0:
                log(f"lens {n}/{len(items)} {it['id']} {out[it['id']]}")
    json.dump(out, open(PRE + "_lens.json", "w", encoding="utf-8"), indent=0)
    log("DONE")
