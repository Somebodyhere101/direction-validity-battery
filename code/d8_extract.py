"""D8 persona-vector extraction and reproduction steps (Chen et al. 2025), run as separate GPU jobs so the
subject model and the judge never share the card.

python d8_extract.py gen     <model key> <phase>   sample pos/neg responses     -> res/d8/<key>_<phase>_extract.jsonl
python d8_extract.py judge   <model key> <phase>   score trait and coherence     -> ..._extract_judged.jsonl
python d8_extract.py steer0a <model key> <phase>   their eval questions, baseline vs +2.0 * vector at their layer
python d8_extract.py judge0a <model key> <phase>   score those                   -> ..._0a.json

Judge: Qwen3-8B (thinking off) with the authors' own sycophancy and coherence prompts, greedy, parsed to a
0-100 integer ('REFUSAL' -> missing), in place of their gpt-4.1-mini (Stage 2b).
"""

import json
import os
import re
import sys
import time

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STEP, KEY, PHASE = sys.argv[1], sys.argv[2], sys.argv[3]
RES = os.path.join(HERE, "res", "d8")
os.makedirs(RES, exist_ok=True)
PRE = os.path.join(RES, f"{KEY}_{PHASE}")
N_SAMPLES = 10 if PHASE == "main" else 3
N_NEW, TEMP, BS = 400, 1.0, int(os.environ.get("D8_BS", "48"))
COEF, N_EVAL = 2.0, 5
T0 = time.time()
EX = json.load(open(os.path.join(HERE, "data", "persona", "sycophantic_extract.json"), encoding="utf-8"))
EV = json.load(open(os.path.join(HERE, "data", "persona", "sycophantic_eval.json"), encoding="utf-8"))
COH = open(os.path.join(HERE, "data", "persona", "coherence_0_100.txt"), encoding="utf-8").read()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def a_or_an(w):
    return "an" if w[0].lower() in "aeiou" else "a"


@torch.no_grad()
def sample(m, prompts, ops_fn=None):
    """Temperature sampling, left-padded; ops_fn(B, L) gives the steering ops for a batch (or None)."""
    out = []
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    eos = torch.tensor(sorted(m.eos), device=m.dev)
    gen = torch.Generator(device=m.dev).manual_seed(1234)
    for b0 in range(0, len(prompts), BS):
        b = [m.ids(p) for p in prompts[b0:b0 + BS]]
        L = max(len(r) for r in b)
        x = torch.tensor([[pad] * (L - len(r)) + r for r in b], device=m.dev)
        am = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in b], device=m.dev)
        pos = (am.cumsum(1) - 1).clamp_min(0)
        cache = transformers.DynamicCache()
        ops = ops_fn(len(b), L) if ops_fn else None
        toks, done = [], torch.zeros(len(b), dtype=torch.bool, device=m.dev)
        p0 = 0
        try:
            for _ in range(N_NEW):
                if ops is not None:
                    ops["pos0"] = p0
                    m.ops = ops
                lg = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].float()
                m.ops = None
                p0 += x.shape[1]
                nxt = torch.multinomial(torch.softmax(lg / TEMP, -1), 1, generator=gen)[:, 0]
                nxt = torch.where(done, eos[0], nxt)
                toks.append(nxt)
                done |= torch.isin(nxt, eos)
                if bool(done.all()):
                    break
                x = nxt[:, None]
                am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
                pos = pos[:, -1:] + 1
        finally:
            m.ops = None
        del cache
        T = torch.stack(toks, 1).cpu()
        out += [m.decode(t) for t in T]
        torch.cuda.empty_cache()
        log(f"sampled {len(out)}/{len(prompts)}")
    return out


def load_subject():
    from p6lib import Model
    return Model(KEY, thinking=False)


def judge_scores(pairs):
    """pairs: list of (prompt_text) -> list of int scores or None."""
    name = "Qwen/Qwen3-8B"
    tok = transformers.AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    out = []
    for i in range(0, len(pairs), 16):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in pairs[i:i + 16]]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = hf.generate(**enc, max_new_tokens=6, do_sample=False)
        for row in g[:, enc["input_ids"].shape[1]:]:
            t = tok.decode(row, skip_special_tokens=True).strip()
            mm = re.match(r"(\d{1,3})", t)
            out.append(min(100, int(mm.group(1))) if mm else None)
        if (i // 16) % 50 == 0:
            log(f"judged {len(out)}/{len(pairs)}")
    return out


if STEP == "gen":
    m = load_subject()
    rows = []
    for pole, name in (("pos", "sycophantic"), ("neg", "helpful")):
        for k, ins in enumerate(EX["instruction"]):
            system = f"You are {a_or_an(name)} {name} assistant. {ins[pole]}"
            for q, question in enumerate(EX["questions"]):
                for s in range(N_SAMPLES):
                    rows.append(dict(pole=pole, k=k, q=q, s=s, system=system, question=question))
    texts = sample(m, [m.chat(r["question"], system=r["system"]) for r in rows])
    with open(PRE + "_extract.jsonl", "w", encoding="utf-8") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps(dict(r, answer=t), ensure_ascii=False) + "\n")
    log("DONE")

elif STEP == "judge":
    rows = [json.loads(l) for l in open(PRE + "_extract.jsonl", encoding="utf-8")]
    tr = judge_scores([EX["eval_prompt"].format(question=r["question"], answer=r["answer"]) for r in rows])
    co = judge_scores([COH.format(question=r["question"], answer=r["answer"]) for r in rows])
    with open(PRE + "_extract_judged.jsonl", "w", encoding="utf-8") as f:
        for r, a, b in zip(rows, tr, co):
            f.write(json.dumps(dict(r, trait=a, coherence=b), ensure_ascii=False) + "\n")
    log("pos trait mean", np.mean([a for r, a in zip(rows, tr) if r["pole"] == "pos" and a is not None]),
        "neg trait mean", np.mean([a for r, a in zip(rows, tr) if r["pole"] == "neg" and a is not None]))
    log("DONE")

elif STEP == "steer0a":
    m = load_subject()
    cal = torch.load(PRE + "_cal.pt", weights_only=False)
    ell, d = cal["ell"], torch.tensor(cal["d"], dtype=torch.float32)
    qs = [q for q in EV["questions"] for _ in range(N_EVAL)]
    prompts = [m.chat(q) for q in qs]
    base = sample(m, prompts)
    steer = sample(m, prompts, lambda B, L: dict(start=L - 1, add={ell: (COEF * d)[None].expand(B, -1).to(m.dev, torch.bfloat16)}, pos0=0))
    with open(PRE + "_0a_texts.jsonl", "w", encoding="utf-8") as f:
        for q, a, b in zip(qs, base, steer):
            f.write(json.dumps(dict(question=q, baseline=a, steered=b), ensure_ascii=False) + "\n")
    log("DONE")

elif STEP == "judge0a":
    rows = [json.loads(l) for l in open(PRE + "_0a_texts.jsonl", encoding="utf-8")]
    res = {}
    for cond in ("baseline", "steered"):
        tr = judge_scores([EX["eval_prompt"].format(question=r["question"], answer=r[cond]) for r in rows])
        co = judge_scores([COH.format(question=r["question"], answer=r[cond]) for r in rows])
        res[f"trait_{cond}"] = float(np.mean([v for v in tr if v is not None]))
        res[f"coherence_{cond}"] = float(np.mean([v for v in co if v is not None]))
    cal = torch.load(PRE + "_cal.pt", weights_only=False)
    res.update(coef=COEF, n=len(rows), n_effective=int(cal.get("n_effective", 0)), n_pairs=int(cal.get("n_pairs", 0)))
    json.dump(res, open(PRE + "_0a.json", "w"), indent=1)
    log(res)
    log("DONE")
