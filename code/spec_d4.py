"""D4 reasoning-memorisation features, LiReFs (Hong et al. 2025, github.com/yihuaihong/Linear_Reasoning_Features)
for paper 6 (PREREGISTRATION.md §3).

Direction, as in their code: the last-token hidden state of 'Q: {question}\nA: ' (raw text, no chat
template) for their 3,000 MMLU-Pro questions; candidate direction at each layer = mean(reasoning) -
mean(memory), reasoning = memory_reason_score > 0.5. Their hidden_states index l (l = 0 embeddings) is
the output of block l-1 here. Layer rule, as in features_intervention.py: for every layer l > 2, scale
the component along that layer's direction by 1 + 0.1 at every block input and every attention and
MLP output, run 5-shot chain-of-thought MMLU-Pro on 200 of the 3,000 questions (200 new tokens), and
keep the layer with the highest accuracy on the reasoning subset.

Two item sets share the direction:
- D4  (key "d4"): paper 4's multiplication items and targets (settling depth D, shown working W), exactly
  as D3, so the two published 'reasoning vs recall' directions meet on the same items.
- D4G (key "d4g"): GSM8K items; tracking target = CoT necessity (accuracy with chain of thought minus
  accuracy answering directly, 4 samples each); drive target = direct-answer accuracy.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re

import numpy as np
import torch
import transformers

from specs import D3, Spec, char_tok

HERE = os.path.dirname(os.path.abspath(__file__))
LDATA = os.path.join(HERE, "data", "lirefs")
CATS = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology', 'health', 'physics',
        'business', 'philosophy', 'economics', 'other', 'psychology', 'history']
SCALE = 0.1
N_SWEEP = 200
N_NEW_SWEEP = 200


def form_options(options):
    s = 'Options are:\n'
    for opt, o in zip(options, 'ABCDEFGHIJ'):
        s += f'({o}): {opt}' + '\n'
    return s


def as_list(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


def mmlu_pred(text, rng):
    mm = re.search(r"answer is \(?([ABCDEFGHIJ])\)?", text)
    return mm.group(1) if mm else rng.choice(list("ABCDEFGHIJ"))  # their scoring: random guess when unparsed


def load_cal_data():
    data = json.load(open(os.path.join(LDATA, "mmlu-pro-3000samples.json"), encoding="utf-8"))
    for x in data:
        x["memory_reason_score"] = float(x["memory_reason_score"])
        x["options"] = as_list(x["options"])
    return data


def cot_prefixes():
    val = json.load(open(os.path.join(LDATA, "mmlupro_validation.json"), encoding="utf-8"))
    pre = {c: "" for c in CATS}
    for d in val:
        pre[d["category"]] += 'Q:' + ' ' + d['question'] + '\n' + form_options(as_list(d['options'])) + '\n' + d['cot_content'] + '\n\n'
    return pre


class ProjScale:
    """Their intervention: x <- x + SCALE * (x . u) u at every block input and attention/MLP output."""

    def __init__(self, m, direction, scale=SCALE):
        self.h = []
        u = torch.tensor(direction / np.linalg.norm(direction), dtype=torch.float32, device=m.dev)

        def f(x):
            uu = u.to(x.dtype)
            return x + scale * (x @ uu)[..., None] * uu

        def pre(mod, inp):
            return (f(inp[0]),) + tuple(inp[1:]) if isinstance(inp, tuple) else f(inp)

        def post(mod, inp, out):
            return (f(out[0]),) + tuple(out[1:]) if isinstance(out, tuple) else f(out)

        for layer in m.layers:
            self.h.append(layer.register_forward_pre_hook(pre))
            self.h.append(layer.self_attn.register_forward_hook(post))
            self.h.append(layer.mlp.register_forward_hook(post))

    def remove(self):
        for h in self.h:
            h.remove()


@torch.no_grad()
def gen_texts(m, prompts, n_new, bs, ids_fn):
    """Greedy, left-padded batches; returns decoded continuations in input order."""
    order = sorted(range(len(prompts)), key=lambda i: -len(ids_fn(prompts[i])))
    out = [None] * len(prompts)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        b = [ids_fn(prompts[i]) for i in idx]
        L = max(len(r) for r in b)
        x = torch.tensor([[pad] * (L - len(r)) + r for r in b], device=m.dev)
        am = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in b], device=m.dev)
        pos = (am.cumsum(1) - 1).clamp_min(0)
        cache = transformers.DynamicCache()
        toks = []
        for _ in range(n_new):
            lg = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
            nxt = lg.argmax(-1)
            toks.append(nxt)
            x = nxt[:, None]
            am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
            pos = pos[:, -1:] + 1
        del cache
        T = torch.stack(toks, 1).cpu()
        for j, i in enumerate(idx):
            out[i] = m.decode(T[j])
        torch.cuda.empty_cache()
    return out


class D4(D3):
    key, name = "d4", "lirefs"

    def calibrate(self, m, log):
        data = load_cal_data()
        L = m.L
        y = np.array([int(x["memory_reason_score"] > 0.5) for x in data])
        A = np.zeros((len(data), L + 1, m.d), dtype=np.float16)  # index = their hidden_states index
        lengths = []
        for i, x in enumerate(data):
            ids = m.raw_ids('Q: ' + x['question'] + "\nA: ")
            lengths.append(len(ids))
            a = m.acts(ids, list(range(L)), [len(ids) - 1])
            emb = m.hf.model.embed_tokens(torch.tensor([ids[-1:]], device=m.dev))[0, 0].float().cpu().numpy()
            A[i, 0] = emb
            for l in range(L):
                A[i, l + 1] = a[l][0]
        md = A[y == 1].astype(np.float64).mean(0) - A[y == 0].astype(np.float64).mean(0)  # [L+1, d]
        # layer sweep (their rule), on one fixed sample of 200 questions for every layer
        pre = cot_prefixes()
        sweep = random.Random(8888).sample(range(len(data)), N_SWEEP)
        prompts = [pre[data[i]['category']] + 'Q: ' + data[i]['question'] + '\n' + form_options(data[i]['options']) + "\n\nA: Let's think step by step. " for i in sweep]
        bs = int(os.environ.get("SWEEP_BS", "8"))
        ids_fn = lambda p: m.raw_ids(p)

        def accuracy(texts):
            rng = random.Random(8888)
            ok = [mmlu_pred(t, rng) == data[i]["answer"] for t, i in zip(texts, sweep)]
            r = [o for o, i in zip(ok, sweep) if y[i] == 1]
            mm = [o for o, i in zip(ok, sweep) if y[i] == 0]
            return dict(all=float(np.mean(ok)), reason=float(np.mean(r)), memory=float(np.mean(mm)))

        base = accuracy(gen_texts(m, prompts, N_NEW_SWEEP, bs, ids_fn))
        log(f"sweep baseline: {base}")
        per = {}
        for l in range(3, L + 1):
            if l == L:  # their loop runs over layers 0..L-1 of hidden_states
                break
            h = ProjScale(m, md[l])
            try:
                per[l] = accuracy(gen_texts(m, prompts, N_NEW_SWEEP, bs, ids_fn))
            finally:
                h.remove()
            log(f"sweep layer {l}: {per[l]}")
        lstar = max(per, key=lambda l: (per[l]["reason"], -l))
        ell = lstar - 1  # hidden_states[l] = output of block l-1
        log(f"selected hidden_states layer {lstar} (block output {ell}): reason acc {per[lstar]['reason']:.3f} vs baseline {base['reason']:.3f}")
        return dict(ell=ell, lstar=lstar, d=md[lstar], H=A[:, lstar].astype(np.float64), y=y.astype(float), lengths=lengths,
                    sweep=dict(baseline=base, per_layer=per, items=sweep), prompts_raw=['Q: ' + x['question'] + "\nA: " for x in data])

    def cal_acts(self, m, cal):
        H = []
        for p in cal["prompts_raw"]:
            ids = m.raw_ids(p)
            H.append(m.acts(ids, [cal["ell"]], [len(ids) - 1])[cal["ell"]][0])
        return np.array(H, dtype=np.float64), cal["y"]

    def check0a(self, m, cal, gen_padded, log):
        """Headline effects: (i) the selected direction separates held-out reasoning from memory questions
        (two-fold AUC on the calibration set); (ii) their intervention at the selected layer raises
        reasoning accuracy on 200 other questions (the sweep's selection is in-sample)."""
        from p6lib import twofold_auc
        data = load_cal_data()
        y = np.array(cal["y"]).astype(int)
        res = dict(twofold_auc=twofold_auc(cal["H"], y), sweep_baseline=cal["sweep"]["baseline"],
                   sweep_selected=cal["sweep"]["per_layer"][cal["lstar"]], lstar=cal["lstar"])
        pre = cot_prefixes()
        rest = [i for i in range(len(data)) if i not in set(cal["sweep"]["items"])]
        held = random.Random(8889).sample(rest, N_SWEEP)
        prompts = [pre[data[i]['category']] + 'Q: ' + data[i]['question'] + '\n' + form_options(data[i]['options']) + "\n\nA: Let's think step by step. " for i in held]
        bs = int(os.environ.get("SWEEP_BS", "8"))

        def acc(texts):
            rng = random.Random(8888)
            ok = [mmlu_pred(t, rng) == data[i]["answer"] for t, i in zip(texts, held)]
            return float(np.mean([o for o, i in zip(ok, held) if y[i] == 1]))
        res["heldout_reason_baseline"] = acc(gen_texts(m, prompts, N_NEW_SWEEP, bs, m.raw_ids))
        h = ProjScale(m, cal["d"])
        try:
            res["heldout_reason_intervened"] = acc(gen_texts(m, prompts, N_NEW_SWEEP, bs, m.raw_ids))
        finally:
            h.remove()
        log(f"0a LiReFs: two-fold AUC {res['twofold_auc']:.3f}; held-out reasoning accuracy {res['heldout_reason_baseline']:.3f} -> "
            f"{res['heldout_reason_intervened']:.3f} with their intervention")
        return res

    def items(self, phase):
        return _d3_items(phase)  # paper 4's items, exactly as D3


def _d3_items(phase):
    from specs import ITEMS
    I = ITEMS["d3"]
    if phase == "pilot":
        pil = I["pilot"]
        pd = set(I["pilot_D"])
        return dict(A=pil, oracle=[x for x in pil if x["id"] not in pd], D=I["pilot_D"], E=I["pilot_D"][:10], B=[])
    return dict(A=I["A"], oracle=I["oracle"], D=I["D"], E=I["E"], B=[])


# =================================================================== D4G
NUMRE = re.compile(r"-?\d[\d,]*\.?\d*")


def gsm_answer(text):
    mm = re.search(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    if mm:
        s = mm.group(1)
    else:
        nums = NUMRE.findall(text)
        if not nums:
            return None
        s = nums[-1]
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


DIRECT_PREFILL = "The answer is "
COT_PREFILL = "Let's think step by step."


class D4G(Spec):
    """LiReFs on GSM8K items. Reading position: the last token of the chat prompt (identical for both
    modes, which differ only in the assistant prefill)."""
    key, name = "d4g", "lirefs_gsm8k"
    modes = ("direct",)
    n_new_mode = dict(direct=16)
    binary_T = False

    def items(self, phase):
        I = json.load(open(os.path.join(HERE, "data", "items_d4g.json"), encoding="utf-8"))
        if phase == "pilot":
            pil = I["pilot"]
            pd = [x["id"] for x in pil][:20]
            return dict(A=pil, oracle=[x for x in pil if x["id"] not in pd], D=pd, E=pd[:6], B=[])
        return dict(A=I["A"], oracle=I["oracle"], D=I["D"], E=I["E"], B=[])

    def base_text(self, m, it):
        return m.chat(it["question"])

    def prompt_ids(self, m, it, mode):
        base = self.base_text(m, it)
        ids = m.ids(base + (DIRECT_PREFILL if mode == "direct" else COT_PREFILL))
        n0 = len(m.ids(base))
        return ids, n0 - 1

    def signs(self, it):
        return (1, -1)

    def outcome(self, m, it, mode, toks, lp0, top1=None):
        text = m.decode(toks)
        a = gsm_answer(DIRECT_PREFILL + text if mode == "direct" else text)
        return dict(ans=a, correct=int(a is not None and abs(a - float(it["answer"].replace(",", ""))) < 1e-6), text=text[:200])


@torch.no_grad()
def sample_texts(m, ids_list, n_new, temp, seed):
    """Temperature sampling for a left-padded batch (one torch generator per call, fixed seed)."""
    gen = torch.Generator(device=m.dev).manual_seed(seed)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    L = max(len(r) for r in ids_list)
    x = torch.tensor([[pad] * (L - len(r)) + r for r in ids_list], device=m.dev)
    am = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in ids_list], device=m.dev)
    pos = (am.cumsum(1) - 1).clamp_min(0)
    cache = transformers.DynamicCache()
    toks = []
    done = torch.zeros(len(ids_list), dtype=torch.bool, device=m.dev)
    eos = torch.tensor(sorted(m.eos), device=m.dev)
    for _ in range(n_new):
        lg = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1].float()
        nxt = torch.multinomial(torch.softmax(lg / temp, -1), 1, generator=gen)[:, 0]
        nxt = torch.where(done, eos[0], nxt)
        toks.append(nxt)
        done |= torch.isin(nxt, eos)
        if bool(done.all()):
            break
        x = nxt[:, None]
        am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
        pos = pos[:, -1:] + 1
    del cache
    T = torch.stack(toks, 1).cpu()
    torch.cuda.empty_cache()
    return [m.decode(t) for t in T]


N_SAMPLES, TEMP, N_COT, N_DIRECT = 4, 0.7, 384, 16
