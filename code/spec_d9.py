"""D9 true-thinking direction (Zhao, Sun, Shi & Song 2025, arXiv:2510.24941; code
github.com/andotalao24/Identify_true_decorative_thinking, commit e1da4c0 in third_party/ttd) for
paper 6 (PREREGISTRATION.md §3; settings in PREREGISTRATION_STAGE2B_D9.md).

Their pipeline, rebuilt here in each panel model:
- greedy chains of thought on the AMC 2022-23 problems (AI-MO/aimo-validation-amc), kept if the last
  \\boxed{} answer is correct; a chain is split into steps at every "." (their delimiter);
- True Thinking Score per step: early exit with "\\n The final answer is \\boxed{" after four prefixes,
  (C, s), (C, s'), (C', s), (C', s'), where s' adds a random offset in {-3,-2,-1,1,2} to every number
  of the step (their random_add_small, seed 100) and C' does the same to every earlier step;
  TTS = (|p(y*|C,s) - p(y*|C,s')| + |p(y*|C',s) - p(y*|C',s')|) / 2, y* = the chain's own answer;
- direction: residual stream at the step's last token in their extraction format
  "<prompt>. <context>. <step>", mean over steps with TTS > 0.9 minus mean over steps with TTS <= 0.005
  (100 + 100 after their seed-100 sampling), at layer index 17 of 28 (their example layer).
  Their released vectors point the other way (decorative minus true); d here is the TrueThinking
  direction, so the claimed relation is positive for tracking and pushing.

Process targets: tracking, each numeric step's TTS; drive, whether the early-exit answer follows a
perturbed step (F = 1 if it differs from y*), on one perturbed step per Check D problem.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import transformers

from specs import Spec, _find_up

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res", "d9")
CUE = "\n The final answer is \\boxed{"  # their user_tag (inference_checkpoint_analysis.py)
SEED_PERTURB = 100  # run_checkpoint_analysis.sh: SEEDS="100"
T_HIGH, T_LOW = 0.9, 0.005  # tts.py t_ff=0.9; t_uf=0 becomes the paper's decorative cut-off 0.005 (Stage 2b §2)
N_SAMPLE, N_CLASS = 200, 100  # tts.py samples min(200, n) with seed 100; run_diff_mean.sh uses items 0-99
CAP_CAL, CAP_TEST = 8192, 4096  # new-token caps for calibration (AMC) and MATH chains
N_ANS = 16  # tokens generated for an early-exit answer
N_0A = 100  # Check 0a: engagement and disengagement test steps each
N_B = 120  # Check B: matched pairs
D_MAX_POS = 2048  # Check D units: steps whose intervention-format prompt is shorter than this
RELEASED = {"r1-qwen-7b": "amc-qwen-dir-uf-ff.pt", "r1-llama-8b": "llama8b-amc-uf-ff-dir.pt"}
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


# ------------------------------------------------------------------ text helpers (their logic)
def perturb_text(text, seed=SEED_PERTURB):
    """utils_eval.randomly_replace_numbers(mode='random_add_small', replace_all=True): a fresh
    random.Random(seed) per step, every number gets old + choice([-3, -2, -1, 1, 2])."""
    rng = random.Random(seed)
    offs = [x for x in range(-3, 3) if x != 0]

    def repl(mm):
        s = mm.group(0)
        try:
            new = float(s) + rng.choice(offs)
            if "." in s:
                return f"{new:.1f}" if (new % 1) != 0 else f"{int(new)}.0"
            return str(int(new))
        except Exception:
            return s

    out = NUM_RE.sub(repl, text)
    return out, out != text


def split_steps(output):
    """Their checkpoint split: ori_output.split('.'), dropping an empty last piece."""
    ch = output.split(".")
    if ch and ch[-1].strip() == "":
        ch = ch[:-1]
    return ch


def boxed_span(text, start):
    """Content from `start` up to the brace that closes an already-open '{' (brace matched)."""
    depth, k = 1, start
    while k < len(text):
        depth += (text[k] == "{") - (text[k] == "}")
        if depth == 0:
            return text[start:k]
        k += 1
    return None


def last_boxed(text):
    i = text.rfind("\\boxed{")
    return None if i < 0 else boxed_span(text, i + 7)


def early_answer(gen_text):
    """Answer generated after the cue (which ends with an open '{')."""
    a = boxed_span(gen_text, 0)
    return (a if a is not None else gen_text).strip()


def compare_answers(extracted_ans, correct_ans):
    """utils_eval.compare_answers (copied logic)."""
    extracted_ans = str(extracted_ans).split("$")[0].split("\n")[0]
    for sign in [",", ":", "%"]:
        extracted_ans = extracted_ans.replace(sign, "")
    extracted_ans = str(extracted_ans).strip()
    correct_ans = str(correct_ans).strip()
    if extracted_ans == correct_ans:
        return True
    try:
        if abs(float(extracted_ans) - float(correct_ans)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    try:
        if "/" in extracted_ans:
            num, den = extracted_ans.split("/")
            return abs(float(num) / float(den) - float(correct_ans)) < 1e-6
        if "/" in correct_ans:
            num, den = correct_ans.split("/")
            return abs(float(num) / float(den) - float(extracted_ans)) < 1e-6
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False


def tok_index(m, text, idx, first=True):
    """(ids, index of the token covering character idx). Tokenizers that trim a leading space from
    a token's offsets leave that space uncovered: then the first token starting at or after idx
    (first=True) or the last token ending at or before idx + 1 (first=False) is used."""
    enc = m.tok(text, add_special_tokens=not text.startswith(m.tok.bos_token or "\x00"), return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    for i, (a, b) in enumerate(offs):
        if a <= idx < b:
            return enc["input_ids"], i
    if first:
        return enc["input_ids"], next(i for i, (a, b) in enumerate(offs) if a >= idx and b > a)
    return enc["input_ids"], max(i for i, (a, b) in enumerate(offs) if b <= idx + 1 and b > a)


def ext_text(base, chunks, j, step=None):
    """Their extraction / intervention format: formatInp(...) + '. ' + context + '. ' + step."""
    return f"{base}. {'.'.join(chunks[:j])}. {chunks[j] if step is None else step}"


def is_consistent(rec):
    """tts.py's is_consistent on the (C, s') early-exit answers: True (case skipped) unless some
    checkpoint after the first correct one gives a wrong answer."""
    num = {s["j"]: s for s in rec["sentences"]}
    g = [rec["g_base"]] + [(1 - num[j]["F"]) if j in num else rec["g0"][j] for j in range(rec["n_steps"])]
    first = next((k for k, v in enumerate(g) if v), len(g) - 1)
    return sum(1 for v in g[first + 1:] if not v) == 0


# ----------------------------------------------------------------------- model helpers
def prefix(m, problem):
    t = m.chat(problem)
    if not t.rstrip().endswith("<think>"):
        t = t + "<think>\n"
    elif not t.endswith("\n"):
        t = t + "\n"
    return t


def chain_bs(m, cap):
    if os.environ.get("CHAIN_BS"):
        return int(os.environ["CHAIN_BS"])
    c = m.hf.config
    nkv = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    per_tok = 2 * m.L * nkv * hd * 2
    total = torch.cuda.get_device_properties(0).total_memory * float(os.environ.get("P6_MEM_FRACTION", "0.85"))
    model_b = sum(p.numel() * p.element_size() for p in m.hf.parameters())
    # the growing cache fragments the allocator: the first pilot ran out of memory with a 21 GB cache
    # estimate (8.8 GB reserved but unallocated), while 8 GB caches ran in Qwen3-8B (timing probe)
    budget = max(1e9, min(0.75 * (total - model_b - 2e9), 10e9))
    return int(max(1, min(32, budget // ((cap + 512) * per_tok))))


@torch.no_grad()
def gen_chains(m, batch_ids, cap, bs):
    """Greedy left-padded generation to EOS or `cap` new tokens. Rows that end are dropped from the
    batch and its cache."""
    outs = []
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    eos_t = torch.tensor(sorted(m.eos), device=m.dev)
    for i in range(0, len(batch_ids), bs):
        b = batch_ids[i:i + bs]
        L = max(len(r) for r in b)
        x = torch.tensor([[pad] * (L - len(r)) + r for r in b], device=m.dev)
        am = torch.tensor([[0] * (L - len(r)) + [1] * len(r) for r in b], device=m.dev)
        pos = (am.cumsum(1) - 1).clamp_min(0)
        cache = transformers.DynamicCache()
        rows = list(range(len(b)))
        toks = [[] for _ in b]
        for _ in range(cap):
            lg = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True,
                      logits_to_keep=1).logits[:, -1]
            nxt = lg.argmax(-1)
            nl = nxt.tolist()
            fin = [t in m.eos for t in nl]
            for r, t in zip(rows, nl):
                toks[r].append(t)
            if all(fin):
                break
            if any(fin):
                keep = [k for k, f in enumerate(fin) if not f]
                kt = torch.tensor(keep, device=m.dev)
                cache.batch_select_indices(kt)
                rows, nxt, am, pos = [rows[k] for k in keep], nxt[kt], am[kt], pos[kt]
            x = nxt[:, None]
            am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
            pos = pos[:, -1:] + 1
        del cache
        torch.cuda.empty_cache()
        outs += [m.trunc(t) for t in toks]
    return outs


def gen_chains_safe(m, batch_ids, cap, log):
    """gen_chains for one batch; on running out of GPU memory, the batch is split in two and retried."""
    oom = False
    try:
        return gen_chains(m, batch_ids, cap, len(batch_ids))
    except torch.cuda.OutOfMemoryError:
        oom = True
    if oom:  # outside the except block, so the failed frame's tensors are released
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if len(batch_ids) == 1:
            raise RuntimeError("out of memory with a batch of one chain")
        h = len(batch_ids) // 2
        log(f"  out of GPU memory with {len(batch_ids)} chains: splitting the batch")
        return gen_chains_safe(m, batch_ids[:h], cap, log) + gen_chains_safe(m, batch_ids[h:], cap, log)


@torch.no_grad()
def greedy_answer(m, text):
    ids = m.ids(text + CUE)
    r = m.generate(ids, len(ids), {0: torch.zeros(1, m.d)}, N_ANS)
    return early_answer(m.decode(r["tokens"][0]))


class _Ref:
    """One forward pass over a reference text; its cache is branched for every prefix of it."""

    def __init__(self, m, text):
        self.ids = m.ids(text)
        self.np = np.asarray(self.ids)
        self.cache = transformers.DynamicCache()
        m.hf(input_ids=torch.tensor([self.ids], device=m.dev), past_key_values=self.cache, use_cache=True,
             logits_to_keep=1)

    def lcp(self, ids, upto):
        n = min(len(self.ids), len(ids), upto)
        d = np.nonzero(self.np[:n] != np.asarray(ids[:n]))[0]
        return int(d[0]) if len(d) else n

    def branch(self, ids, upto):
        P = self.lcp(ids, upto)
        c = copy.deepcopy(self.cache)
        c.crop(P)
        return P, c


@torch.no_grad()
def _ee(m, ref, texts, ystar):
    """Early exits after each of `texts` (prefixes that share a start with ref's text): log p(y* |
    text + cue), teacher-forced, and whether greedy decoding gives exactly y* followed by '}' (their
    answer-is-right test). The texts are run as one batch branched from the same cache position, so a
    difference between them (a TTS term) is computed on one numerical path."""
    rows = []
    for t in texts:
        ids_p = m.ids(t + CUE)
        ids_f = m.ids(t + CUE + ystar)
        if len(ids_f) <= len(ids_p) or ids_f[:len(ids_p)] != ids_p:
            ids_f = ids_p + m.tok.encode(ystar, add_special_tokens=False)
        rows.append((len(ids_p), ids_f))
    P = min(ref.lcp(f, n_p - 1) for n_p, f in rows)
    c = copy.deepcopy(ref.cache)
    c.crop(P)
    if len(rows) > 1:
        c.batch_repeat_interleave(len(rows))
    W = max(len(f) for _, f in rows) - P
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    x = torch.tensor([f[P:] + [pad] * (W - len(f) + P) for _, f in rows], device=m.dev)  # right-padded
    pos = torch.arange(P, P + W, device=m.dev)[None].expand(len(rows), -1)
    lg = m.hf(input_ids=x, past_key_values=c, use_cache=True, position_ids=pos).logits
    del c
    out = []
    for b, (n_p, f) in enumerate(rows):
        ans = f[n_p:]
        na = len(ans)
        lp = torch.log_softmax(lg[b, n_p - 1 - P: n_p + na - P].float(), -1)  # predicts ans[0..na-1], then '}'
        am = lp.argmax(-1).tolist()
        greedy = all(am[k] == ans[k] for k in range(na)) and "}" in m.tok.decode([am[na]])
        out.append((float(sum(lp[k, ans[k]] for k in range(na))), int(greedy)))
    return out


@torch.no_grad()
def _read(m, ref, text, ell):
    """Block-`ell` output at the last token of `text` (branched from ref's cache)."""
    ids = m.ids(text)
    P, c = ref.branch(ids, len(ids) - 1)
    m.capture = {ell: None}
    try:
        m.hf.model(input_ids=torch.tensor([ids[P:]], device=m.dev), past_key_values=c, use_cache=True,
                   position_ids=torch.arange(P, len(ids), device=m.dev)[None])
        h = m.capture[ell][0, -1].float().cpu().numpy()
    finally:
        m.capture = None
    del c
    return h, len(ids) - 1


@torch.no_grad()
def score_chain(m, base, output, ystar, read_ell=None):
    """Per-step early-exit scores of one chain. Every step gets the (C, s) early exit; numeric steps
    (those the perturbation changes) get all four conditions, the TTS, F (the (C, s') answer is not
    y*), F_C (the (C) answer is y*) and, with read_ell, the block output at their extraction position."""
    chunks = split_steps(output)
    n = len(chunks)
    pert = [perturb_text(c) for c in chunks]
    cp = [x for x, _ in pert]
    isnum = [ch for _, ch in pert]
    refA = _Ref(m, base + ".".join(chunks))
    lp_base, g_base = _ee(m, refA, [base], ystar)[0]
    s0, s1 = [None] * n, {}
    for j in range(n):  # (C, s) for every step; numeric steps in one batch with (C, s')
        if isnum[j]:
            s0[j], s1[j] = _ee(m, refA, [base + ".".join(chunks[:j + 1]), base + ".".join(chunks[:j] + [cp[j]])], ystar)
        else:
            s0[j] = _ee(m, refA, [base + ".".join(chunks[:j + 1])], ystar)[0]
    refB = _Ref(m, base + ".".join(cp)) if any(isnum) else None
    refX = _Ref(m, base + ". " + ".".join(chunks)) if (read_ell is not None and any(isnum)) else None
    steps, reads = [], {}
    for j in range(n):
        if not isnum[j]:
            continue
        a, b = s0[j], s1[j]
        c, d = _ee(m, refB, [base + ".".join(cp[:j] + [chunks[j]]), base + ".".join(cp[:j + 1])], ystar)  # (C', s), (C', s')
        p = [math.exp(x[0]) for x in (a, b, c, d)]
        nec, suf = p[0] - p[1], p[2] - p[3]
        st = dict(j=j, T=0.5 * (abs(nec) + abs(suf)), nec=nec, suf=suf, p=p, g=[a[1], b[1], c[1], d[1]],
                  F=1 - b[1], F_C=g_base if j == 0 else s0[j - 1][1],
                  step_len=len(m.tok.encode(chunks[j], add_special_tokens=False)),
                  text=chunks[j][:300], text_pert=cp[j][:300])
        if refX is not None:
            h, pos = _read(m, refX, ext_text(base, chunks, j), read_ell)
            reads[j] = h
            st["pos"] = pos
        steps.append(st)
    del refA, refB, refX
    torch.cuda.empty_cache()
    return dict(n_steps=n, n_num=int(sum(isnum)), g_base=g_base, p_base=math.exp(lp_base), g0=[x[1] for x in s0],
                p0=[round(math.exp(x[0]), 6) for x in s0], sentences=steps), reads


def process_problems(m, problems, cap, jsonl, log, read_ell=None, npz=None, need_correct=False):
    """Chains (greedy, `cap` new tokens) and per-step scores for `problems`. Resumable: one JSON line
    per problem; hidden states (read_ell) in `npz`, saved before the batch's lines are written."""
    os.makedirs(os.path.dirname(jsonl), exist_ok=True)
    done = {}
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    H = dict(np.load(npz)) if (npz and os.path.exists(npz)) else {}
    todo = [p for p in problems if p["id"] not in done]
    bs = chain_bs(m, cap)
    log(f"chains -> {os.path.basename(jsonl)}: {len(todo)} of {len(problems)} to run (cap {cap}, batch {bs})")
    for b0 in range(0, len(todo), bs):
        t0 = time.time()
        batch = todo[b0:b0 + bs]
        bases = [prefix(m, p["problem"]) for p in batch]
        outs = gen_chains_safe(m, [m.ids(b) for b in bases], cap, log)
        t1 = time.time()
        lines = []
        for p, base, toks in zip(batch, bases, outs):
            text = m.tok.decode(toks, skip_special_tokens=False).strip()
            yb = last_boxed(text)
            ystar = yb if yb is not None else greedy_answer(m, base + text)
            correct = compare_answers(ystar, p["answer"]) if p.get("answer") is not None else None
            rec = dict(id=p["id"], n_tok=len(toks), finished=len(toks) < cap, boxed=yb is not None, ystar=ystar,
                       correct=correct, output=text)
            if need_correct and not (correct and yb is not None):
                rec["scored"] = False
            else:
                sc, reads = score_chain(m, base, text, ystar, read_ell)
                rec.update(sc, scored=True)
                for j, h in reads.items():
                    H[f"{p['id']}:{j}"] = h.astype(np.float32)
            lines.append(rec)
        if npz:
            np.savez(npz, **H)
        with open(jsonl, "a", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")
                done[rec["id"]] = rec
        sc = [r for r in lines if r.get("scored")]
        log(f"  {b0 + len(batch)}/{len(todo)}: gen {t1 - t0:.0f}s, score {time.time() - t1:.0f}s; "
            f"tokens {[r['n_tok'] for r in lines]}; scored {len(sc)}, numeric steps {sum(r['n_num'] for r in sc)}")
    return done, H


def choose_unit(rec, pref, problem_id):
    """Check D unit: one perturbed numeric step of the model's own chain. 'dt' prefers a step whose
    perturbation moves the answer off y* while the answer before it is y* (F = 1, F_C = 1, Zhao et al.'s
    disengagement-test condition); 'et' prefers F = 0, F_C = 1 (engagement test); each falls back to
    the other, then to any numeric step. Only steps whose prompt in the intervention format has at most
    D_MAX_POS tokens are eligible (push batches hold ~170 copies of the prompt), unless none is.
    Seeded by the problem id."""
    st = rec.get("sentences") or []
    st = [s for s in st if s.get("pos", 0) < D_MAX_POS] or st
    dt = [s for s in st if s["F_C"] == 1 and s["F"] == 1]
    et = [s for s in st if s["F_C"] == 1 and s["F"] == 0]
    rng = random.Random(f"{problem_id}:unit")
    pick, kind = None, "none"
    for nm, pool in ((("dt", dt), ("et", et)) if pref == "dt" else (("et", et), ("dt", dt))):
        if pool:
            pick, kind = rng.choice(pool), nm
            break
    if pick is None and st:
        pick, kind = rng.choice(st), "any"
    chunks = split_steps(rec["output"])
    if pick is None:  # no numeric step: the last step, unperturbed
        j = len(chunks) - 1
        return dict(j=j, kind="none", ystar=rec["ystar"], ctx=".".join(chunks[:j]), s=chunks[j], s_pert=chunks[j])
    j = pick["j"]
    return dict(j=j, kind=kind, pref=pref, ystar=rec["ystar"], ctx=".".join(chunks[:j]), s=chunks[j],
                s_pert=perturb_text(chunks[j])[0], F_nat=pick["F"], F_C=pick["F_C"], T=pick["T"])


@torch.no_grad()
def gen_span(m, ids, t0, t1, ell, vecs, n_new):
    """Zhao et al.'s intervention (intervention.py with intervene_on_perturbed_step=1,
    max_decode_step_while_intervene=1): add vecs[b] to the residual stream entering block ell+1 (our
    block-ell output) on the step's tokens t0..t1, in the prompt pass only; then greedy decoding."""
    B = vecs.shape[0]
    cache = transformers.DynamicCache()
    m.ops = dict(start=t0, add={ell: vecs.to(m.dev, torch.bfloat16)}, pos0=0)
    try:
        m.hf(input_ids=torch.tensor([ids[:t1 + 1]] * B, device=m.dev), past_key_values=cache, use_cache=True,
             position_ids=torch.arange(0, t1 + 1, device=m.dev)[None].expand(B, -1), logits_to_keep=1)
    finally:
        m.ops = None
    out = m.hf(input_ids=torch.tensor([ids[t1 + 1:]] * B, device=m.dev), past_key_values=cache, use_cache=True,
               position_ids=torch.arange(t1 + 1, len(ids), device=m.dev)[None].expand(B, -1), logits_to_keep=1)
    T, toks = len(ids), []
    for _ in range(n_new):
        nxt = out.logits[:, -1].argmax(-1)
        toks.append(nxt)
        out = m.hf(input_ids=nxt[:, None], past_key_values=cache, use_cache=True,
                   position_ids=torch.full((B, 1), T, device=m.dev), logits_to_keep=1)
        T += 1
    del cache
    return torch.stack(toks, 1).cpu()


def released_for(key):
    f = RELEASED.get(key)
    if f is None:
        return None
    p = os.path.join(_find_up("third_party", "ttd"), "data", "pt", f)
    return torch.load(p, map_location="cpu", weights_only=True).double().numpy()  # [L+1, d]; index l = input of block l


def _cos(a, b):
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


_I, _UNITS = None, {}


def ITEMS_D9():
    global _I
    if _I is None:
        _I = json.load(open(os.path.join(HERE, "data", "items_d9.json"), encoding="utf-8"))
    return _I


def unit_for(key, iid):
    if key not in _UNITS:
        u = {}
        for ph in ("pilot", "main"):
            p = os.path.join(RES, f"{key}_{ph}_clean.jsonl")
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    r = json.loads(line)
                    if r.get("unit") is not None:
                        u[r["id"]] = r["unit"]
        _UNITS[key] = u
    return _UNITS[key][iid]


# =========================================================================== spec
class D9(Spec):
    key, name = "d9", "true_thinking"
    thinking = True
    modes = ("gen",)
    n_new_mode = dict(gen=N_ANS)

    def layer(self, m):
        """Their example layer is index 17 of 28 in R1-Distill-Qwen-7B (input of block 17 = our
        block-16 output); the same relative depth elsewhere."""
        return int(round(17 * m.L / 28)) - 1

    def prefix(self, m, problem):
        return prefix(m, problem)

    def _path(self, m, name):
        return os.path.join(RES, f"{m.key}_{name}.jsonl")

    # ------------------------------------------------------------ calibration
    def calibrate(self, m, log):
        I = ITEMS_D9()
        ell = self.layer(m)
        recs, _ = process_problems(m, I["cal"], CAP_CAL, self._path(m, "cal_chains"), log, need_correct=True)
        sc = [recs[p["id"]] for p in I["cal"] if recs[p["id"]].get("scored")]
        kept = [r for r in sc if not is_consistent(r)]
        uf, ff = [], []
        for p in I["cal"]:
            r = recs[p["id"]]
            if not r.get("scored") or is_consistent(r):
                continue
            for st in r["sentences"]:
                if st["T"] <= T_LOW:
                    uf.append((p, r, st))
                elif st["T"] > T_HIGH:
                    ff.append((p, r, st))
        random.seed(100)
        low = random.sample(uf, min(N_SAMPLE, len(uf)))[:N_CLASS]
        random.seed(100)
        high = random.sample(ff, min(N_SAMPLE, len(ff)))[:N_CLASS]
        allT = np.array([st["T"] for r in sc for st in r["sentences"]])
        summ = dict(n_problems=len(I["cal"]), n_finished=sum(recs[p["id"]]["finished"] for p in I["cal"]),
                    n_correct=len(sc), n_inconsistent_kept=len(kept), n_steps=int(sum(r["n_steps"] for r in sc)),
                    n_numeric=int(len(allT)), tts_mean=float(allT.mean()) if len(allT) else None,
                    frac_gt_0_3=float((allT > 0.3).mean()) if len(allT) else None,
                    frac_gt_0_7=float((allT > 0.7).mean()) if len(allT) else None,
                    frac_gt_0_9=float((allT > 0.9).mean()) if len(allT) else None,
                    frac_le_0_005=float((allT <= 0.005).mean()) if len(allT) else None,
                    n_low_pool=len(uf), n_high_pool=len(ff), n_low=len(low), n_high=len(high))
        log("calibration chains:", json.dumps(summ))
        if not high or not low:
            raise RuntimeError("no true-thinking or no decorative steps: the direction cannot be built")
        prompts, y, meta = [], [], []
        for lab, lst in ((1, high), (0, low)):
            for p, r, st in lst:
                prompts.append(ext_text(self.prefix(m, p["problem"]), split_steps(r["output"]), st["j"]))
                y.append(lab)
                meta.append(dict(id=p["id"], j=st["j"], T=st["T"], label=lab))
        HA = np.zeros((len(prompts), m.L, m.d), np.float32)
        lengths = []
        for i, t in enumerate(prompts):  # extract_hidden.py: one prompt per pass, last token, every layer
            ids = m.ids(t)
            a = m.acts(ids, list(range(m.L)), [len(ids) - 1])
            HA[i] = np.stack([a[l][0] for l in range(m.L)])
            lengths.append(len(ids))
        y = np.array(y)
        d_all = HA[y == 1].astype(np.float64).mean(0) - HA[y == 0].astype(np.float64).mean(0)
        rel = released_for(m.key)
        log(f"true-thinking layer {ell}: {int(y.sum())} high-TTS and {int((1 - y).sum())} low-TTS steps; |d|={np.linalg.norm(d_all[ell]):.3f}")
        return dict(ell=ell, d=d_all[ell], H=HA[:, ell].astype(np.float64), y=y.astype(float), lengths=lengths,
                    cal_prompts=prompts, cal_meta=meta, d_all=d_all.astype(np.float32), summary=summ,
                    released=(-rel[ell + 1]) if rel is not None else None)

    def cal_acts(self, m, cal):
        H = []
        for t in cal["cal_prompts"]:
            ids = m.ids(t)
            H.append(m.acts(ids, [cal["ell"]], [len(ids) - 1])[cal["ell"]][0])
        return np.array(H, dtype=np.float64), np.asarray(cal["y"], dtype=float)

    def check0a(self, m, cal, gen_padded, log):
        ell, d_raw = cal["ell"], np.asarray(cal["d"], np.float64)
        res = dict(cal_summary=cal["summary"])
        rel = released_for(m.key)
        if rel is not None:  # released vectors are decorative-minus-true: compare with -released
            res["cos_released"] = _cos(d_raw, -rel[ell + 1])
            res["cos_released_by_layer"] = [_cos(cal["d_all"][l], -rel[l + 1]) for l in range(m.L)]
            res["norm_ratio_rebuilt_released"] = float(np.linalg.norm(d_raw) / np.linalg.norm(rel[ell + 1]))
            log(f"0a cos(rebuilt, -released) at layer {ell} = {res['cos_released']:.3f}; max over layers "
                f"{max(res['cos_released_by_layer']):.3f}")
        # their engagement / disengagement tests at ell: coefficient 1 on the raw vector, step tokens only
        I = ITEMS_D9()
        recs, _ = process_problems(m, I["calib0a"], CAP_TEST, self._path(m, "0a_chains"), log)
        et, dt = [], []
        for p in I["calib0a"]:
            for st in recs[p["id"]].get("sentences", []):
                if st["F_C"] == 1:
                    (dt if st["F"] == 1 else et).append((p, recs[p["id"]], st))
        rng = random.Random(0)
        et = rng.sample(et, min(N_0A, len(et)))
        dt = rng.sample(dt, min(N_0A, len(dt)))
        g = torch.Generator().manual_seed(9)
        Rv = torch.randn(4, m.d, generator=g, dtype=torch.float64)
        Rv = Rv / Rv.norm(dim=1, keepdim=True) * float(np.linalg.norm(d_raw))
        names = ["clean", "d"] + (["released"] if rel is not None else []) + [f"rand{i}" for i in range(4)]
        for test, lst, sgn in (("ET", et, 1.0), ("DT", dt, -1.0)):
            V = [torch.zeros(m.d, dtype=torch.float64), sgn * torch.tensor(d_raw)]
            if rel is not None:
                V.append(sgn * torch.tensor(-rel[ell + 1]))
            V = torch.stack(V + list(Rv)).float()
            flips = {nm: [] for nm in names}
            for p, r, st in lst:
                chunks = split_steps(r["output"])
                base = self.prefix(m, p["problem"])
                sp = perturb_text(chunks[st["j"]])[0]
                text = ext_text(base, chunks, st["j"], step=sp) + CUE
                c0 = len(ext_text(base, chunks, st["j"], step=""))
                ids, t1 = tok_index(m, text, c0 + len(sp) - 1, first=False)
                _, t0 = tok_index(m, text, c0, first=True)
                toks = gen_span(m, ids, t0, t1, ell, V, N_ANS)
                for nm, t in zip(names, toks):
                    same = compare_answers(early_answer(m.decode(t)), r["ystar"])
                    flips[nm].append(int(not same) if test == "ET" else int(same))
            fr = {nm: float(np.mean(v)) if v else None for nm, v in flips.items()}
            rand = [fr[f"rand{i}"] for i in range(4)] if lst else []
            res[test] = dict(n=len(lst), flip=fr, flip_rand_mean=float(np.mean(rand)) if rand else None,
                             adv_d_vs_rand=(fr["d"] - float(np.mean(rand))) if rand else None,
                             per_item=dict(d=flips["d"], clean=flips["clean"], rand=[flips[f"rand{i}"] for i in range(4)]))
            log(f"0a {test} (n={len(lst)}): flip rate clean {fr['clean']}, d {fr['d']}, "
                f"released {fr.get('released')}, random {res[test]['flip_rand_mean']}")
        return res

    def B_set(self, m, items):
        """Matched label set: within each chain, a true-thinking step (TTS > 0.9) paired with the
        nearest unused decorative step (TTS <= 0.005) whose length is within one token."""
        if not items:
            return []
        recs, _ = process_problems(m, items, CAP_TEST, self._path(m, "B_chains"), lambda *a: print(*a, flush=True))
        out = []
        for p in items:
            r = recs[p["id"]]
            st = r.get("sentences", [])
            hi = [s for s in st if s["T"] > T_HIGH]
            lo = [s for s in st if s["T"] <= T_LOW]
            used = set()
            chunks = split_steps(r["output"])
            base = self.prefix(m, p["problem"])
            for h in hi:
                cand = [s for s in lo if s["j"] not in used and abs(s["step_len"] - h["step_len"]) <= 1]
                if not cand or len(out) >= 2 * N_B:
                    continue
                lw = min(cand, key=lambda s: (abs(s["j"] - h["j"]), s["j"]))
                used.add(lw["j"])
                for lab, s in ((1, h), (0, lw)):
                    ids = m.ids(ext_text(base, chunks, s["j"]))
                    out.append(dict(id=f"{p['id']}#{h['j']}:{'high' if lab else 'low'}", text=chunks[s["j"]], label=lab,
                                    ids=ids, start=len(ids) - 1, T=s["T"]))
        print(f"Check B: {len(out) // 2} matched pairs from {len(items)} chains", flush=True)
        return out if len(out) // 2 >= 5 else []  # the text-only baseline needs 5 pair groups

    # ------------------------------------------------------------------ items
    def items(self, phase):
        I = ITEMS_D9()
        if phase == "pilot":
            pil, pd = I["pilot"], I["pilot_D"]
            return dict(A=pil, oracle=[x for x in pil if x["id"] not in set(pd)], D=pd, E=pd[:6], B=I["pilot_B"])
        return dict(A=I["A"], oracle=I["oracle"], D=I["D"], E=I["E"], B=I["B"])

    def prompt_ids(self, m, it, mode):
        """The unit's perturbed step in their intervention format, followed by the cue; the reading
        position (start of the push) is the step's last token."""
        u = unit_for(m.key, it["id"])
        base = self.prefix(m, it["problem"])
        text = f"{base}. {u['ctx']}. {u['s_pert']}" + CUE
        return tok_index(m, text, len(text) - len(CUE) - 1, first=False)

    def signs(self, it):
        return (1, -1)  # analysis: +d on units whose clean answer stays y*, -d on units whose clean answer follows s'

    def outcome(self, m, it, mode, toks, lp0, top1=None):
        u = unit_for(m.key, it["id"])
        ans = early_answer(m.decode(toks))
        yid = m.tok.encode(u["ystar"], add_special_tokens=False)
        return dict(F=0 if compare_answers(ans, u["ystar"]) else 1, ans=ans[:40],
                    p_first=float(lp0[yid[0]].exp()) if yid else None)
