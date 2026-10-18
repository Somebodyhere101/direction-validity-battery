"""Sampled generation for paper 6's long-generation target D6 (thinking-token counts), on top of p6lib.

- Sampling: temperature 0.6, top-p 0.95 (the authors' settings), as Gumbel-max inside the nucleus,
  which is an exact draw from the renormalised nucleus distribution.
- Noise: counter-based. The Gumbel vector a row uses at its k-th generated token is a fixed hash of
  (key, k, token id), where the key is (item, sample index). It does not depend on the batch or on
  when the row runs. Rows that share a key are common-random-number twins: a push that does not
  change the logits enough to flip a choice leaves the chain unchanged, and re-running a row
  reproduces its chain (up to bf16 batch effects).
- Stopping: a row stops at </think> (answer_max = 0) or answer_max tokens after it, at an end-of-turn
  token, or when it has generated `cap` thinking tokens without </think> (capped; T = cap).
- `stream_generate` (clean, Check 0a, Check D): continuous batching. Up to `slots` rows run at once;
  finished rows leave the batch and its KV cache, and pending rows are admitted in groups (prompt pass
  on their own, then merged into the cache, left-padded to a shared column layout; leading columns
  that no running row uses are trimmed). Memory is slots x (prompt + cap + answer) tokens at most.
  On an out-of-memory decoding step the cache is cut back, the rows with the fewest generated tokens
  (half of them) are put back in the queue to restart, and the number of slots is lowered.
- `sample_generate` (Check E): static batches in lockstep (row 0 is the clean twin that re-imposition
  reads), no compaction.
- Pushes use p6lib's hooks (Model.ops: "add" block -> [B, d], "start", "pos0", "reimp"). A row's push
  starts at its last prompt token ("last", the battery's push) or at position 0 ("all", the authors'
  every-position steering); every generated token is pushed.
"""

from __future__ import annotations

import collections
import os
import zlib

import torch
import torch.nn.functional as F
import transformers
import transformers.cache_utils

M32 = 0xFFFFFFFF


def kv_bytes_per_token(m):
    c = m.hf.config
    nkv = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return 2 * m.L * nkv * hd * 2  # keys and values, bf16


def weight_bytes(m):
    return sum(p.numel() * p.element_size() for p in m.hf.parameters())


def rows_for(m, max_len, overcommit=1.0, reserve_gb=2.0, cap_rows=96):
    """Rows whose KV cache at max_len (+ one growth chunk) tokens fits the capped allocator's budget
    (x overcommit). Cache edits (growth, row selection, merges) copy one layer at a time; that copy
    and some slack for the allocator are counted."""
    frac = float(os.environ.get("P6_MEM_FRACTION", "0.85"))
    total = torch.cuda.get_device_properties(0).total_memory * frac
    free = total - weight_bytes(m) - reserve_gb * 2 ** 30
    n = int(overcommit * free / (kv_bytes_per_token(m) * (max_len + CHUNK) * (1 + 3 / m.L)))
    return max(1, min(cap_rows, n))


def seed_of(key):
    return zlib.crc32(repr(key).encode())


def _mix32(x):
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32  # wraps in int64; the low 32 bits are exact
    return x ^ (x >> 16)


def gumbel_rows(seeds, steps, V):
    """Gumbel noise [B, V] that depends only on (seed, step, token id)."""
    s = _mix32((seeds.to(torch.int64) & M32) ^ _mix32((steps.to(torch.int64) * 0x9E3779B9 + 0x632BE5AB) & M32))
    idx = torch.arange(V, device=seeds.device, dtype=torch.int64)
    h = _mix32((s[:, None] + idx[None, :] * 0x85EBCA6B) & M32)
    h = _mix32(h ^ 0x27D4EB2F)
    u = (h.to(torch.float64) + 0.5) / 4294967296.0
    return (-torch.log(-torch.log(u))).float()


def nucleus_gumbel(logits, G, temp, top_p):
    z = logits.float() / temp
    p = torch.softmax(z, -1)
    sp, si = p.sort(-1, descending=True)
    rm = (sp.cumsum(-1) - sp) >= top_p  # outside the nucleus (the first token is always kept)
    z = z.masked_fill(torch.zeros_like(rm).scatter(-1, si, rm), float("-inf"))
    return (z + G).argmax(-1)


# ============================================================ KV cache and decode attention
CHUNK = 256


class GrowLayer(transformers.cache_utils.DynamicLayer):
    """DynamicLayer whose keys and values live in a buffer [B, H, C, D] that grows by CHUNK columns
    at a time; new tokens are written in place (DynamicLayer concatenates, i.e. copies the whole
    layer, at every token). `off` marks the first live column (leading columns no row uses are
    dropped by moving it), `n` the end. keys/values are views [B, H, off:n, D]."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.kbuf = self.vbuf = None
        self.off = self.n = 0

    def _views(self):
        self.keys, self.values = self.kbuf[:, :, self.off:self.n], self.vbuf[:, :, self.off:self.n]

    def set_buffers(self, kbuf, vbuf, off, n):
        self.kbuf, self.vbuf, self.off, self.n = kbuf, vbuf, off, n
        self.dtype, self.device, self.is_initialized = kbuf.dtype, kbuf.device, True
        self._views()

    def update(self, key_states, value_states, *args, **kwargs):
        B, H, T, D = key_states.shape
        if self.kbuf is None:
            self.set_buffers(key_states.new_zeros(B, H, T + CHUNK, D), value_states.new_zeros(B, H, T + CHUNK, D), 0, 0)
        if self.n + T > self.kbuf.shape[2]:  # grow (and compact to off = 0)
            w = self.n - self.off
            kb, vb = key_states.new_zeros(B, H, w + T + CHUNK, D), value_states.new_zeros(B, H, w + T + CHUNK, D)
            kb[:, :, :w], vb[:, :, :w] = self.kbuf[:, :, self.off:self.n], self.vbuf[:, :, self.off:self.n]
            self.kbuf, self.vbuf, self.off, self.n = kb, vb, 0, w
        self.kbuf[:, :, self.n:self.n + T] = key_states
        self.vbuf[:, :, self.n:self.n + T] = value_states
        self.n += T
        self._views()
        return self.keys, self.values

    def get_seq_length(self):
        return 0 if self.kbuf is None else self.n - self.off

    def batch_select_indices(self, indices):
        if self.kbuf is not None:
            self.kbuf, self.vbuf = self.kbuf[indices], self.vbuf[indices]
            self._views()

    def crop(self, width):  # absolute live width, used to undo a failed step
        if self.kbuf is not None and self.n - self.off > width:
            self.n = self.off + width
            self._views()

    def trim_left(self, lead):
        self.off += lead
        self._views()


def new_cache():
    c = transformers.DynamicCache()
    c.layer_class_to_replicate = GrowLayer
    return c


def p6_decode_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """Single-token decoding with grouped-query heads as one batched matmul per layer, without
    expanding keys and values to every query head (transformers' SDPA path does that whenever a
    padding mask is present). Prompt passes (more than one query token) use SDPA unchanged.
    Scores and softmax in float32."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if query.shape[2] != 1:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                      is_causal=is_causal, **kwargs)
    B, Hq, _, D = query.shape
    Hkv, W = key.shape[1], key.shape[2]
    rep = Hq // Hkv
    scaling = scaling if scaling is not None else D ** -0.5
    q = (query.reshape(B * Hkv, rep, D).float() * scaling).to(key.dtype)
    k = key.reshape(B * Hkv, W, D)
    v = value.reshape(B * Hkv, W, D)
    try:
        s = torch.bmm(q, k.transpose(1, 2), out_dtype=torch.float32)
    except TypeError:
        s = torch.bmm(q, k.transpose(1, 2)).float()
    s = s.view(B, Hkv, rep, W)
    if attention_mask is not None:
        am = attention_mask[:, :, -1:, -W:]
        s = s.masked_fill(~am, float("-inf")) if am.dtype == torch.bool else s + am.float()
    p = torch.softmax(s, -1).to(value.dtype).view(B * Hkv, rep, W)
    o = torch.bmm(p, v).view(B, Hq, 1, D)
    return o.transpose(1, 2).contiguous(), None


def _set_attn(m, name):
    m.hf.config._attn_implementation = name
    for mod in m.hf.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = name


def install_fast_decode(m):
    """Register p6_decode_attention (with SDPA's mask format) and switch the model to it."""
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    AttentionInterface.register("p6decode", p6_decode_attention)
    AttentionMaskInterface.register("p6decode", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    _set_attn(m, "p6decode")


def use_sdpa(m):
    _set_attn(m, "sdpa")


def _cut_cache(cache, width):
    for lay in cache.layers:
        lay.crop(width)


def _result(tok_row, n, tl, ended, cap):
    t = [int(v) for v in tok_row[:n]]
    if tl >= 0:
        T, capped = tl, 0
    elif ended:
        T, capped = n - 1, 0  # end of turn without </think>
    else:
        T, capped = cap, 1
    return dict(T=int(T), capped=capped, think_end=int(tl >= 0), eos_no_think_end=int(tl < 0 and ended), toks=t)


# ============================================================ continuous batching
@torch.no_grad()
def stream_generate(m, rows, cap, answer_max=0, slots=None, temp=0.6, top_p=0.95, on_done=None, log=None, admit_frac=0.25):
    """rows: list of dicts with "ids" (prompt token list), "key" (noise key), optional "add"
    ({block: tensor [d]}) and "start" ("last" or "all"). Returns one result dict per row, in order;
    on_done(i, result) is called as each row finishes."""
    dev = m.dev
    N = len(rows)
    think_end = m.tok.convert_tokens_to_ids("</think>")
    eos = torch.tensor(sorted(m.eos), device=dev)
    V = m.hf.lm_head.weight.shape[0]
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    if slots is None:
        slots = rows_for(m, max(len(r["ids"]) for r in rows) + cap + answer_max)
    out_buf = torch.full((N, cap + answer_max), -1, dtype=torch.int32, device=dev)
    results = [None] * N
    pending = collections.deque(range(N))
    S = None  # batch state: rid, x, pos, nst, tl, ended, seeds, am, cache, add
    n_evict = 0
    test_oom = int(os.environ.get("P6_TEST_OOM_AT", "0"))  # self-test only: fake an out-of-memory error at this decoding step
    n_dec = 0

    def select(S, keep):
        for k in ("rid", "x", "pos", "nst", "tl", "ended", "seeds", "am"):
            S[k] = S[k][keep]
        S["cache"].batch_select_indices(keep)
        S["add"] = {l: v[keep] for l, v in S["add"].items()}

    def record(S, nxt):
        """Append nxt to every row of S; returns the mask of rows that finished."""
        nst = S["nst"]
        out_buf[S["rid"], nst] = nxt.to(torch.int32)
        te = (nxt == think_end) & (S["tl"] < 0)
        S["tl"] = torch.where(te, nst, S["tl"])
        ise = torch.isin(nxt, eos)
        S["ended"] = S["ended"] | ise
        fin = ise | ((S["tl"] < 0) & (nst + 1 >= cap)) | ((S["tl"] >= 0) & (nst - S["tl"] >= answer_max))
        S["nst"] = nst + 1
        return fin

    def retire(S, fin):
        idx = torch.where(fin)[0]
        if len(idx) == 0:
            return
        rid, n, tl, en = (S[k][idx].tolist() for k in ("rid", "nst", "tl", "ended"))
        toks = out_buf[S["rid"][idx]].cpu()
        for j, r in enumerate(rid):
            results[r] = _result(toks[j].tolist(), n[j], tl[j], en[j], cap)
            if on_done:
                on_done(r, results[r])

    def admit(group, mode):
        ids = [rows[i]["ids"] for i in group]
        Lg = max(len(r) for r in ids)
        x = torch.tensor([[pad] * (Lg - len(r)) + r for r in ids], device=dev)
        am = torch.tensor([[0] * (Lg - len(r)) + [1] * len(r) for r in ids], device=dev)
        pos = (am.cumsum(1) - 1).clamp_min(0)
        layers = sorted({l for i in group for l in (rows[i].get("add") or {})})
        add = {}
        for l in layers:
            add[l] = torch.stack([(rows[i].get("add") or {}).get(l, torch.zeros(m.d)).float() for i in group]).to(dev, torch.bfloat16)
        cache = new_cache()
        m.ops = dict(start=(Lg - 1 if mode == "last" else 0), add=add, reimp=None, pos0=0) if add else None
        try:
            o = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        finally:
            m.ops = None
        G = dict(rid=torch.tensor(group, device=dev), x=None, pos=torch.tensor([[len(r)] for r in ids], device=dev),
                 nst=torch.zeros(len(group), dtype=torch.long, device=dev), tl=torch.full((len(group),), -1, dtype=torch.long, device=dev),
                 ended=torch.zeros(len(group), dtype=torch.bool, device=dev),
                 seeds=torch.tensor([seed_of(rows[i]["key"]) for i in group], dtype=torch.long, device=dev), am=am, cache=cache, add=add)
        nxt = nucleus_gumbel(o.logits[:, -1], gumbel_rows(G["seeds"], G["nst"], V), temp, top_p)
        del o
        G["x"] = nxt[:, None]
        fin = record(G, nxt)
        retire(G, fin)
        keep = torch.where(~fin)[0]
        if len(keep) < len(group):
            if len(keep) == 0:
                return None
            select(G, keep)
        return G

    def merge(S, G):
        """Merge G's rows into S. On out-of-memory the layers already merged are cut back to S's own
        rows and columns, and the error is raised again (the caller re-queues G's rows)."""
        if S is None:
            return G
        W, Wg = S["am"].shape[1], G["am"].shape[1]
        Wn = max(W, Wg)
        B0, Bg = len(S["rid"]), len(G["rid"])
        done = []
        try:
            for lm, lg in zip(S["cache"].layers, G["cache"].layers):  # one layer at a time: fresh right-aligned buffers
                H, D = lm.kbuf.shape[1], lm.kbuf.shape[3]
                kb = lm.kbuf.new_zeros(B0 + Bg, H, Wn + CHUNK, D)
                vb = lm.vbuf.new_zeros(B0 + Bg, H, Wn + CHUNK, D)
                kb[:B0, :, Wn - W:Wn], vb[:B0, :, Wn - W:Wn] = lm.keys, lm.values
                kb[B0:, :, Wn - Wg:Wn], vb[B0:, :, Wn - Wg:Wn] = lg.keys, lg.values
                lg.set_buffers(lg.kbuf[:0], lg.vbuf[:0], 0, 0)
                lm.set_buffers(kb, vb, 0, Wn)
                done.append(lm)
        except torch.OutOfMemoryError:
            for lm in done:
                lm.set_buffers(lm.kbuf[:B0], lm.vbuf[:B0], Wn - W, Wn)
            raise
        S["am"] = torch.cat([F.pad(S["am"], (Wn - W, 0)), F.pad(G["am"], (Wn - Wg, 0))], 0)
        for k in ("rid", "x", "pos", "nst", "tl", "ended", "seeds"):
            S[k] = torch.cat([S[k], G[k]], 0)
        B, Bg = len(S["rid"]) - len(G["rid"]), len(G["rid"])
        for l in sorted(set(S["add"]) | set(G["add"])):
            a = S["add"].get(l, torch.zeros(B, m.d, dtype=torch.bfloat16, device=dev))
            b = G["add"].get(l, torch.zeros(Bg, m.d, dtype=torch.bfloat16, device=dev))
            S["add"][l] = torch.cat([a, b], 0)
        return S

    while pending or (S is not None and len(S["rid"])):
        B = 0 if S is None else len(S["rid"])
        free = slots - B
        if pending and free > 0 and (B == 0 or free >= max(1, int(admit_frac * slots))):
            mode = rows[pending[0]].get("start", "last")
            group = []
            while pending and len(group) < free and rows[pending[0]].get("start", "last") == mode:
                group.append(pending.popleft())
            try:
                G = admit(group, mode)
                if G is not None:
                    S = merge(S, G) if (S is not None and B) else G
            except torch.OutOfMemoryError:
                G = None
                torch.cuda.empty_cache()
                for r in group:  # chains that already finished at their first token keep their result
                    if results[r] is None:
                        out_buf[r].fill_(-1)
                pending.extendleft(reversed([r for r in group if results[r] is None]))
                if B == 0 and len(group) == 1:
                    raise
                slots = max(1, B if B else len(group) // 2)
                n_evict += 1
                if log:
                    log(f"OOM while admitting {len(group)} rows; slots now {slots}")
                continue
            continue
        if S is None or B == 0:
            continue
        # one decoding step for every running row
        W = S["am"].shape[1]
        am_full = torch.cat([S["am"], torch.ones_like(S["am"][:, :1])], 1)
        m.ops = dict(start=0, add=S["add"], reimp=None, pos0=0) if S["add"] else None
        n_dec += 1
        try:
            if test_oom and n_dec == test_oom:
                raise torch.OutOfMemoryError("self-test")
            o = m.hf(input_ids=S["x"], attention_mask=am_full, position_ids=S["pos"], past_key_values=S["cache"], use_cache=True,
                     logits_to_keep=1)
            nxt = nucleus_gumbel(o.logits[:, -1], gumbel_rows(S["seeds"], S["nst"], V), temp, top_p)
            del o
        except torch.OutOfMemoryError:
            o = None
            m.ops = None
            _cut_cache(S["cache"], W)
            if B == 1:
                raise
            nk = B - max(1, B // 4)  # re-queue a quarter: the rows with the least work done
            order = torch.argsort(S["nst"], descending=True)
            keep, drop = order[:nk], order[nk:]
            for r in reversed(S["rid"][drop].tolist()):
                out_buf[r].fill_(-1)
                pending.appendleft(r)
            select(S, keep)
            torch.cuda.empty_cache()
            slots = max(1, nk)
            n_evict += 1
            if log:
                log(f"OOM while decoding: {B - nk} of {B} rows re-queued; slots now {slots}")
            continue
        finally:
            m.ops = None
        S["am"] = am_full
        S["x"] = nxt[:, None]
        S["pos"] = S["pos"] + 1
        fin = record(S, nxt)
        if bool(fin.any()):
            retire(S, fin)
            keep = torch.where(~fin)[0]
            if len(keep) == 0:
                S = None
                continue
            select(S, keep)
        # trim leading columns that no running row uses
        used = S["am"].any(0)
        lead = int(torch.argmax(used.to(torch.int8)).item()) if bool(used.any()) else 0
        if lead >= 64:
            S["am"] = S["am"][:, lead:]
            for lay in S["cache"].layers:
                lay.trim_left(lead)
    torch.cuda.empty_cache()
    return results


# ============================================================ static batches (Check E)
@torch.no_grad()
def _generate(m, rows_ids, keys, add, start, cap, answer_max, temp, top_p, reimp):
    """Lockstep generation of one batch, no compaction (re-imposition reads the clean twin in row 0
    at every step); finished rows are fed end-of-turn tokens, as in Model.generate."""
    dev = m.dev
    B = len(rows_ids)
    think_end = m.tok.convert_tokens_to_ids("</think>")
    eos = torch.tensor(sorted(m.eos), device=dev)
    V = m.hf.lm_head.weight.shape[0]
    seeds = torch.tensor([seed_of(k) for k in keys], dtype=torch.long, device=dev)
    Lp = max(len(r) for r in rows_ids)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    if add is not None and start != 0:
        assert all(len(r) == Lp for r in rows_ids), "pushes from the reading position need equal-length prompts"
    x = torch.tensor([[pad] * (Lp - len(r)) + r for r in rows_ids], device=dev)
    am = torch.tensor([[0] * (Lp - len(r)) + [1] * len(r) for r in rows_ids], device=dev)
    pos = (am.cumsum(1) - 1).clamp_min(0)
    out = torch.full((B, cap + answer_max), -1, dtype=torch.long, device=dev)
    think_len = torch.full((B,), -1, dtype=torch.long, device=dev)
    n_gen = torch.zeros(B, dtype=torch.long, device=dev)
    ended = torch.zeros(B, dtype=torch.bool, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    addd = {l: v.to(dev, torch.bfloat16) for l, v in add.items()} if add else {}
    rp = dict(reimp) if reimp else None
    m.ops = dict(start=start if start is not None else 0, add=addd, reimp=rp, pos0=0) if (addd or rp) else None
    cache = new_cache()
    try:
        step = 0
        while True:
            if m.ops is not None:
                m.ops["pos0"] = 0 if step == 0 else Lp + step - 1
            o = m.hf(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            nxt = nucleus_gumbel(o.logits[:, -1], gumbel_rows(seeds, n_gen, V), temp, top_p)
            del o
            act = ~done
            nxt = torch.where(act, nxt, eos[0])
            ia = torch.where(act)[0]
            na = nxt[act]
            out[ia, step] = na
            is_te = (na == think_end) & (think_len[ia] < 0)
            think_len[ia[is_te]] = step
            is_eos = torch.isin(na, eos)
            ended[ia[is_eos]] = True
            tl = think_len[ia]
            fin_a = is_eos | ((tl < 0) & (step + 1 >= cap)) | ((tl >= 0) & (step - tl >= answer_max))
            n_gen[ia] += 1
            done[ia[fin_a]] = True
            step += 1
            x = nxt[:, None]
            am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
            pos = pos[:, -1:] + 1
            if bool(done.all()):
                break
    finally:
        m.ops = None
        del cache
        torch.cuda.empty_cache()
    tl, ng, ee, toks = think_len.cpu().tolist(), n_gen.cpu().tolist(), ended.cpu().tolist(), out.cpu()
    return [_result(toks[b].tolist(), ng[b], tl[b], ee[b], cap) for b in range(B)]


def sample_generate(m, rows_ids, keys, add=None, start=None, cap=4096, answer_max=0, temp=0.6, top_p=0.95, reimp=None, log=None):
    """One lockstep batch (Check E). Raises on out-of-memory; the caller sizes the batch."""
    return _generate(m, rows_ids, keys, add, start, cap, answer_max, temp, top_p, reimp)


@torch.no_grad()
def tf_kl(m, ids, push_start, kl_start, add, cont, rows_per=None):
    """p6lib.Model.tf_kl with separate push and KL start positions (the authors' steering pushes from
    position 0; KL is always summed from the reading position). Row 0 of `add` must be zeros (clean).
    Rows are processed in chunks that each re-run the clean row."""
    full = ids + cont
    B = next(iter(add.values())).shape[0]
    T = len(full)
    if rows_per is None:
        rows_per = max(2, int(12 * 4096 / max(T, 1)))
    V = m.hf.lm_head.weight.shape[0]
    last = len(full) - 1 if cont else len(full)
    kl_all = [0.0] * B
    kl_first = [0.0] * B
    body = list(range(1, B))
    for c0 in range(0, max(1, len(body)), rows_per - 1):
        rows = [0] + body[c0:c0 + rows_per - 1]
        sub = {l: v[rows].to(m.dev, torch.bfloat16) for l, v in add.items()}
        x = torch.tensor([full] * len(rows), device=m.dev)
        m.ops = dict(start=push_start, add=sub, reimp=None, pos0=0)
        try:
            h = m.hf.model(input_ids=x, use_cache=False).last_hidden_state
        finally:
            m.ops = None
        kls = []
        step = max(1, int(0.5e9 // (4 * len(rows) * V)))
        for p0 in range(kl_start, last, step):
            lp = torch.log_softmax(m.hf.lm_head(h[:, p0:min(last, p0 + step)]).float(), -1)
            kls.append((lp[0:1].exp() * (lp[0:1] - lp)).sum(-1).clamp_min(0))
            del lp
        del h
        K = torch.cat(kls, 1).cpu()
        torch.cuda.empty_cache()
        for b, r in enumerate(rows):
            if r == 0 and c0 > 0:
                continue
            kl_all[r] = float(K[b].sum())
            kl_first[r] = float(K[b, len(ids) - 1 - kl_start])
    return dict(kl_all=kl_all, kl_first=kl_first)
