"""Paper 6 battery engine: a generic model wrapper with per-row interventions on the residual
stream, greedy batched generation over conditions that share one prompt, teacher-forced KL, the
logit lens, and the control directions of PREREGISTRATION.md §3 (N_len, N_pc1, N_perm, oracle,
Gaussian random).

Conventions
- "layer l" always means the OUTPUT of block l (0-indexed), as in paper 4 and in the CR detector
  and Marks & Tegmark's code. Arditi et al.'s "resid_pre of layer l" is the output of block l-1.
- A push adds a vector to the output of one block at every position >= `start` (the reading
  position), including generated tokens (§2).
- Rows of a batch share one prompt (no padding); row 0 is always the clean run.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "q3-1.7b": "Qwen/Qwen3-1.7B",
    "q25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "q3-8b": "Qwen/Qwen3-8B",
    "olmo-7b": "allenai/OLMo-2-1124-7B-Instruct",
    "r1-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "r1-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "q3-14b": "Qwen/Qwen3-14B",
}


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


class Model:
    def __init__(self, key: str, device: str = "cuda", thinking: bool = False, gpu_layers: int | None = None):
        self.key, self.name, self.dev, self.thinking = key, MODELS[key], device, thinking
        if os.environ.get("P6_MEM_FRACTION") and device.startswith("cuda"):
            # keep the caching allocator inside VRAM: on Windows (WDDM) an over-full card spills to
            # system memory silently and runs ~50x slower, which long batched generations triggered
            torch.cuda.set_per_process_memory_fraction(float(os.environ["P6_MEM_FRACTION"]))
        self.tok = transformers.AutoTokenizer.from_pretrained(self.name)
        if gpu_layers is None:
            self.hf = transformers.AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.bfloat16).to(device).eval()
        else:
            sys.path.insert(0, os.path.join(HERE, "..", "..", "effort_jspace"))
            from core import _stream_from_cpu
            self.hf = transformers.AutoModelForCausalLM.from_pretrained(self.name, dtype=torch.bfloat16).eval()
            mm = self.hf.model
            for mod in (mm.embed_tokens, mm.norm, mm.rotary_emb, self.hf.lm_head):
                mod.to(device)
            for i, layer in enumerate(mm.layers):
                if i < gpu_layers:
                    layer.to(device)
                else:
                    _stream_from_cpu(layer, device)
        for p in self.hf.parameters():
            p.requires_grad_(False)
        self.layers = self.hf.model.layers
        self.L = len(self.layers)
        self.d = self.hf.config.hidden_size
        self.norm = self.hf.model.norm
        self.ops = None
        self.capture = None
        eos = self.hf.generation_config.eos_token_id
        eos = eos if isinstance(eos, (list, tuple)) else [eos]
        self.eos = set(int(e) for e in eos if e is not None) | {self.tok.eos_token_id}
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
            i = self.tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i != self.tok.unk_token_id and i is not None:
                self.eos.add(i)
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._hook(i))
        self._abl_handles = []

    # ------------------------------------------------------------------ hooks
    def _hook(self, i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            o = self.ops
            if o is not None:
                T = h.shape[1]
                if o.get("pos0") is None:  # absolute position of h[:, 0]
                    pos = torch.arange(T, device=h.device)
                else:
                    pos = o["pos0"] + torch.arange(T, device=h.device)
                m = (pos >= o["start"]).to(h.dtype)[None, :, None]
                add = o.get("add", {}).get(i)
                if add is not None:
                    h = h + m * add[:, None, :]
                rp = o.get("reimp")
                if rp is not None:
                    sel = (rp["layer"] == i)
                    if bool(sel.any()):
                        u = rp["dir"]
                        p = (h.float() @ u)  # [B, T]
                        delta = (p[0:1] - p) * sel[:, None].float()
                        h = h + (m[..., 0] * delta.to(h.dtype))[..., None] * u.to(h.dtype)[None, None, :]
            if self.capture is not None and i in self.capture:
                self.capture[i] = h.detach()
            if isinstance(out, tuple):
                return (h,) + tuple(out[1:])
            return h
        return f

    def set_ablation(self, direction):
        """Arditi et al.'s directional ablation: remove the component along `direction` from the
        embeddings and from every attention and MLP write (for OLMo-2, from the outputs of the
        post-norms that write to the residual stream). None removes the hooks."""
        for h in self._abl_handles:
            h.remove()
        self._abl_handles = []
        if direction is None:
            return
        u = torch.tensor(unit(direction), dtype=torch.float32, device=self.dev)

        def proj_out(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            uu = u.to(h.dtype)
            h = h - (h @ uu)[..., None] * uu
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        self._abl_handles.append(self.hf.model.embed_tokens.register_forward_hook(proj_out))
        for layer in self.layers:
            if hasattr(layer, "post_feedforward_layernorm"):  # OLMo-2: post-norm writes
                mods = [layer.post_attention_layernorm, layer.post_feedforward_layernorm]
            else:
                mods = [layer.self_attn, layer.mlp]
            for mod in mods:
                self._abl_handles.append(mod.register_forward_hook(proj_out))

    # ---------------------------------------------------------------- prompts
    def chat(self, user: str, system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {}
        if "Qwen3" in self.name:
            kw["enable_thinking"] = self.thinking
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=not text.startswith(self.tok.bos_token or "\x00"))["input_ids"]

    def raw_ids(self, text: str) -> list[int]:
        """Plain text with the tokenizer's default special tokens (BOS where the model uses one)."""
        return self.tok(text)["input_ids"]

    def sub_end(self, ids: list[int], prefix_text: str) -> int:
        """Index of the last token of `prefix_text` within `ids` (the chat text must start with it)."""
        pre = self.ids(prefix_text)
        n = len(pre)
        # tokenisation at the boundary can merge; back off until the prefix matches
        while n > 0 and ids[:n] != pre[:n]:
            n -= 1
        return n - 1

    # ------------------------------------------------------------ activations
    @torch.no_grad()
    def acts(self, ids: list[int], layers, positions) -> dict:
        """Hidden states (block outputs) at the given absolute positions, batch of one prompt."""
        self.capture = {l: None for l in layers}
        try:
            self.hf.model(input_ids=torch.tensor([ids], device=self.dev))
            return {l: self.capture[l][0, positions].float().cpu().numpy() for l in layers}
        finally:
            self.capture = None

    @torch.no_grad()
    def last_logits(self, batch_ids: list[list[int]], ops=None) -> torch.Tensor:
        """Last-position logits for a left-padded batch (used only for scoring functions)."""
        n = max(len(x) for x in batch_ids)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        x = torch.tensor([[pad] * (n - len(r)) + r for r in batch_ids], device=self.dev)
        am = torch.tensor([[0] * (n - len(r)) + [1] * len(r) for r in batch_ids], device=self.dev)
        pos = (am.cumsum(1) - 1).clamp_min(0)
        if ops is not None:
            ops = dict(ops, add={l: v.to(self.dev, torch.bfloat16) for l, v in ops.get("add", {}).items()})
        self.ops = ops
        try:
            return self.hf(input_ids=x, attention_mask=am, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
        finally:
            self.ops = None

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, ids: list[int], start: int, add: dict, n_new: int, reimp=None, lens=False,
                 lens_layers=None, stop_all_eos=True) -> dict:
        """Greedy continuation of one prompt for B condition rows. `add` maps layer -> [B, d]
        tensor (row 0 must be zeros = clean). Returns tokens [B, n] (eos-padded after stop), the
        first-step log-probs, and with lens=True the logit-lens top-1 at every layer and step."""
        B = next(iter(add.values())).shape[0]
        addd = {l: v.to(self.dev, torch.bfloat16) for l, v in add.items()}
        x = torch.tensor([ids] * B, device=self.dev)
        T = len(ids)
        cache = transformers.DynamicCache()
        out_tok, top1s, lp0 = [], [], None
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        eos_t = torch.tensor(sorted(self.eos), device=self.dev)
        self.ops = dict(start=start, add=addd, reimp=reimp, pos0=0)
        try:
            for step in range(n_new):
                if lens:
                    self.capture = {l: None for l in (lens_layers or range(self.L))}
                p0 = 0 if step == 0 else T - 1
                self.ops["pos0"] = p0
                pos = torch.arange(p0, p0 + x.shape[1], device=self.dev)[None].expand(B, -1)
                o = self.hf(input_ids=x, past_key_values=cache, use_cache=True, position_ids=pos, logits_to_keep=1)
                logits = o.logits[:, -1].float()
                nxt = logits.argmax(-1)
                if step == 0:
                    lp0 = torch.log_softmax(logits, -1)
                if lens:
                    top1s.append(self._lens_top1())
                nxt = torch.where(done, eos_t[0], nxt)
                out_tok.append(nxt)
                done |= torch.isin(nxt, eos_t)
                x = nxt[:, None]
                T += 1
                if stop_all_eos and bool(done.all()):
                    break
        finally:
            self.ops, self.capture = None, None
        del cache
        res = dict(tokens=torch.stack(out_tok, 1).cpu(), lp0=lp0)
        torch.cuda.empty_cache()
        if lens:
            res["top1"] = torch.stack(top1s, 1).cpu()  # [B, steps, L]
        return res

    def _lens_top1(self):
        t1 = []
        for l in sorted(self.capture):
            lg = self.hf.lm_head(self.norm(self.capture[l][:, -1:]))[:, 0]
            t1.append(lg.argmax(-1))
        return torch.stack(t1, -1)

    @torch.no_grad()
    def tf_kl(self, ids: list[int], start: int, add: dict, cont: list[int], reimp=None,
              extra_cols=None) -> dict:
        """Teacher-forced on ids + cont (the clean row's own continuation): KL(row 0 || row b)
        summed over every position from `start` to the end (kl_all), at the first position after
        the prompt (kl_first), and over `extra_cols` (absolute positions) if given."""
        full = ids + cont
        B = next(iter(add.values())).shape[0]
        x = torch.tensor([full] * B, device=self.dev)
        self.ops = dict(start=start, add={l: v.to(self.dev, torch.bfloat16) for l, v in add.items()},
                        reimp=reimp, pos0=0)
        try:
            h = self.hf.model(input_ids=x, use_cache=False).last_hidden_state
        finally:
            self.ops = None
        kls = []
        last = len(full) - 1 if cont else len(full)
        V = self.hf.lm_head.weight.shape[0]
        step = max(1, int(0.5e9 // (4 * B * V)))  # positions per chunk (~0.5 GB of float32 log-probs)
        for c0 in range(start, last, step):
            lp = torch.log_softmax(self.hf.lm_head(h[:, c0:min(last, c0 + step)]).float(), -1)  # [B, c, V]
            kls.append((lp[0:1].exp() * (lp[0:1] - lp)).sum(-1).clamp_min(0))
            del lp
        del h
        K = torch.cat(kls, 1).cpu()  # [B, positions from start]
        torch.cuda.empty_cache()
        out = dict(kl_all=K.sum(1).tolist(), kl_first=K[:, len(ids) - 1 - start].tolist())
        if extra_cols:
            out["kl_extra"] = K[:, [c - start for c in extra_cols]].sum(1).tolist()
        return out

    # ------------------------------------------------------------- utilities
    def decode(self, toks) -> str:
        toks = [int(t) for t in toks]
        cut = len(toks)
        for i, t in enumerate(toks):
            if t in self.eos:
                cut = i
                break
        return self.tok.decode(toks[:cut], skip_special_tokens=True)

    def trunc(self, toks) -> list[int]:
        toks = [int(t) for t in toks]
        for i, t in enumerate(toks):
            if t in self.eos:
                return toks[:i]
        return toks

    def randomize_weights(self, seed: int = 0):
        """Check 0b: re-initialise every weight with the architecture's own initialiser."""
        torch.manual_seed(seed)
        for m in self.hf.modules():
            self.hf._init_weights(m)
        for mod in self.hf.modules():
            if hasattr(mod, "weight") and "Norm" in type(mod).__name__ and mod.weight is not None:
                torch.nn.init.ones_(mod.weight)


# ---------------------------------------------------------------- directions
def mean_diff(H, y):
    H, y = np.asarray(H, np.float64), np.asarray(y)
    return H[y == 1].mean(0) - H[y == 0].mean(0)


def n_len(H, y, lengths):
    """Within-class long-minus-short direction (§3): median split by token length per class."""
    H, y, lengths = np.asarray(H, np.float64), np.asarray(y), np.asarray(lengths)
    parts = []
    for c in (0, 1):
        idx = np.where(y == c)[0]
        order = idx[np.argsort(lengths[idx], kind="stable")]
        h = len(order) // 2
        parts.append(H[order[h:]].mean(0) - H[order[:h]].mean(0))
    return unit(np.mean(parts, 0))


def n_pc1(H, y, d):
    H, y = np.asarray(H, np.float64), np.asarray(y)
    C = H.copy()
    for c in (0, 1):
        C[y == c] -= C[y == c].mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    v = Vt[0]
    return unit(v if v @ d >= 0 else -v)


def n_perm(H, y, n=20, seed=6):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        yp = rng.permutation(np.asarray(y))
        out.append(unit(mean_diff(H, yp)))
    return np.array(out)


def oracle(H, T, binary=False):
    """Difference of means between the top and bottom terciles of T (class means if binary)."""
    H, T = np.asarray(H, np.float64), np.asarray(T, np.float64)
    if binary:
        return unit(H[T == 1].mean(0) - H[T == 0].mean(0))
    lo, hi = np.quantile(T, [1 / 3, 2 / 3])
    return unit(H[T >= hi].mean(0) - H[T <= lo].mean(0))


def gaussian(R, d, seed=1234):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(R, d, generator=g, dtype=torch.float64)
    return (V / V.norm(dim=1, keepdim=True)).numpy()


def auc(scores, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, scores))


def twofold_auc(H, y, seed=0):
    """Held-out AUC of the mean-difference direction: build on one half, score the other, both
    ways, averaged (Check 0b's 'held-out calibration-style prompts')."""
    H, y = np.asarray(H, np.float64), np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    out = []
    for tr, te in ((a, b), (b, a)):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        dd = mean_diff(H[tr], y[tr])
        out.append(auc(H[te] @ dd, y[te]))
    return float(np.mean(out)) if out else None


def settle_top1(top1_row) -> int:
    """Smallest layer l such that the lens top-1 equals the final top-1 at every layer >= l."""
    L = top1_row.shape[0]
    final = top1_row[-1]
    l = L - 1
    while l > 0 and top1_row[l - 1] == final:
        l -= 1
    return l


# ------------------------------------------------------------- condition grid
def build_conditions(d_unit, g, signs, ctrl: dict, perm, gauss, cfg, L, ell, e_items=False):
    """Rows for one item (Check C/D, and E when e_items). Every row: (meta, layer, vector).
    signs: the push signs applicable to this item, e.g. (+1,) or (+1, -1)."""
    rows = [(dict(kind="clean"), ell, np.zeros_like(d_unit))]
    for s in signs:
        for k in cfg["k"]:
            rows.append((dict(kind="d", k=s * k), ell, s * k * g * d_unit))
            for name, v in ctrl.items():
                rows.append((dict(kind=name, k=s * k), ell, s * k * g * v))
        for k in cfg.get("k_d_extra", []):  # the authors' published strength (secondary, §5 Check D)
            rows.append((dict(kind="d_auth", k=s * k), ell, s * k * g * d_unit))
    for j in range(cfg["n_perm_push"]):
        for c in cfg["c"]:
            rows.append((dict(kind="perm", j=j, c=c), ell, c * g * perm[j]))
    for j in range(cfg["R"]):
        for c in cfg["c"]:
            rows.append((dict(kind="rand", j=j, c=c), ell, c * g * gauss[j]))
    if e_items:
        later = list(range(ell + 1, L))
        if L > 32:
            later = later[::2]
        for s in signs:
            for lp in later:  # E1: re-impose the clean projection at block lp
                rows.append((dict(kind="E1", k=s, lp=lp), ell, s * g * d_unit))
            for c in cfg["e2_c"]:  # E2: last-block push of size c * g (KL matched later)
                rows.append((dict(kind="E2", k=s, c=c), L - 1, s * c * g * d_unit))
    return rows


def rows_to_ops(rows, d, dev="cpu"):
    """Pack rows into the per-layer `add` dict and the re-imposition spec."""
    layers = sorted({r[1] for r in rows})
    B = len(rows)
    add = {l: torch.zeros(B, d, dtype=torch.float32) for l in layers}
    rl = torch.full((B,), -1, dtype=torch.long)
    for b, (meta, l, v) in enumerate(rows):
        add[l][b] = torch.tensor(v, dtype=torch.float32)
        if meta["kind"] == "E1":
            rl[b] = meta["lp"]
    return add, rl
