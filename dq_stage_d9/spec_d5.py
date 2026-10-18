"""D5 backtracking (Ward et al. 2025; PREREGISTRATION.md §3) for paper 6.

Direction: Ward et al.'s recipe rebuilt in each panel model from their annotated chains
(new_annotated_chains/all_annotated_chains.json, 1,000 chains): block output at their layer
(10 of 32 in Llama-8B; the same relative depth elsewhere), mean over the six tokens at offsets
-13..-8 before every annotated sentence; d = mean(backtracking) - mean(all categories).
Tracking target: whether the model's own next sentence is a backtrack. Drive target: backtracking
sentences per 1,000 thinking tokens. Primary label: Qwen3-8B judge if it passes validation,
otherwise the authors' keyword list ("Wait", "Hmm"), per §0.4.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import torch

from specs import Spec, _find_up

HERE = os.path.dirname(os.path.abspath(__file__))
KEYWORDS = ("Wait", "Hmm")  # Ward et al.'s count_patterns list
OFF = (13, 8)  # positions start-13 .. start-8 (their start-12-1 .. start-12+4)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


# a sentence ends at . ! ? followed by whitespace (not inside numbers such as 4.50), or at a newline
SENT_RE = re.compile(r"[^\n]+?(?:[.!?]+(?=\s)|(?=\n)|$)")


def kw(text):
    return int(any(k in text for k in KEYWORDS))


def extract_annotations(t):
    """Ward et al.'s parser for ["category"] ... ["end-section"] chains (copied logic)."""
    out, pos = [], 0
    while True:
        a = t.find('["', pos)
        if a == -1:
            break
        b = t.find('"]', a)
        if b == -1:
            break
        cat = t[a + 2:b]
        if cat == "end-section":
            pos = b + 2
            continue
        e = t.find('["end-section"]', b + 2)
        if e == -1:
            pos = b + 2
            continue
        out.append((cat, t[b + 2:e].strip()))
        pos = e + 15
    return out


CATS = ["backtracking", "uncertainty-estimation", "initializing", "deduction", "example-testing", "adding-knowledge"]


class D5(Spec):
    key, name = "d5", "backtracking"
    thinking = True
    n_new_mode = dict(gen=256)
    n_chain = 768

    def layer(self, m):
        return int(round(10 * m.L / 32))

    def prefix(self, m, problem):
        t = m.chat(problem)
        if not t.rstrip().endswith("<think>"):
            t = t + "<think>\n"
        elif not t.endswith("\n"):
            t = t + "\n"
        return t

    # ------------------------------------------------------------ calibration
    def calibrate(self, m, log):
        tp = _find_up("third_party", "latent-backtracking")
        chains = json.load(open(os.path.join(tp, "new_annotated_chains", "all_annotated_chains.json"), encoding="utf-8"))
        ell = self.layer(m)
        H, y, lengths, cats = [], [], [], []
        for ch in chains:
            ids = m.tok(self.prefix(m, ch["problem"]), add_special_tokens=False)["input_ids"]
            if m.tok.bos_token_id is not None and (not ids or ids[0] != m.tok.bos_token_id) and "Llama" in m.name:
                ids = [m.tok.bos_token_id] + ids
            starts = []
            for i, (cat, text) in enumerate(extract_annotations(ch["annotated_chain"])):
                seg = m.tok.encode((" " + text) if i > 0 else text, add_special_tokens=False)
                starts.append((cat, len(ids)))
                ids += seg
            ids = ids[:4096]
            starts = [(c, s) for c, s in starts if c in CATS and s - OFF[0] >= 0 and s < len(ids)]
            if not starts:
                continue
            a = m.acts(ids, [ell], list(range(len(ids))))[ell]  # [T, d]
            for c, s in starts:
                H.append(a[s - OFF[0]: s - OFF[1] + 1].mean(0))
                y.append(int(c == "backtracking"))
                lengths.append(s)
                cats.append(c)
        H = np.array(H, dtype=np.float32)
        y = np.array(y)
        d = H[y == 1].astype(np.float64).mean(0) - H.astype(np.float64).mean(0)  # Ward: category mean - overall mean
        log(f"backtracking layer {ell}: {len(y)} annotated sentences, {int(y.sum())} backtracking; |d|={np.linalg.norm(d):.3f}")
        return dict(ell=ell, d=d, H=H.astype(np.float64), y=y.astype(float), lengths=lengths, cats=cats)

    def cal_acts(self, m, cal):
        tp = _find_up("third_party", "latent-backtracking")
        chains = json.load(open(os.path.join(tp, "new_annotated_chains", "all_annotated_chains.json"), encoding="utf-8"))[:300]
        H, y = [], []
        for ch in chains:
            ids = m.tok(self.prefix(m, ch["problem"]), add_special_tokens=False)["input_ids"]
            starts = []
            for i, (cat, text) in enumerate(extract_annotations(ch["annotated_chain"])):
                seg = m.tok.encode((" " + text) if i > 0 else text, add_special_tokens=False)
                starts.append((cat, len(ids)))
                ids += seg
            ids = ids[:4096]
            starts = [(c, s) for c, s in starts if c in CATS and s - OFF[0] >= 0 and s < len(ids)]
            if not starts:
                continue
            a = m.acts(ids, [cal["ell"]], list(range(len(ids))))[cal["ell"]]
            for c, s in starts:
                H.append(a[s - OFF[0]: s - OFF[1] + 1].mean(0))
                y.append(int(c == "backtracking"))
        return np.array(H, dtype=np.float64), np.array(y, dtype=float)

    def check0a(self, m, cal, gen_padded, log):
        res = {}
        tp = _find_up("third_party", "latent-backtracking")
        if "Llama" in m.name:
            ft = torch.load(os.path.join(tp, "ft_steering_vectors.pt"), map_location="cpu")["backtracking"].double().numpy()
            res["cos_released_ft"] = float(cal["d"] @ ft / np.linalg.norm(cal["d"]) / np.linalg.norm(ft))
            log(f"0a cos(rebuilt, released ft vector) = {res['cos_released_ft']:.3f}")
        # headline effect: steering at the authors' magnitude raises the keyword rate (128 tokens, 20 problems)
        probs = [x["question"] for x in ITEMS_D5()["calib0a"]]
        ids = [m.ids(self.prefix(m, p + SUFFIX)) for p in probs]
        d = torch.tensor(cal["d"], dtype=torch.float32)
        for name, mag in (("baseline", 0.0), ("authors_strength", AUTHORS_MAG)):
            ops = (lambda B, mag=mag: dict(start=0, add={cal["ell"]: (mag * d)[None].expand(B, -1)}, pos0=None)) if mag else None
            texts = gen_padded(m, ids, ops, 128)
            res[f"kw_{name}"] = float(np.mean([sum(t.count(k) for k in KEYWORDS) / max(1, len(m.tok(t)["input_ids"])) * 100 for t in texts]))
            log(f"0a keyword % ({name}) = {res[f'kw_{name}']:.3f}")
        return res

    # ------------------------------------------------------------------ items
    def items(self, phase):
        I = ITEMS_D5()
        if phase == "pilot":
            pil = I["pilot"]
            pd = [x["id"] for x in pil][:20]
            return dict(A=pil, oracle=[x for x in pil if x["id"] not in pd], D=pd, E=pd[:6], B=[])
        return dict(A=I["A"], oracle=I["oracle"], D=I["D"], E=I["E"], B=[])

    def prompt_ids(self, m, it, mode):
        ids = m.ids(self.prefix(m, it["question"] + SUFFIX))
        return ids, len(ids) - 1

    def signs(self, it):
        return (1, -1)

    # ---------------------------------------------------------------- outcomes
    def sentences(self, m, toks, positions=True):
        """Sentences of the thinking text (before </think>); with positions=True also the index of
        the token that contains each sentence's first character."""
        toks = m.trunc(toks)
        end_think = m.tok.convert_tokens_to_ids("</think>")
        if end_think in toks:
            toks = toks[: toks.index(end_think)]
        offs = [len(m.tok.decode(toks[:i], skip_special_tokens=False)) for i in range(len(toks))] if positions else None
        text = m.tok.decode(toks, skip_special_tokens=False)
        out = []
        for mm in SENT_RE.finditer(text):
            s = mm.group().strip()
            if not s:
                continue
            ti = None
            if positions:
                c0 = mm.start() + (len(mm.group()) - len(mm.group().lstrip()))
                ti = max((i for i, o in enumerate(offs) if o <= c0), default=None)
                if ti is None:
                    continue
            out.append(dict(text=s, tok=ti, complete=mm.end() < len(text) or text.rstrip().endswith((".", "!", "?"))))
        return out, len(toks)

    def outcome(self, m, it, mode, toks, lp0, top1=None):
        sents, n = self.sentences(m, toks, positions=False)
        nbt = sum(kw(s["text"]) for s in sents)
        return dict(T=1000.0 * nbt / max(n, 1), n_bt=nbt, n_tok=n, n_sent=len(sents),
                    sents=[s["text"][:200] for s in sents])


AUTHORS_MAG = 8.0  # mid-grid magnitude of Ward et al.'s steering scripts (multiplies the raw vector)
_I = None


def ITEMS_D5():
    global _I
    if _I is None:
        _I = json.load(open(os.path.join(HERE, "data", "items_d5.json"), encoding="utf-8"))
    return _I
