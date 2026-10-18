"""D6 reasoning strength (Sheng et al. 2025; PREREGISTRATION.md §3) for paper 6.

Authors' recipe (github.com/AlphaLab-USTC/LRM-plans-CoT, commit 2ba2485; embed.py, AnalyzeOneModel.py,
utils.construct_diff_df, steer_qwen2_vllm.py):
- prompt: their embed.py template rendered by jinja2, which Python's string escapes turn into
  "Please reason step by step, and put your final answer within \\x08oxed{}\\nThis is the problem:\\n"
  + problem + "\\n    " (the "\\b" of "\\boxed" is a backspace character in their source; kept byte for
  byte), in the model's chat template with the generation prompt, which ends with "<think>\\n";
- activations: every hidden state at the last prompt token, over their question file Data/Questions/MATH.json
  (the 7,500 MATH training problems; "Level ?" counted as Level 3);
- vector at each layer: r = mean over k = 2..5 of (mean_k - mean_1), the mean of the four level
  differences r_{k<-1};
- their steering adds lambda * r^(i) to the input of every decoder layer i at every position
  (prompt and generated), lambda in [-0.2, 0.2] (Qwen) and [-0.02, 0.02] (R1-Distill-Llama-8B, Table 10).
Battery claim tuple: the single-layer vector at their single-layer index, hidden state
int(0.8 * (L + 1)) (AnalyzeOneModel.py "Extract steering vectors / Single layer"), i.e. block output
int(0.8 * (L + 1)) - 1; reading position the last prompt token ("\\n" after "<think>").
Tracking target T: thinking tokens (generated tokens before </think>), mean over 4 samples
(temperature 0.6, top-p 0.95), capped (Stage 2b); drive target: the same count under a push.
"""

from __future__ import annotations

import json
import os

import numpy as np

from p6lib import auc, unit
from specs import Spec

HERE = os.path.dirname(os.path.abspath(__file__))
# their embed.py template after Python string escapes ("\b" -> backspace) and jinja2 rendering
TEMPLATE = "Please reason step by step, and put your final answer within \x08oxed{}\nThis is the problem:\n{p}\n    "
# lambda_a: 0.2 is the end of their sweep for R1-Distill-Qwen; 0.02 is their Table 10 value for R1-Distill-Llama-8B.
# Qwen3 (not in their study): the smallest of {0.2, 0.5, 1.0} at which the pilot model (Qwen3-1.7B) met the
# Check 0a gate, else 0.2 (rule fixed before the pilot's lambda sweep was read). None met it: 0.2.
AUTHORS_LAMBDA = {"r1-qwen-7b": 0.2, "r1-llama-8b": 0.02, "q3-8b": 0.2, "q3-1.7b": 0.2}
AUTHORS_LAMBDA_DEFAULT = 0.2


def user_text(problem):
    return TEMPLATE.replace("{p}", problem)


# ------------------------------------------------------------ label-free controls over 5 levels
def n_len_levels(H, levels, lengths):
    """N_len (Stage 1 §3): within each difficulty level, long half minus short half (median split by
    prompt token length), averaged over the five levels."""
    H, levels, lengths = np.asarray(H, np.float64), np.asarray(levels), np.asarray(lengths)
    parts = []
    for lv in sorted(set(levels.tolist())):
        idx = np.where(levels == lv)[0]
        order = idx[np.argsort(lengths[idx], kind="stable")]
        h = len(order) // 2
        parts.append(H[order[h:]].mean(0) - H[order[:h]].mean(0))
    return unit(np.mean(parts, 0))


def n_pc1_levels(H, levels, d):
    """N_pc1: first principal component after centring each level on its own mean; sign set so that
    it projects positively on d."""
    H, levels = np.asarray(H, np.float64), np.asarray(levels)
    C = H.copy()
    for lv in sorted(set(levels.tolist())):
        C[levels == lv] -= C[levels == lv].mean(0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    v = Vt[0]
    return unit(v if v @ d >= 0 else -v)


def level_vector(H, levels):
    """The authors' construction: mean over k = 2..5 of (mean_k - mean_1)."""
    H, levels = np.asarray(H, np.float64), np.asarray(levels)
    m1 = H[levels == 1].mean(0)
    return np.mean([H[levels == k].mean(0) - m1 for k in range(2, 6)], 0)


def n_perm_levels(H, levels, n=20, seed=6):
    """N_perm: the authors' construction after a random relabelling of the five levels (sizes kept)."""
    rng = np.random.default_rng(seed)
    return np.array([unit(level_vector(H, rng.permutation(np.asarray(levels)))) for _ in range(n)])


class D6(Spec):
    key, name = "d6", "reasoning_strength"
    thinking = True
    n_samples = 4

    def layer(self, m):
        return int(0.8 * (m.L + 1)) - 1  # block output index of their hidden-state index int(0.8 * (L+1))

    def authors_lambda(self, m):
        return AUTHORS_LAMBDA.get(m.key, AUTHORS_LAMBDA_DEFAULT)

    def prefix(self, m, problem):
        t = m.chat(user_text(problem))
        if not t.rstrip().endswith("<think>"):
            t = t + "<think>\n"
        elif not t.endswith("\n"):
            t = t + "\n"
        return t

    def prompt_ids(self, m, it, mode="gen"):
        ids = m.ids(self.prefix(m, it["problem"]))
        return ids, len(ids) - 1

    # ------------------------------------------------------------ calibration
    def cal_items(self):
        return json.load(open(os.path.join(HERE, "data", "calibration_d6.json"), encoding="utf-8"))

    def calibrate(self, m, log):
        C = self.cal_items()
        ell = self.layer(m)
        L = m.L
        levels = np.array([x["level"] for x in C])
        S = np.zeros((5, L, m.d), dtype=np.float64)  # per-level sums of every block output at the last prompt token
        n = np.zeros(5)
        H, lengths = np.zeros((len(C), m.d), dtype=np.float32), []
        for i, x in enumerate(C):
            ids = self.prompt_ids(m, x)[0]
            a = m.acts(ids, list(range(L)), [len(ids) - 1])
            A = np.stack([a[l][0] for l in range(L)]).astype(np.float64)  # [L, d]
            S[x["level"] - 1] += A
            n[x["level"] - 1] += 1
            H[i] = A[ell]
            lengths.append(len(ids))
            if i % 1000 == 0:
                log(f"calibration {i}/{len(C)}")
        M = S / n[:, None, None]  # level means [5, L, d]
        Rk = M[1:] - M[0:1]  # r_{k<-1} for k = 2..5, [4, L, d]
        R_all = Rk.mean(0)  # the authors' vector at every block output [L, d]
        cosm = []
        for l in range(L):
            U = Rk[:, l] / np.linalg.norm(Rk[:, l], axis=1, keepdims=True)
            G = U @ U.T
            cosm.append(float(G[np.triu_indices(4, 1)].mean()))
        d = R_all[ell]
        y = (levels > 1).astype(float)
        log(f"reasoning-strength layer {ell}: |d|={np.linalg.norm(d):.3f}; mean pairwise cos(r_k<-1) {cosm[ell]:.3f}; "
            f"max over layers {max(cosm[1:]):.3f} at block {int(np.argmax(cosm[1:])) + 1}")
        Ul = Rk[:, ell] / np.linalg.norm(Rk[:, ell], axis=1, keepdims=True)
        return dict(ell=ell, d=d, H=H.astype(np.float64), y=y, levels=levels, lengths=lengths, R_all=R_all,
                    level_counts=n.tolist(), cos_pairwise_by_layer=cosm, rk_norms=np.linalg.norm(Rk[:, ell], axis=1).tolist(),
                    rk_cos=(Ul @ Ul.T).tolist())

    def cal_acts(self, m, cal):
        """Check 0b: every fifth calibration problem (1,500), at the same layer and position."""
        C = self.cal_items()[::5]
        H = []
        for x in C:
            ids = self.prompt_ids(m, x)[0]
            H.append(m.acts(ids, [cal["ell"]], [len(ids) - 1])[cal["ell"]][0])
        return np.array(H, dtype=np.float64), np.array([float(x["level"] > 1) for x in C])

    # ------------------------------------------------------------------ items
    def items(self, phase):
        I = ITEMS_D6()
        if phase == "pilot":
            pd = set(I["pilot_D"])
            return dict(A=I["pilot"], oracle=[x for x in I["pilot"] if x["id"] not in pd], D=I["pilot_D"], E=I["pilot_E"],
                        B=I["pilot_B"], B_pairs=30, calib0a=I["calib0a"])
        return dict(A=I["A"], oracle=I["oracle"], D=I["D"], E=I["E"], B=I["B"], B_pairs=120, calib0a=I["calib0a"])

    def signs(self, it):
        return (1, -1)

    def B_set(self, m, pool, n_pairs=120):
        """Check B: level-1 vs level-2..5 test problems, same subject, formatted prompt lengths within
        +-1 token. Greedy in the pool's fixed (seeded) order, nearest length first; evaluation only."""
        if not pool:
            return []
        L1 = [dict(x, n=len(self.prompt_ids(m, x)[0])) for x in pool["lv1"]]
        LK = [dict(x, n=len(self.prompt_ids(m, x)[0])) for x in pool["lvk"]]
        used, out = set(), []
        for a in L1:
            cands = [(abs(b["n"] - a["n"]), j) for j, b in enumerate(LK)
                     if j not in used and b["subject"] == a["subject"] and abs(b["n"] - a["n"]) <= 1]
            if not cands:
                continue
            j = min(cands)[1]
            used.add(j)
            b = LK[j]
            pid = f"p{len(out) // 2}"
            for lab, x in ((0, a), (1, b)):
                ids, st = self.prompt_ids(m, x)
                out.append(dict(id=f"{pid}:{x['id']}", text=x["problem"], label=lab, ids=ids, start=st, level=x["level"]))
            if len(out) // 2 >= n_pairs:
                break
        return out


def gate_d6(c0, g):
    """Check 0a gate (Stage 2b): with the authors' every-layer steering at their strength on the 20
    MATH-500 problems, -lambda lowers the mean thinking-token count by at least the drive SESOI (10%)
    and +lambda does not lower it."""
    return bool(c0["ratio_minus"] <= g["ratio_minus_max"] and c0["ratio_plus"] >= g["ratio_plus_min"])


_I = None


def ITEMS_D6():
    global _I
    if _I is None:
        _I = json.load(open(os.path.join(HERE, "data", "items_d6.json"), encoding="utf-8"))
    return _I
