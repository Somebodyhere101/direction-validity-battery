"""D9 smoke test (pilot model, a few minutes): cache branching against plain forward passes,
score_chain, the authors' span intervention and the push prompt. python smoke_d9.py [model key]"""
import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from p6lib import Model  # noqa: E402
import spec_d9 as S  # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "q3-1.7b"
m = Model(key, thinking=True)
I = S.ITEMS_D9()
probs = I["pilot"][:2]
t0 = time.time()
bases = [S.prefix(m, p["problem"]) for p in probs]
outs = S.gen_chains(m, [m.ids(b) for b in bases], 384, 2)
print("gen", round(time.time() - t0, 1), "s", [len(o) for o in outs], flush=True)
text = m.tok.decode(outs[0], skip_special_tokens=False).strip()
base = bases[0]
chunks = S.split_steps(text)
print("steps", len(chunks), "numeric", sum(S.perturb_text(c)[1] for c in chunks))
print("example perturbation:", [(c, S.perturb_text(c)[0]) for c in chunks if S.perturb_text(c)[1]][:2])
ystar = S.greedy_answer(m, base + text)
print("ystar (early exit at the end):", repr(ystar))


@torch.no_grad()
def plain_logp(prompt_text):
    ids_p = m.ids(prompt_text + S.CUE)
    ids_f = m.ids(prompt_text + S.CUE + ystar)
    lg = m.hf(input_ids=torch.tensor([ids_f], device=m.dev)).logits[0].float()
    lp = torch.log_softmax(lg, -1)
    return float(sum(lp[len(ids_p) - 1 + k, ids_f[len(ids_p) + k]] for k in range(len(ids_f) - len(ids_p))))


ref = S._Ref(m, base + ".".join(chunks))
cp = [S.perturb_text(c)[0] for c in chunks]
errs, derrs = [], []
for j in (0, len(chunks) // 2, len(chunks) - 1):
    ta, tb = base + ".".join(chunks[:j + 1]), base + ".".join(chunks[:j] + [cp[j]])
    (a, _), (b, _) = S._ee(m, ref, [ta, tb], ystar)
    pa, pb = plain_logp(ta), plain_logp(tb)
    (z1, _), (z2, _) = S._ee(m, ref, [ta, ta], ystar)
    errs += [abs(a - pa), abs(b - pb)]
    derrs.append(abs((math.exp(a) - math.exp(b)) - (math.exp(pa) - math.exp(pb))))
    print(f"step {j}: pair logp {a:.4f} {b:.4f} | plain {pa:.4f} {pb:.4f} | identical-pair diff {z1 - z2:.2e}")
print("max |branched - plain| logp:", max(errs), "| max |p-difference error|:", max(derrs))
refX = S._Ref(m, base + ". " + ".".join(chunks))
j = len(chunks) // 2
h, pos = S._read(m, refX, S.ext_text(base, chunks, j), 16)
ids = m.ids(S.ext_text(base, chunks, j))
h2 = m.acts(ids, [16], [len(ids) - 1])[16][0]
print("read check: pos", pos, len(ids) - 1, "rel diff", float(np.linalg.norm(h - h2) / np.linalg.norm(h2)))
t1 = time.time()
sc, reads = S.score_chain(m, base, text, ystar, read_ell=16)
print("score_chain", round(time.time() - t1, 1), "s; n_steps", sc["n_steps"], "n_num", sc["n_num"], "reads", len(reads))
for st in sc["sentences"][:8]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items() if k in ("j", "T", "nec", "suf", "F", "F_C", "pos", "step_len")})
rec = dict(sc, output=text, ystar=ystar, id="smoke")
print("is_consistent", S.is_consistent(rec))
u = S.choose_unit(rec, "dt", "smoke")
print("unit", {k: v for k, v in u.items() if k != "ctx"})
txt = f"{base}. {u['ctx']}. {u['s_pert']}" + S.CUE
ids, t1_ = S.tok_index(m, txt, len(txt) - len(S.CUE) - 1, first=False)
c0 = len(f"{base}. {u['ctx']}. ")
_, t0_ = S.tok_index(m, txt, c0, first=True)
print("span tokens", repr(m.tok.decode(ids[t0_:t1_ + 1])), "| step", repr(u["s_pert"]))
V = torch.stack([torch.zeros(m.d), torch.randn(m.d) * 30, -torch.randn(m.d) * 30])
toks = S.gen_span(m, ids, t0_, t1_, 16, V, S.N_ANS)
print("span-push answers:", [S.early_answer(m.decode(t)) for t in toks])
r = m.generate(ids, t1_, {16: torch.zeros(1, m.d)}, S.N_ANS)
print("generate clean answer:", S.early_answer(m.decode(r["tokens"][0])))
print("chain_bs cal/test:", S.chain_bs(m, S.CAP_CAL), S.chain_bs(m, S.CAP_TEST))
print("SMOKE OK", round(time.time() - t0, 1), "s")
