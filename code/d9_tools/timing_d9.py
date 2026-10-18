"""D9 run-time probe on a panel model, pilot problems only (no results used): chain generation at
both caps, step scoring, and one push-sized batch. python timing_d9.py <model key>"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from p6lib import Model  # noqa: E402
import spec_d9 as S  # noqa: E402

key = sys.argv[1]
m = Model(key, thinking=True)
I = S.ITEMS_D9()
out = dict(model=key, L=m.L, ell=int(round(17 * m.L / 28)) - 1)


def tick():
    torch.cuda.synchronize()
    return time.time()


for name, probs, cap in (("test", I["pilot"][:16], S.CAP_TEST), ("cal", I["cal"][:8], S.CAP_CAL)):
    bs = S.chain_bs(m, cap)
    probs = probs[:bs]
    bases = [S.prefix(m, p["problem"]) for p in probs]
    t = tick()
    outs = S.gen_chains(m, [m.ids(b) for b in bases], cap, bs)
    dt = tick() - t
    out[f"gen_{name}"] = dict(batch=bs, seconds=dt, max_tokens=max(len(o) for o in outs),
                              tokens=[len(o) for o in outs], s_per_step=dt / max(len(o) for o in outs))
    print(name, out[f"gen_{name}"], flush=True)
    sc_t, n_steps, n_num = 0.0, 0, 0
    for p, base, toks in list(zip(probs, bases, outs))[:3]:
        text = m.tok.decode(toks, skip_special_tokens=False).strip()
        y = S.last_boxed(text) or S.greedy_answer(m, base + text)
        t = tick()
        sc, _ = S.score_chain(m, base, text, y, read_ell=out["ell"] if name == "test" else None)
        sc_t += tick() - t
        n_steps += sc["n_steps"]
        n_num += sc["n_num"]
    out[f"score_{name}"] = dict(seconds=sc_t, n_steps=n_steps, n_num=n_num, s_per_numeric_step=sc_t / max(n_num, 1))
    print(name, out[f"score_{name}"], flush=True)
    if name == "test":
        text = m.tok.decode(outs[0], skip_special_tokens=False).strip()
        ids = m.ids(bases[0] + text[:6000] + S.CUE)
        for rows in (24, 48):
            z = {out["ell"]: torch.zeros(rows, m.d)}
            try:
                t = tick()
                r = m.generate(ids, len(ids) - 12, z, S.N_ANS)
                m.tf_kl(ids, len(ids) - 12, z, m.trunc(r["tokens"][0]))
                out[f"push_batch_{rows}"] = dict(prompt_tokens=len(ids), seconds=tick() - t,
                                                 max_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
            except torch.cuda.OutOfMemoryError:
                out[f"push_batch_{rows}"] = "OOM"
                torch.cuda.empty_cache()
            print(rows, out[f"push_batch_{rows}"], flush=True)
os.makedirs(os.path.join("res", "d9"), exist_ok=True)
json.dump(out, open(os.path.join("res", "d9", f"{key}_timing.json"), "w"), indent=1)
print("DONE", json.dumps(out))
