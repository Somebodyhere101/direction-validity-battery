"""Split-half reliability of the rebuilt D9 direction (CPU): cosine between difference-of-means
directions built from disjoint halves of the calibration steps, split by problem (chain) so the
halves share no chain. python splithalf_d9.py <cal.pt>"""
import sys

import numpy as np
import torch

cal = torch.load(sys.argv[1], weights_only=False)
H, y = np.asarray(cal["H"]), np.asarray(cal["y"])
chain = np.array([mm["id"] for mm in cal["cal_meta"]])
ids = np.unique(chain)
rng = np.random.default_rng(0)
cs, cs_full = [], []
d_full = H[y == 1].mean(0) - H[y == 0].mean(0)
for _ in range(500):
    a = set(rng.permutation(ids)[: len(ids) // 2])
    ma = np.array([c in a for c in chain])
    if min((ma & (y == 1)).sum(), (~ma & (y == 1)).sum(), (ma & (y == 0)).sum(), (~ma & (y == 0)).sum()) == 0:
        continue
    da = H[ma & (y == 1)].mean(0) - H[ma & (y == 0)].mean(0)
    db = H[~ma & (y == 1)].mean(0) - H[~ma & (y == 0)].mean(0)
    cs.append(da @ db / np.linalg.norm(da) / np.linalg.norm(db))
    cs_full.append(da @ d_full / np.linalg.norm(da) / np.linalg.norm(d_full))
cs = np.array(cs)
r = np.median(cs)
print(f"chains {len(ids)}, high {int(y.sum())}, low {int((1 - y).sum())}, splits {len(cs)}")
print(f"split-half cos: median {r:.3f}, 5-95% [{np.quantile(cs, 0.05):.3f}, {np.quantile(cs, 0.95):.3f}]")
# two independent full-size estimates: noise variance halves, so cos ~ r_full = 2r / (1 + r) (Spearman-Brown analogue)
print(f"implied cos between two independent full-size rebuilds: {2 * r / (1 + r):.3f}")
print(f"cos(half, full) median {np.median(cs_full):.3f}")
