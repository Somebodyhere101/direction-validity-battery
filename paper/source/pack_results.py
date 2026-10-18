"""Packs paper 6's per-item results for release (run on DQ from C:\\Users\\CAAB\\jspace):
python pack_results.py <out.zip>

- One canonical source folder per direction (where its runs were produced).
- Main-phase per-item records only: clean, push, pushE, chains, sweep and judge files. No vectors (.pt, .npz)
  are included, so no direction (the refusal-removal direction in particular) is redistributed.
- Model-written text is removed: every string value longer than MAXLEN characters is dropped, with its key
  path counted in strip_report.json so that the removal can be audited. The build stops if any such string
  survives. Numbers, labels, ids and parsed answers are kept unchanged.
"""

import collections
import json
import os
import re
import sys
import zipfile

MAXLEN = 40
# model-written or prompt text, dropped whatever its length
DROP_KEYS = {"text", "text_pert", "texts", "sents", "all_sents", "answer", "answers", "system", "question", "chain",
             "chains", "response", "completion", "gen_text", "cont", "steered", "baseline", "sentence", "prefix",
             "s_pert", "ctx"}
DROP_IF_STR = {"s"}  # D9 step text; "s" is a number elsewhere
string_keys = collections.Counter()
SRC = {
    "d1": "paper6/res/d1", "d2": "paper6/res/d2", "d3": "paper6/res/d3",
    "d4": "paper6c/res/d4", "d4g": "paper6c/res/d4g", "d5": "paper6b/res/d5",
    "d6": "paper6_d6/res/d6", "d7": "paper6_d7/res/d7", "d8": "paper6c/res/d8", "d9": "paper6_d9/res/d9",
}
KEEP = re.compile(r"_main_(clean|push|pushE|chains|osweep)\.jsonl$|_main_(clean|push)_judge\.json$|_main_extract_judged\.jsonl$")
dropped = collections.Counter()


def scrub(x, path):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in DROP_KEYS or (k in DROP_IF_STR and isinstance(v, str)):
                dropped[f"{path}.{k} (key)"] += 1
                continue
            if isinstance(v, str) and len(v) > MAXLEN:
                dropped[f"{path}.{k}"] += 1
                continue
            if isinstance(v, str):
                string_keys[f"{path.split('/')[0]}:{path.split('.', 1)[-1] if '.' in path else ''}.{k}"] += 1
            out[k] = scrub(v, f"{path}.{k}")
        return out
    if isinstance(x, list):
        kept = []
        for v in x:
            if isinstance(v, str) and len(v) > MAXLEN:
                dropped[path + "[]"] += 1
                continue
            kept.append(scrub(v, path + "[]"))
        return kept
    return x


def check(x):
    if isinstance(x, dict):
        return all(check(v) for v in x.values())
    if isinstance(x, list):
        return all(check(v) for v in x)
    return not (isinstance(x, str) and len(x) > MAXLEN)


out = sys.argv[1]
n_files = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, folder in SRC.items():
        if not os.path.isdir(folder):
            print("missing", folder)
            continue
        for name in sorted(os.listdir(folder)):
            if not KEEP.search(name) or "superseded" in name:
                continue
            p = os.path.join(folder, name)
            if name.endswith(".jsonl"):
                lines = []
                for line in open(p, encoding="utf-8"):
                    r = scrub(json.loads(line), f"{d}/{name}")
                    assert check(r), (d, name)
                    lines.append(json.dumps(r, separators=(",", ":")))
                z.writestr(f"results/{d}/{name}", "\n".join(lines) + "\n")
            else:
                r = scrub(json.load(open(p, encoding="utf-8")), f"{d}/{name}")
                assert check(r), (d, name)
                z.writestr(f"results/{d}/{name}", json.dumps(r, separators=(",", ":")))
            n_files += 1
    report = {k: v for k, v in sorted(dropped.items())}
    z.writestr("results/strip_report.json", json.dumps(dict(maxlen=MAXLEN, drop_keys=sorted(DROP_KEYS), dropped_values_by_path=report,
                                                             remaining_string_fields=dict(sorted(string_keys.items()))), indent=1))
print(n_files, "files;", sum(dropped.values()), "strings dropped across", len(dropped), "key paths")
print("remaining string fields (all <= %d chars):" % MAXLEN)
for k, v in sorted(string_keys.items()):
    print(f"{v:>9}  {k}")
