"""Download D9's problem sets as JSON through the Hugging Face datasets-server API (no numpy needed).
python -I fetch_d9_data.py <out_dir>
- AI-MO/aimo-validation-amc (83 AMC 2022-23 problems; the set behind Zhao et al.'s AMC outputs:
  same id/problem/answer/url fields)
- EleutherAI/hendrycks_math, test split, all seven subjects (Zhao et al.'s out-of-domain MATH set)
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
API = "https://datasets-server.huggingface.co/rows?"


def rows(dataset, config, split):
    out, off = [], 0
    while True:
        q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=off, length=100))
        for attempt in range(5):
            try:
                with urllib.request.urlopen(API + q, timeout=60) as r:
                    js = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:  # rate limit or transient error
                print("retry", dataset, config, off, e)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {dataset} {config} {off}")
        batch = [x["row"] for x in js["rows"]]
        out += batch
        off += len(batch)
        if not batch or off >= js.get("num_rows_total", off):
            return out


amc = rows("AI-MO/aimo-validation-amc", "default", "train")
json.dump(amc, open(os.path.join(OUT, "aimo_validation_amc.json"), "w", encoding="utf-8"), indent=0)
print("amc", len(amc))
math = {}
for cfg in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"):
    math[cfg] = rows("EleutherAI/hendrycks_math", cfg, "test")
    print(cfg, len(math[cfg]))
json.dump(math, open(os.path.join(OUT, "hendrycks_math_test.json"), "w", encoding="utf-8"), indent=0)
