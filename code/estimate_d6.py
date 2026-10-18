"""D6 run-time estimate per panel model (Stage 1 §8 long-generation rule). Pure Python.

python estimate_d6.py <dir with the pilot result files>

Inputs (pilot phase):
- <model>_pilot_timing_cap4096.json for each panel model: 20 pilot problems x 4 samples, clean-style,
  continuous batching at the main cap (4096): throughput (tokens per second) and thinking lengths;
- q3-1.7b_pilot_push.jsonl: tokens per pushed chain relative to the clean chains (the pushes' effect on
  length), and KL seconds per item.
Cost model, per (direction, model):
  tokens per clean chain   = mean of min(T, 4096) over the model's timing chains (+ 300 answer tokens, clean stage)
  tokens per pushed chain  = the same x the pilot's pushed/clean ratio
  generation hours         = chains x tokens per chain / the model's timing throughput
  KL hours (push)          = pilot KL seconds per item x parameter ratio x (pushes / pilot pushes)
                             x (chain length / pilot chain length)
  calibration              = 0.25 h of single forwards for a 7-8B model, plus Check 0a's 240 chains
Designs: "full" = Stage 1 §8 (Check D 120 items, R = 12 x 7 norms, 8 permuted x 7 norms); "reduced" =
R = 8, 5 norms, 80 items. Chains per Check D item: 4 clean + 4 x (2 primary d + 6 control + 2 authors') + 4
exploratory d (one sample each) + one per random/permuted direction and norm (188 full, 128 reduced).
Throughput is the timing probe's at its slot count (2 GiB reserve in the main runs gives the 8B models ~15%
more slots, so these figures are upper bounds). Check E (only for signs that pass Check D) is estimated per sign:
lockstep batches of rows_for(cap) rows that run to the longest chain, i.e. ~ chains x cap tokens.
"""
import json
import os
import sys

RES = sys.argv[1]
CAP = 4096
PARAMS = {"q3-1.7b": 2.03, "r1-qwen-7b": 7.62, "r1-llama-8b": 8.03, "q3-8b": 8.19}  # billions
LAYERS = {"q3-1.7b": 28, "r1-qwen-7b": 28, "r1-llama-8b": 32, "q3-8b": 36}
PANEL = ["r1-qwen-7b", "r1-llama-8b", "q3-8b"]


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


push = [json.loads(l) for l in open(os.path.join(RES, "q3-1.7b_pilot_push.jsonl"), encoding="utf-8")]
chains_pilot = mean(sum(len(r["s"]) for r in rec["rows"]) for rec in push)
push_ratio = mean(rec["gen_tokens"] for rec in push) / chains_pilot / mean(rec["T_clean"] for rec in push)
kl_item = mean(rec["kl_seconds"] for rec in push)
pushes_pilot = mean(sum(1 for r in rec["rows"] if r["kind"] != "clean") for rec in push)
cont_pilot = mean(rec["n_cont"] for rec in push)


def design(kind):
    items, R, nn = (120, 12, 7) if kind == "full" else (80, 8, 5)
    # 4 clean + 4 x (2 primary d, 6 control, 2 authors') + 1 x 4 exploratory d + one per random/permuted direction and norm
    return items, 4 + 4 * (2 + 6 + 2) + 4 + (R + 8) * nn, 6 + 6 + 2 + (R + 8) * nn  # items, chains/item, distinct pushes/item


out = dict(pilot=dict(push_ratio=push_ratio, kl_seconds_per_item=kl_item, pushes_per_item=pushes_pilot, cont_tokens=cont_pilot), models={})
for mk in PANEL + ["q3-1.7b"]:
    t = load(f"{mk}_pilot_timing_cap{CAP}.json")
    if not t:
        continue
    T, tps, pr = t["T"], t["tok_per_s"], PARAMS[mk] / PARAMS["q3-1.7b"]
    tok = mean(min(x, CAP) for x in T)
    res = dict(tok_per_s=round(tps), slots=t["slots"], peak_gb=round(t["peak_gb"], 1), T_mean=round(mean(T)), capped_frac=round(mean(t["capped"]), 3),
               T_by_level={lv: round(mean(x for x, l in zip(T, t["level"]) if l == lv)) for lv in sorted(set(t["level"]))}, est={})
    clean_h = 1440 * (tok + 300) / tps / 3600
    cal_h = 0.25 + 240 * tok / tps / 3600
    for kind in ("full", "reduced"):
        items, chains, pushes = design(kind)
        gen_h = items * chains * tok * push_ratio / tps / 3600
        kl_h = items * kl_item * pr * (pushes / pushes_pilot) * (tok / cont_pilot) / 3600
        res["est"][kind] = {k: round(v, 1) for k, v in dict(cal=cal_h, clean=clean_h, push_gen=gen_h, push_kl=kl_h,
                                                            total=cal_h + clean_h + gen_h + kl_h).items()}
    later = len(range(int(0.8 * (LAYERS[mk] + 1)), LAYERS[mk]))
    later = later if LAYERS[mk] <= 32 else (later + 1) // 2
    res["est"]["E_per_sign"] = round(60 * 4 * (later + 7) * CAP / tps / 3600, 1)
    out["models"][mk] = res
print(json.dumps(out, indent=1, default=float))
