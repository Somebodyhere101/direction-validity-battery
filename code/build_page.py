"""Builds paper 6's results page from the per-model summaries: python build_page.py -> page.html"""

import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
MODEL_LABEL = {"q25-7b": "Qwen2.5-7B-Instruct", "q3-8b": "Qwen3-8B", "olmo-7b": "OLMo-2-7B-Instruct",
               "r1-llama-8b": "R1-Distill-Llama-8B", "r1-qwen-7b": "R1-Distill-Qwen-7B", "q3-8b-think": "Qwen3-8B (thinking)"}
COLUMNS = [("q25-7b", "Qwen2.5-7B-Instruct"), ("q3-8b", "Qwen3-8B"), ("olmo-7b", "OLMo-2-7B-Instruct"),
           ("r1-llama-8b", "R1-Distill-Llama-8B"), ("r1-qwen-7b", "R1-Distill-Qwen-7B"), ("q3-8b-think", "Qwen3-8B, thinking")]
TEXT = json.load(open(os.path.join(HERE, "page_text.json"), encoding="utf-8"))

DIRS = [
    dict(key="d1", id="D1", name="Refusal", role="positive control", claim="tracks and drives", panel=["q25-7b", "q3-8b", "olmo-7b"],
         drive_target="sub", drive_unit="change in refusal rate beyond the reference",
         facts=[("source", "Arditi et al. 2024"), ("target", "actual refusal, phrase-matched"), ("items", "472 XSTest / OR-Bench"), ("SESOI", "10 points")]),
    dict(key="d2", id="D2", name="Truth", role="static-concept control", claim="tracks and drives", panel=["q25-7b", "q3-8b", "olmo-7b"],
         drive_target="V", drive_unit="change in the rate of 'True' verdicts beyond the reference",
         facts=[("source", "Marks & Tegmark 2024"), ("target", "the model's own verdict"), ("items", "240 curated statements"), ("SESOI", "10 points")]),
    dict(key="d3", id="D3", name="Compute–retrieve axis", role="known-answer case", claim="tracks and drives", panel=["q25-7b", "q3-8b", "olmo-7b"],
         drive_target="D", drive_unit="change in settling depth (fraction of depth) beyond the reference",
         facts=[("source", "Ramdan 2026a"), ("target", "logit-lens settling depth"), ("items", "240 format-matched products"), ("SESOI", "1 layer")]),
    dict(key="d4", id="D4", name="Reasoning vs memorisation (LiReFs)", role="process claim", claim="tracks and drives", panel=["q25-7b", "q3-8b", "olmo-7b"],
         drive_target="D", drive_unit="change in settling depth (fraction of depth) beyond the reference",
         facts=[("source", "Hong et al. 2025"), ("target", "logit-lens settling depth (D3's items)"), ("items", "240 format-matched products"), ("SESOI", "1 layer")]),
    dict(key="d4g", id="D4G", name="LiReFs on GSM8K", role="process claim (same direction as D4)", claim="tracks and drives", panel=["q25-7b", "q3-8b", "olmo-7b"],
         drive_target="correct", drive_unit="change in direct-answer accuracy beyond the reference",
         facts=[("source", "Hong et al. 2025"), ("target", "chain-of-thought necessity; drive: direct-answer accuracy"), ("items", "400 GSM8K problems"), ("SESOI", "10 points")]),
    dict(key="d5", id="D5", name="Backtracking", role="process claim", claim="tracks and drives", panel=["r1-llama-8b", "r1-qwen-7b", "q3-8b"],
         drive_target="T", drive_unit="change in backtracking sentences per 1,000 thinking tokens",
         facts=[("source", "Ward et al. 2025"), ("target", "the model's own next sentence"), ("items", "240 GSM8K chains"), ("SESOI", "25% of the clean rate")]),
    dict(key="d6", id="D6", name="Reasoning strength", role="process claim", claim="tracks and drives", panel=["r1-qwen-7b", "r1-llama-8b", "q3-8b"],
         drive_target="T", drive_unit="change in thinking length beyond the reference",
         facts=[("source", "Sheng et al. 2025"), ("target", "the model's own thinking length"), ("items", "240 MATH problems"), ("SESOI", "10% of the clean length")]),
    dict(key="d7", id="D7", name="Known-entity awareness", role="process claim", claim="tracks and drives", panel=["q25-7b", "q3-8b", "olmo-7b"],
         drive_target="drv", drive_unit="change in refusal (known) or hallucination (unknown) beyond the reference",
         facts=[("source", "Ferrando et al. 2025"), ("target", "the attribute is actually recalled"), ("items", "240 entities, token-count stratified"), ("SESOI", "10 points")]),
    dict(key="d8", id="D8", name="Sycophancy (persona vector)", role="process claim", claim="tracks and drives", panel=["q25-7b", "q3-8b", "olmo-7b"],
         drive_target="flip", drive_unit="change in the rate of flipping to the user's wrong answer",
         facts=[("source", "Chen et al. 2025"), ("target", "the answer flips to the user's stated wrong option"), ("items", "800 ARC questions"), ("SESOI", "10 points")]),
    dict(key="d9", id="D9", name="True thinking", role="process claim", claim="tracks and drives", panel=["r1-qwen-7b", "r1-llama-8b", "q3-8b"],
         drive_target="F", drive_unit="change in the rate at which the answer follows a perturbed step",
         facts=[("source", "Zhao et al. 2025"), ("target", "each step's True Thinking Score"), ("items", "240 MATH chains"), ("SESOI", "10 points")]),
]


def scrub(s):
    """Drop model-written text from a summary before it is embedded in the page (D7's Check 0a keeps sample answers
    under check0a.rows), and refuse to build if any long string is left: the page shows numbers, not model outputs."""
    if s is None:
        return None
    if isinstance(s.get("check0a"), dict):
        s["check0a"] = {k: v for k, v in s["check0a"].items() if k not in ("rows", "texts", "samples")}

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for v in x:
                walk(v, path + "[]")
        elif isinstance(x, str) and len(x) > 200:
            raise SystemExit(f"long string in summary at {path}: refusing to embed possible model text")
    walk(s, s.get("direction", "?") + "/" + s.get("model", "?"))
    return s


def state_of(d, m):
    logs = os.path.join(RES, d, f"{m}_main_push.jsonl")
    if os.path.exists(os.path.join(RES, d, f"{m}_main_summary.json")):
        return "done"
    return "running" if os.path.exists(logs) else "queued"


data = dict(generated=time.strftime("%Y-%m-%d %H:%M AST"), columns=[], directions=[], hypotheses=[], notes=TEXT["notes"], checks=TEXT["checks"])
for k, lab in COLUMNS:
    data["columns"].append(dict(key=k, label=lab))
n_done = n_all = 0
for d in DIRS:
    out = dict(d)
    out.update(TEXT["directions"].get(d["key"], {}))
    out["models"] = []
    for m in d["panel"]:
        col = "q3-8b-think" if (d["key"] in ("d5", "d6", "d9") and m == "q3-8b") else m
        p = os.path.join(RES, d["key"], f"{m}_main_summary.json")
        s = scrub(json.load(open(p, encoding="utf-8"))) if os.path.exists(p) else None
        n_all += 1
        n_done += s is not None
        out["models"].append(dict(key=col, label=MODEL_LABEL[col], summary=s, state=state_of(d["key"], m)))
    vs = [x["summary"]["verdict"] for x in out["models"] if x["summary"]]
    out["overall"] = None
    # Stage 1 §6: the verdict received in at least two of three panel models, which is settled once two agree
    out["overall"] = next((v for v in set(vs) if vs.count(v) >= 2), "MIXED" if len(vs) == 3 else None)
    data["directions"].append(out)
agg_p = os.path.join(RES, "aggregate_main.json")
data["hypotheses"] = TEXT["hypotheses"]
data["status"] = f"{n_done} of {n_all} direction × model runs analysed · built {data['generated']} · all runs complete · D10 not run (secondary, cost)"
html = open(os.path.join(HERE, "page_template.html"), encoding="utf-8").read().replace("__DATA__", json.dumps(data, default=str))
open(os.path.join(HERE, "page.html"), "w", encoding="utf-8").write(html)
print("page.html", len(html) // 1024, "KB;", n_done, "of", n_all, "runs")
