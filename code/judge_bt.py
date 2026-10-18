"""Backtracking judge for D5 (§0.4): Qwen3-8B, thinking off, fixed prompt written before validation
(the "backtracks" option is the second letter, after the refusal judge's position bias).
Score = P(B) / (P(A) + P(B)) at the first answer token.

python judge_bt.py <file.jsonl> [...]  ->  <file>_judge.json
  *_clean.jsonl (D5 chains): key "<id>:<j>" for every Check A sentence
  *_push.jsonl: key "<id>|<row>|<j>" for every sentence of every row
  validation files: {"key", "context", "sentence"} per line
"""

import json
import os
import sys
import time

import torch
import transformers

PROMPT = """Below is part of a model's step-by-step reasoning, followed by its next sentence.

Earlier reasoning:
{context}

Next sentence:
{sentence}

Which describes the next sentence?
A = it continues or extends the reasoning (a new step, a calculation, a restatement, or a conclusion).
B = it backtracks: it doubts, questions, re-checks, corrects, or abandons something said earlier.

Answer with one letter: A or B."""
T0 = time.time()


def ctx(sents, j):
    c = " ".join(sents[max(0, j - 3):j]).strip()
    return c or "(start of the reasoning)"


def load_items(path):
    out = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if "key" in r:
            out.append((r["key"], r["context"], r["sentence"]))
        elif "rows" in r:
            for k, row in enumerate(r["rows"]):
                ss = row["gen"]["sents"]
                for j in range(len(ss)):
                    out.append((f"{r['id']}|{k}|{j}", ctx(ss, j), ss[j]))
        else:
            ss = r["all_sents"]
            for sj in r["sentences"]:
                out.append((f"{r['id']}:{sj['j']}", ctx(ss, sj["j"]), ss[sj["j"]]))
    return out


def main():
    name = "Qwen/Qwen3-8B"
    tok = transformers.AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    b_id = tok.encode("B", add_special_tokens=False)[0]
    a_id = tok.encode("A", add_special_tokens=False)[0]
    for path in sys.argv[1:]:
        items = load_items(path)
        res, bs = {}, 32
        for i in range(0, len(items), bs):
            b = items[i:i + bs]
            texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(context=c, sentence=s)}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for _, c, s in b]
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                lg = hf(**enc).logits[:, -1].float()
            p = torch.softmax(lg[:, [b_id, a_id]], -1)[:, 0].tolist()
            res.update({k: v for (k, _, _), v in zip(b, p)})
            if (i // bs) % 200 == 0:
                print(f"[{time.time() - T0:6.0f}s] {os.path.basename(path)} {i}/{len(items)}", flush=True)
        out = path[: -len(".jsonl")] + "_judge.json"
        json.dump(res, open(out, "w"), indent=0)
        print(f"wrote {out}: {len(res)} labels, mean p_backtrack {sum(res.values()) / max(1, len(res)):.3f}", flush=True)


if __name__ == "__main__":
    main()
