import transformers, torch, inspect, importlib.util as u
from transformers import DynamicCache as D
print(transformers.__version__, torch.__version__)
print([m for m in dir(D) if not m.startswith("_")])
print("crop" in dir(D), inspect.signature(D.__init__))
for n in ("datasets", "pyarrow", "sklearn", "scipy", "pandas"):
    print(n, u.find_spec(n) is not None)
for name in ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "Qwen/Qwen3-1.7B"):
    tok = transformers.AutoTokenizer.from_pretrained(name)
    kw = {"enable_thinking": True} if "Qwen3" in name else {}
    print(name, repr(tok.apply_chat_template([{"role": "user", "content": "Q?"}], tokenize=False, add_generation_prompt=True, **kw)))
    print("  ", tok.encode("142", add_special_tokens=False), tok.encode("x\n The final answer is \\boxed{142}", add_special_tokens=False))
import os
for f in ("amc-qwen-dir-uf-ff.pt", "llama8b-amc-uf-ff-dir.pt", "nemotron-1d5b-amc-dir-uf-ff.pt"):
    p = os.path.join("..", "third_party", "ttd", "data", "pt", f)
    if os.path.exists(p):
        v = torch.load(p, map_location="cpu", weights_only=True)
        print(f, tuple(v.shape), v.dtype, [round(float(x), 2) for x in v.float().norm(dim=-1)])
