"""Paper 6 downloads on DQ: panel models (ungated) and datasets. Run without HF_HUB_OFFLINE."""
from huggingface_hub import snapshot_download
import sys
MODELS = ["allenai/OLMo-2-1124-7B-Instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
          "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
DATA = ["bench-llm/or-bench", "Paul/XSTest"]
for d in DATA:
    try:
        p = snapshot_download(d, repo_type="dataset")
        print("OK", d, p, flush=True)
    except Exception as e:
        print("FAIL", d, repr(e)[:300], flush=True)
for m in MODELS:
    try:
        p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
        print("OK", m, p, flush=True)
    except Exception as e:
        print("FAIL", m, repr(e)[:300], flush=True)
print("ALLDONE", flush=True)
