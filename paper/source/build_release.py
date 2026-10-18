"""Assembles paper 6's release from the study folder (../paper6): the git repository (repo/) and the
Zenodo files (zenodo_upload/). Standard library only. Run from paper6_ms/: python build_release.py

- The repository mirrors paper6/'s layout, so `sha256sum -c PREREG_SHA256SUMS` runs from its root. Files are
  copied byte for byte. Some shared files (analyze.py, run.py, stage2_settings.json) were edited after
  earlier freezes, as DEVIATIONS_LOG.md records; for those, only the latest hash line can match.
- Data files built from repositories without a licence, or containing competition problems (AMC), are not
  redistributed; DATA.md says how the make_items scripts rebuild them from the original sources.
- No direction vectors (.pt, .npz) are included. Per-run summaries are scrubbed of model-written text
  (D7's Check 0a sample answers), as on the results page. Per-item results are packed separately on the
  GPU machine by pack_results.py, with every model-written string removed.
"""

import hashlib
import json
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.normpath(os.path.join(HERE, "..", "paper6"))
REPO = os.path.join(HERE, "repo")
ZEN = os.path.join(HERE, "zenodo_upload")
NAME = "direction-validity-battery"

TOP = ["PREREGISTRATION.md", "PREREGISTRATION_DRAFT_original.md", "PREREGISTRATION_STAGE2A.md",
       "PREREGISTRATION_STAGE2B_D4.md", "PREREGISTRATION_STAGE2B_D5.md", "PREREGISTRATION_STAGE2B_D6.md",
       "PREREGISTRATION_STAGE2B_D7.md", "PREREGISTRATION_STAGE2B_D8.md", "PREREGISTRATION_STAGE2B_D9.md",
       "PREREG_SHA256SUMS", "DEVIATIONS_LOG.md", "INTEGRATION_D6.md", "INTEGRATION_D7.md", "INTEGRATION_D9.md",
       "LITERATURE.md", "D9_MAIN_JOBS.txt"]
CODE_EXCLUDE_DIRS = {"res", "res_superseded", "logs", "__pycache__", ".claude", "base"}
CODE_EXCLUDE_FILES = {"wait_dq.ps1", "inspect_cache_api.py", "page.html"}
DATA_EXCLUDE = {"calibration.json", "items.json", "items_d9.json", "mmlu-pro-3000samples.json"}
PAPER = ["main.tex", "references.bib", "build_release.py", "pack_results.py"]


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def scrub(s):
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
            raise SystemExit(f"long string left in {path}")
    walk(s, s.get("direction", "?") + "/" + s.get("model", "?"))
    return s


if os.path.exists(REPO):
    shutil.rmtree(REPO)
os.makedirs(REPO)
for f in TOP:
    copy(os.path.join(STUDY, f), os.path.join(REPO, f))

# the frozen Stage 2a code, repacked without the two data files that DATA.md withholds
with zipfile.ZipFile(os.path.join(STUDY, "code_stage2a_frozen.zip")) as zin,         zipfile.ZipFile(os.path.join(REPO, "code_stage2a_frozen_nodata.zip"), "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        if item.filename not in ("data/items.json", "data/calibration.json"):
            zout.writestr(item, zin.read(item.filename))
# D9's frozen copies of the shared files (hashed in the D9 block), without their data folder
for f in os.listdir(os.path.join(STUDY, "dq_stage_d9")):
    if os.path.isfile(os.path.join(STUDY, "dq_stage_d9", f)):
        copy(os.path.join(STUDY, "dq_stage_d9", f), os.path.join(REPO, "dq_stage_d9", f))
# hashed pilot files that hold only labels and numbers (blind validation labels, pilot summaries, timings)
for rel in ["res/d1/pilot_judge_validation_assistant_labels.json", "res/d1/pilot_judge_validation2_assistant_labels.json",
            "res/d5/pilot_bt_validation_assistant_labels.json", "res/d7/pilot_blind_d7_assistant_labels.json",
            "res/d6/q3-1.7b_pilot_summary.json", "res/d9/q3-1.7b_pilot_cal.json", "res/d9/q3-1.7b_pilot_summary.json",
            "res/d9/q3-8b_timing.json"]:
    copy(os.path.join(STUDY, "code", rel), os.path.join(REPO, "code", rel))

code = os.path.join(STUDY, "code")
n_code = 0
for root, dirs, files in os.walk(code):
    dirs[:] = [d for d in dirs if d not in CODE_EXCLUDE_DIRS]
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), code)
        if f in CODE_EXCLUDE_FILES or f.endswith((".pt", ".npz", ".pyc", ".log")):
            continue
        if rel.startswith("data") and f in DATA_EXCLUDE:
            continue
        copy(os.path.join(root, f), os.path.join(REPO, "code", rel))
        n_code += 1

# results: per-run summaries (scrubbed), the aggregate, the exploratory strength analyses, the page
res = os.path.join(code, "res")
n_res = 0
for d in sorted(os.listdir(res)):
    p = os.path.join(res, d)
    if not os.path.isdir(p):
        continue
    for f in sorted(os.listdir(p)):
        if f.endswith("_main_summary.json"):
            s = scrub(json.load(open(os.path.join(p, f), encoding="utf-8")))
            os.makedirs(os.path.join(REPO, "results", d), exist_ok=True)
            json.dump(s, open(os.path.join(REPO, "results", d, f), "w", encoding="utf-8"), indent=1)
            n_res += 1
copy(os.path.join(res, "aggregate_main.json"), os.path.join(REPO, "results", "aggregate_main.json"))
explore = os.path.join(HERE, "explore")
if os.path.isdir(explore):
    for d in sorted(os.listdir(explore)):
        for f in sorted(os.listdir(os.path.join(explore, d))):
            copy(os.path.join(explore, d, f), os.path.join(REPO, "results", "exploratory", d, f))
copy(os.path.join(code, "page.html"), os.path.join(REPO, "results", "page.html"))

for f in PAPER:
    copy(os.path.join(HERE, f), os.path.join(REPO, "paper", "source", f))
copy(os.path.join(HERE, "main.pdf"), os.path.join(REPO, "paper", f"{NAME}.pdf"))
for f in ("README.md", "DATA.md", "LICENSE", "CITATION.cff", "requirements.txt", ".gitattributes"):
    copy(os.path.join(HERE, "release_files", f), os.path.join(REPO, f))

# Zenodo: the PDF, the repository as a zip, and (added separately) the per-item results zip
os.makedirs(ZEN, exist_ok=True)
copy(os.path.join(HERE, "main.pdf"), os.path.join(ZEN, f"{NAME}.pdf"))
zp = os.path.join(ZEN, f"{NAME}-source.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(REPO):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.join(NAME, os.path.relpath(full, REPO)))
print(f"repo: {n_code} code files, {n_res} summaries; zenodo: {os.path.getsize(zp) // 1024} KB source zip")
for f in sorted(os.listdir(ZEN)):
    print(" ", f, hashlib.md5(open(os.path.join(ZEN, f), "rb").read()).hexdigest())
