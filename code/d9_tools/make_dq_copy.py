"""Stage the D9 run folder (jspace/paper6_d9): the shared files as they are in paper6/code, with
D9's changes applied by exact string replacement (each must match once), plus D9's own files.
python -I make_dq_copy.py <paper6/code dir> <stage dir>
The changes are the ones listed in INTEGRATION_D9.md."""
import hashlib
import json
import os
import shutil
import sys

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(DST, "data"), exist_ok=True)


def patch(name, reps):
    s = open(os.path.join(SRC, name), encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in s else "\n"  # keep the file's line endings
    for old, new in reps:
        old, new = old.replace("\n", nl), new.replace("\n", nl)
        assert s.count(old) == 1, (name, old[:60], s.count(old))
        s = s.replace(old, new)
    open(os.path.join(DST, name), "w", encoding="utf-8", newline="").write(s)


RUN = [
    ('SPECS["d8"] = D8\n',
     'SPECS["d8"] = D8\nfrom spec_d9 import D9; SPECS["d9"] = D9  # noqa: E402,E702\n'),
    ('    if DIR == "d5":  # oracle from the oracle chains\' sentences: next sentence backtracks or not\n',
     '    if DIR in ("d5", "d9"):  # oracle from the oracle chains\' sentences (D5: next sentence backtracks; D9: step TTS)\n'),
]
ANALYZE = [
    ('             d8=json.load(open(os.path.join(HERE, "data", "items_d8.json"), encoding="utf-8")))\n',
     '             d8=json.load(open(os.path.join(HERE, "data", "items_d8.json"), encoding="utf-8")),\n'
     '             d9=json.load(open(os.path.join(HERE, "data", "items_d9.json"), encoding="utf-8")))\n'),
    ('def T_row(rec, j, row, target):\n',
     'def T_row(rec, j, row, target):\n'
     '    if DIR == "d9":  # the early-exit answer follows the perturbed step (differs from y*)\n'
     '        return row["gen"]["F"]\n'),
    ('elif DIR == "d5":  # SESOI: a 25% relative change of the clean rate (§3)\n',
     'elif DIR == "d9":  # drive: rate at which the answer follows a perturbed step, SESOI 10 points (§0.5)\n'
     '    DRIVE = [("F", 0.10)]\n'
     '    BINARY = False  # tracking target (TTS) is continuous\n'
     'elif DIR == "d5":  # SESOI: a 25% relative change of the clean rate (§3)\n'),
    ('else:\n    A = [byid[i] for i in A_ids if i in byid and T_clean(byid[i]) is not None]\n',
     'elif DIR == "d9":  # one Check A unit per numeric step of the model\'s own chain; covariates: position, step length\n'
     '    A = [dict(sj, id=f"{i}:{sj[\'j\']}", chain=i) for i in A_ids if i in byid for sj in byid[i]["sentences"]]\n'
     '    T = np.array([a["T"] for a in A], dtype=float)\n'
     '    nprompt = np.array([[a["pos"], a["step_len"]] for a in A], dtype=float)\n'
     '    GROUPS = [a["chain"] for a in A]\n'
     'else:\n    A = [byid[i] for i in A_ids if i in byid and T_clean(byid[i]) is not None]\n'),
    ('        if DIR == "d1" and (sign == 1) != (it["label"] == 0):\n            continue\n',
     '        if DIR == "d1" and (sign == 1) != (it["label"] == 0):\n            continue\n'
     '        if DIR == "d9" and (sign == 1) != (rec["rows"][0]["gen"]["F"] == 0):  # +d where the clean answer\n'
     '            continue  # stays y* (engagement), -d where it follows the perturbed step (disengagement)\n'),
    ('elif DIR == "d5":\n    reproduced = (c0["kw_authors_strength"]',
     'elif DIR == "d9":  # Stage 2b D9 §5: released-vector cosine (R1 models) and the engagement / disengagement tests\n'
     '    adv = [c0.get(t, {}).get("adv_d_vs_rand") for t in ("ET", "DT")]\n'
     '    reproduced = (c0.get("cos_released") is None or c0["cos_released"] >= gate["cos_released_min"]) and \\\n'
     '                 any(a is not None and a >= gate["flip_adv_min"] for a in adv)\n'
     'elif DIR == "d5":\n    reproduced = (c0["kw_authors_strength"]'),
]
patch("run.py", RUN)
patch("analyze.py", ANALYZE)
st = json.load(open(os.path.join(SRC, "stage2_settings.json"), encoding="utf-8"))
st["check0a"]["d9"] = json.load(open(os.path.join(SRC, "d9_tools", "stage2_settings_d9.json"), encoding="utf-8"))
json.dump(st, open(os.path.join(DST, "stage2_settings.json"), "w", encoding="utf-8"), indent=1)
for f in ("p6lib.py", "specs.py", "spec_d5.py", "spec_d4.py", "spec_d8.py", "spec_d9.py", "d9_clean.py", "cr4_parsers.py",
          "config_pilot.json", "config_main.json"):
    shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
for f in ("items.json", "calibration.json", "items_d5.json", "items_d4g.json", "items_d8.json", "items_d9.json"):
    shutil.copy(os.path.join(SRC, "data", f), os.path.join(DST, "data", f))
for root, _, files in os.walk(DST):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.relpath(p, DST))
