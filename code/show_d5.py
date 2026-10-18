import json, sys
for m in sys.argv[1:]:
    s = json.load(open(f"res/d5/{m}_main_summary.json"))
    print(m, s["verdict"], "judgeA", s["checkA"].get("d_vs_judge_T"))
    for k, v in s["checkD"].items():
        auth = (v.get("authors_strength") or {}).get("rand") or {}
        print(" ", k, {f: (round(v[f]["effect"], 2), [round(c, 2) for c in v[f]["ci"]]) for f in ("rand", "perm") if v.get(f, {}).get("effect") is not None},
              v["verdict"], "| authors' strength vs rand:", round(auth.get("effect") or 0, 2), [round(c, 2) for c in auth.get("ci", [])])
