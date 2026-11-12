#!/usr/bin/env python3
import json, statistics
from pathlib import Path

P=Path("__PROJECT_ROOT__"); R=P/"stage8_4_levir_multiseed_server"; OUT=R/"results/summary"
METRICS=("BLEU-4","ROUGE-L","CIDEr")
SEEN=[f"{d}__{s}" for d in ("cloud","haze","gaussian_noise","jpeg","pixel_dropout","radiometric_shift") for s in ("A","B","both")]
UNSEEN=[f"{d}__{s}" for d in ("blur","low_resolution","misregistration") for s in ("A","B","both")]
sources={20261008:{"DINO-Aug":P/"stage8_2_dino_controls/results/LEVIR_MCI/dino_aug/all_metrics.json","P3":P/"stage6_7_partial_hybrid/results/p3_partial_relation/all_metrics.json"}}
for seed in (20261018,20261028):
    sources[seed]={m:R/f"results/seed_{seed}/{d}/all_metrics.json" for m,d in (("DINO-Aug","dino_aug"),("P3","p3"))}
def agg(path):
    x=json.loads(path.read_text()); out={}
    for g,cs in (("Clean",["clean"]),("Seen",SEEN),("Unseen",UNSEEN)):
        for m in METRICS: out[f"{g} {m}"]=sum(float(x[c][m]) for c in cs)/len(cs)
    return out
rows=[{"seed":s,"model":m,**agg(p)} for s,ms in sources.items() for m,p in ms.items()]
OUT.mkdir(parents=True,exist_ok=True); (OUT/"per_seed.json").write_text(json.dumps(rows,indent=2))
lines=["# Three-seed stability","","Seeds: 20261008, 20261018, 20261028. Values are mean ± sample standard deviation.",""]
for model in ("DINO-Aug","P3"):
    lines += [f"## {model}",""]
    for key in [f"{g} {m}" for g in ("Clean","Seen","Unseen") for m in METRICS]:
        v=[r[key] for r in rows if r["model"]==model]
        lines.append(f"- {key}: {statistics.mean(v):.6f} ± {statistics.stdev(v):.6f}")
    lines.append("")
(OUT/"summary.md").write_text("\n".join(lines),encoding="utf-8"); print("SUMMARY_OK")
