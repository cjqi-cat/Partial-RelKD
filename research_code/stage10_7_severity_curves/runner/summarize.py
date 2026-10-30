#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

MODELS=("change_agent","p3","p3_nomask"); LABELS={"change_agent":"Change-Agent","p3":"P3","p3_nomask":"P3-NoMask"}
METRICS=("BLEU-4","ROUGE-L","CIDEr"); SEEN={"cloud","haze","gaussian_noise","jpeg","pixel_dropout","radiometric_shift"};UNSEEN={"blur","low_resolution","misregistration"}
def load(p): return json.loads(p.read_text(encoding="utf-8"))
def mean(rows,m): return float(np.mean([x[m] for x in rows]))
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();out=a.root/"results";summary={}
 detail=[]
 for model in MODELS:
  clean=load(out/model/"clean"/"clean.metrics.json"); summary[model]={"clean":{m:clean[m] for m in METRICS},"severity":{}}
  for s in range(1,6):
   rows=list(load(out/model/f"severity_{s}"/"all_metrics.json").values())
   groups={"seen":[x for x in rows if x["condition"].split("__")[0] in SEEN],"unseen":[x for x in rows if x["condition"].split("__")[0] in UNSEEN],"mixed":[x for x in rows if x["condition"].startswith("mixed__")]}
   for side in ("A","B","both"):groups[f"unseen_{side}"]=[x for x in groups["unseen"] if x["condition"].endswith("__"+side)]
   summary[model]["severity"][str(s)]={g:{m:mean(rs,m) for m in METRICS} for g,rs in groups.items()}
   for x in rows: detail.append({"model":LABELS[model],"severity":s,"condition":x["condition"],**{m:x[m] for m in METRICS}})
  summary[model]["curve"]={}
  for g in ("seen","unseen","mixed","unseen_A","unseen_B","unseen_both"):
   summary[model]["curve"][g]={}
   for m in METRICS:
    ys=np.array([summary[model]["severity"][str(s)][g][m] for s in range(1,6)]);clean_v=summary[model]["clean"][m]
    summary[model]["curve"][g][m]={"auc_mean":float(ys.mean()),"s5":float(ys[-1]),"s5_drop":float(ys[-1]-clean_v),"s5_retention_percent":float(100*ys[-1]/clean_v),"slope":float(np.polyfit(np.arange(1,6),ys,1)[0])}
 (out/"severity_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
 with (out/"condition_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=detail[0].keys());w.writeheader();w.writerows(detail)
 lines=["# LEVIR-MCI五级退化强度曲线", "","> severity=2与原calibrated-v3逐像素一致；所有模型共享测试图像、算子、seed=20260928。AUC为S1–S5等权均值。",""]
 for group,title in (("seen","Seen"),("unseen","Unseen"),("mixed","Mixed"),("unseen_A","Unseen-A"),("unseen_B","Unseen-B"),("unseen_both","Unseen-both")):
  lines += [f"## {title}","","| Model | Metric | Clean | S1 | S2 | S3 | S4 | S5 | AUC | S5 Drop | S5 Retention | Slope |","|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for model in MODELS:
   for metric in METRICS:
    vals=[summary[model]["severity"][str(s)][group][metric] for s in range(1,6)];c=summary[model]["curve"][group][metric]
    lines.append(f"| {LABELS[model]} | {metric} | {summary[model]['clean'][metric]:.6f} | "+" | ".join(f"{x:.6f}" for x in vals)+f" | {c['auc_mean']:.6f} | {c['s5_drop']:+.6f} | {c['s5_retention_percent']:.2f}% | {c['slope']:+.6f} |")
  lines.append("")
 (out/"summary.md").write_text("\n".join(lines),encoding="utf-8")
 try:
  import matplotlib.pyplot as plt
  figdir=out/"figures";figdir.mkdir(exist_ok=True)
  colors={"change_agent":"#777777","p3":"#377eb8","p3_nomask":"#e41a1c"}
  for group in ("seen","unseen","mixed"):
   for metric in ("BLEU-4","CIDEr"):
    fig,ax=plt.subplots(figsize=(6,4));xs=np.arange(0,6)
    for model in MODELS:
     ys=[summary[model]["clean"][metric]]+[summary[model]["severity"][str(s)][group][metric] for s in range(1,6)]
     ax.plot(xs,ys,marker="o",linewidth=2,label=LABELS[model],color=colors[model])
    ax.set(xlabel="Severity (0 = clean)",ylabel=metric,title=f"{group.capitalize()} {metric}",xticks=xs);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(figdir/f"{group}_{metric.replace('-','').lower()}.png",dpi=220);plt.close(fig)
 except ImportError: print("matplotlib unavailable; tables were still generated")
 print("SUMMARY_OK")
if __name__=="__main__":main()
