#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ORDER=("change_agent","raw","control","full","lora","partial_last2")
LABELS={"change_agent":"Original Change-Agent","raw":"Raw DINOv2","control":"Control Adapter","full":"DINO Full Adapter","lora":"QKV-LoRA","partial_last2":"Partial-last2"}
METRICS=("Bleu_4","ROUGE_L","CIDEr")


def values(payload):
    src=payload.get("metrics",payload)
    return {"Bleu_4":float(src.get("Bleu_4",src.get("BLEU_4"))),"ROUGE_L":float(src["ROUGE_L"]),"CIDEr":float(src["CIDEr"])}


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); a=p.parse_args(); out=a.root/"analysis"; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for variant in ORDER:
        base=a.root/"baseline/change_agent" if variant=="change_agent" else a.root/"evaluation/results"/variant
        paths=sorted(base.glob("*/metrics.json"))
        if len(paths)!=31: raise RuntimeError(f"{variant}: expected 31 metrics, found {len(paths)}")
        for path in paths:
            data=json.loads(path.read_text(encoding="utf-8")); count=int(data.get("samples",data.get("sample_count",0)))
            if count!=1929: raise RuntimeError(f"{variant}/{path.parent.name}: samples={count}")
            rows.append({"variant":variant,"label":LABELS[variant],"condition":data.get("condition",path.parent.name),"samples":count,**values(data)})
    write_csv(out/"six_model_all_186_metrics.csv",rows)
    table=defaultdict(dict)
    for row in rows: table[row["variant"]][row["condition"]]=row
    summary=[]
    for variant in ORDER:
        clean=table[variant]["clean"]; degraded=[r for c,r in table[variant].items() if c!="clean"]
        item={"variant":variant,"label":LABELS[variant],"conditions":31}
        for metric in METRICS:
            mean=statistics.fmean(r[metric] for r in degraded)
            item[f"clean_{metric}"]=clean[metric]; item[f"degraded_mean_{metric}"]=mean
            item[f"relative_drop_pct_{metric}"]=100*(clean[metric]-mean)/clean[metric]
        summary.append(item)
    write_csv(out/"six_model_summary.csv",summary)
    wins=[]
    for variant in ORDER:
        item={"variant":variant,"label":LABELS[variant]}
        for metric in METRICS:
            item[metric]=sum(max(table[v][c][metric] for v in ORDER)==table[variant][c][metric] for c in table[variant] if c!="clean")
        wins.append(item)
    write_csv(out/"winner_counts.csv",wins)
    lines=["# Calibrated-v3六模型正式对比","","统一使用LEVIR-MCI test 1,929对，calibrated-v3在线退化，seed=20260928；五个DINO路线重新训练，Original Change-Agent使用同像素基线结果。","","| 模型 | clean B4 | 退化均值 B4 | clean R-L | 退化均值 R-L | clean CIDEr | 退化均值 CIDEr |","|---|---:|---:|---:|---:|---:|---:|"]
    for x in summary: lines.append(f"| {x['label']} | {x['clean_Bleu_4']:.5f} | {x['degraded_mean_Bleu_4']:.5f} | {x['clean_ROUGE_L']:.5f} | {x['degraded_mean_ROUGE_L']:.5f} | {x['clean_CIDEr']:.5f} | {x['degraded_mean_CIDEr']:.5f} |")
    lines += ["","## 相对clean下降","","| 模型 | BLEU-4 | ROUGE-L | CIDEr |","|---|---:|---:|---:|"]
    for x in summary: lines.append(f"| {x['label']} | {x['relative_drop_pct_Bleu_4']:.2f}% | {x['relative_drop_pct_ROUGE_L']:.2f}% | {x['relative_drop_pct_CIDEr']:.2f}% |")
    (out/"六模型_calibrated_v3_对比报告.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest={"status":"complete","models":list(ORDER),"metric_files":len(rows),"expected":186,"samples_per_condition":1929,"profile":"calibrated_v3","test_seed":20260928}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=="__main__": main()
