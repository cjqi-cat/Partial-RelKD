import argparse
from pathlib import Path
from common import load_json, save_json

SEEN=("cloud","haze","gaussian_noise","jpeg","pixel_dropout","radiometric_shift")
UNSEEN=("blur","low_resolution","misregistration")
METRICS=("BLEU-4","ROUGE-L","CIDEr")


def avg(rows,kinds,key):
    values=[v[key] for name,v in rows.items() if name.split("__")[0] in kinds and "__" in name]
    return sum(values)/len(values)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results",required=True); args=ap.parse_args()
    root=Path(args.results); table=[]
    for model_dir in sorted(x for x in root.iterdir() if x.is_dir()):
        p=model_dir/"all_metrics.json"
        if not p.exists(): continue
        rows=load_json(p); item={"Model":model_dir.name}
        for group,kinds in (("Clean",None),("Seen",SEEN),("Unseen",UNSEEN)):
            for metric in METRICS: item[f"{group} {metric}"]=rows["clean"][metric] if kinds is None else avg(rows,kinds,metric)
        table.append(item)
    save_json(table,root/"summary.json")
    cols=["Model"]+[f"{g} {m}" for g in ("Clean","Seen","Unseen") for m in METRICS]
    lines=["# Stage10 权威Change Caption基线（LEVIR-MCI geometry-quality LOFO）","", "|"+"|".join(cols)+"|", "|"+"|".join(["---"]+["---:"]*(len(cols)-1))+"|"]
    for x in table: lines.append("|"+"|".join([x["Model"]]+[f"{x[c]:.6f}" for c in cols[1:]])+"|")
    (root/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines)); print("SUMMARY_OK")
if __name__=="__main__": main()
