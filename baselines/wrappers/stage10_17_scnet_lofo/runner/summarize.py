import argparse
from pathlib import Path
from common import SEEN,UNSEEN,load_json,save_json
M=('BLEU-4','ROUGE-L','CIDEr')
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args(); rows=load_json(Path(a.input)/'all_metrics.json')
def mean(k,m): return sum(v[m] for n,v in rows.items() if '__' in n and n.split('__')[0] in k)/sum(1 for n in rows if '__' in n and n.split('__')[0] in k)
r={'model':'SCNet','protocol':'geometry-quality LOFO'}
for m in M:
    r['Clean '+m]=rows['clean'][m]; r['Seen '+m]=mean(SEEN,m); r['Unseen '+m]=mean(UNSEEN,m); r['Mixed '+m]=mean(('mixed',),m)
save_json(r,Path(a.output).with_suffix('.json')); line='| SCNet | '+' | '.join(f"{r[g+' '+m]:.6f}" for g in ('Clean','Seen','Unseen','Mixed') for m in M)+' |'; text='# SCNet — LEVIR-MCI geometry-quality LOFO\n\n| Model | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr | Mixed B4 | Mixed R-L | Mixed CIDEr |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'+line+'\n'; Path(a.output).write_text(text); print(text); print('SUMMARY_OK')
