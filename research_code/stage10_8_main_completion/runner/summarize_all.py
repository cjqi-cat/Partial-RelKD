#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

M=('BLEU-4','ROUGE-L','CIDEr');SIDES=('A','B','both');ALL=('cloud','haze','gaussian_noise','blur','jpeg','low_resolution','pixel_dropout','misregistration','radiometric_shift','mixed')
F={'atmospheric':('cloud','haze'),'geometry_quality':('blur','low_resolution','misregistration'),'noise_loss':('gaussian_noise','jpeg','pixel_dropout')}
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def avg(d,conds):return {m:float(np.mean([d[c][m] for c in conds])) for m in M}
def fmt(x):return f'{x:.6f}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();project=a.root.parent;out=a.root/'results';out.mkdir(exist_ok=True)
 lines=['# Change Caption主线补全实验总表','','## 三个完整留出家族','','| Family | Model | Clean B4 | Clean R-L | Clean CIDEr | Seen B4 | Seen R-L | Seen CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for family,unseen_types in F.items():
  held=set(unseen_types)|{'mixed'};seen_types=[x for x in ALL if x not in held];seen=[f'{x}__{s}' for x in seen_types for s in SIDES];unseen=[f'{x}__{s}' for x in unseen_types for s in SIDES]
  roots={}
  if family=='geometry_quality':
   roots={'Change-Agent':project/'stage10_7_severity_curves/results/change_agent/severity_2','DINO-Aug':project/'stage8_2_dino_controls/results/LEVIR_MCI/dino_aug','P3':project/'stage10_7_severity_curves/results/p3/severity_2','P3-NoMask':project/'stage10_7_severity_curves/results/p3_nomask/severity_2'}
   cleanroots={k:(project/'stage10_7_severity_curves/results'/('change_agent' if k=='Change-Agent' else 'p3' if k=='P3' else 'p3_nomask')/'clean' if k!='DINO-Aug' else v) for k,v in roots.items()}
  else:
   base=out/'lofo'/family;roots={'Change-Agent':project/'stage10_7_severity_curves/results/change_agent/severity_2','DINO-Aug':base/'dino_aug','P3':base/'p3','P3-NoMask':base/'p3_nomask'};cleanroots=dict(roots);cleanroots['Change-Agent']=project/'stage10_7_severity_curves/results/change_agent/clean'
  for model,r in roots.items():
   if not (r/'all_metrics.json').exists():continue
   d=load(r/'all_metrics.json');cr=cleanroots[model];cd=load(cr/'all_metrics.json') if (cr/'all_metrics.json').exists() else d;clean=cd['clean'] if 'clean' in cd else load(cr/'clean.metrics.json');sv=avg(d,seen);uv=avg(d,unseen)
   lines.append(f"| {family} | {model} | {fmt(clean['BLEU-4'])} | {fmt(clean['ROUGE-L'])} | {fmt(clean['CIDEr'])} | {fmt(sv['BLEU-4'])} | {fmt(sv['ROUGE-L'])} | {fmt(sv['CIDEr'])} | {fmt(uv['BLEU-4'])} | {fmt(uv['ROUGE-L'])} | {fmt(uv['CIDEr'])} |")
 lines += ['','## P3-NoMask训练随机种子稳定性','','| Seed | Clean B4 | Clean R-L | Clean CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr |','|---:|---:|---:|---:|---:|---:|---:|']
 seedroots={20261008:project/'stage10_7_severity_curves/results/p3_nomask/severity_2',20261018:out/'multiseed/geometry_seed20261018',20261028:out/'multiseed/geometry_seed20261028'};vals=[]
 for seed,r in seedroots.items():
  if not (r/'all_metrics.json').exists():continue
  d=load(r/'all_metrics.json');clean=(load(project/'stage10_7_severity_curves/results/p3_nomask/clean/clean.metrics.json') if seed==20261008 else d['clean']);u=avg(d,[f'{x}__{s}' for x in F['geometry_quality'] for s in SIDES]);vals.append([clean[m] for m in M]+[u[m] for m in M]);lines.append(f"| {seed} | "+' | '.join(fmt(x) for x in vals[-1])+' |')
 if vals:
  ar=np.asarray(vals);lines.append(f"| mean±std | "+' | '.join(f'{ar[:,i].mean():.6f}±{ar[:,i].std(ddof=1):.6f}' for i in range(ar.shape[1]))+' |')
 lines += ['','## 测试退化随机种子稳定性（severity=2）','','| Model | Group | Metric | Mean | Std | Seeds |','|---|---|---|---:|---:|---:|']
 seed_values=(20260928,20261028,20261128)
 for model in ('change_agent','p3','p3_nomask'):
  ds=[]
  for seed in seed_values:
   r=(project/'stage10_7_severity_curves/results'/model/'severity_2') if seed==20260928 else (out/f'test_seeds/seed{seed}/{model}')
   if (r/'all_metrics.json').exists():ds.append(load(r/'all_metrics.json'))
  for group,types in [('seen',('cloud','haze','gaussian_noise','jpeg','pixel_dropout','radiometric_shift')),('unseen',F['geometry_quality']),('mixed',('mixed',))]:
   conds=[f'{x}__{s}' for x in types for s in SIDES]
   for metric in M:
    vv=[avg(d,conds)[metric] for d in ds]
    if vv:lines.append(f"| {model} | {group} | {metric} | {np.mean(vv):.6f} | {np.std(vv,ddof=1) if len(vv)>1 else 0:.6f} | {len(vv)} |")
 lines += ['','## 目标数据集完整监督：P3-NoMask','','| Dataset | Model | Clean B4 | Clean R-L | Clean CIDEr | Unseen B4 | Unseen R-L | Unseen CIDEr |','|---|---|---:|---:|---:|---:|---:|---:|']
 for dataset in ('SECOND_CC','WHU_CDC'):
  existing=project/'stage8_1_multidataset_full/results'/dataset
  model_roots={'Change-Agent':existing/'change_agent','P3':existing/'p3','P3-NoMask':out/'cross'/dataset/'p3_nomask'}
  for model,r in model_roots.items():
   if not (r/'all_metrics.json').exists():continue
   d=load(r/'all_metrics.json');clean=d['clean'];u=avg(d,[f'{x}__{s}' for x in F['geometry_quality'] for s in SIDES]);lines.append(f"| {dataset} | {model} | {clean['BLEU-4']:.6f} | {clean['ROUGE-L']:.6f} | {clean['CIDEr']:.6f} | {u['BLEU-4']:.6f} | {u['ROUGE-L']:.6f} | {u['CIDEr']:.6f} |")
 lines += ['','## 说明','','- Change-Agent在不同LOFO家族中不训练，因此其同一31条件预测按各家族重新分组即可；不会人为重复训练。','- atmospheric/noise-loss只用各自允许退化训练，mixed随留出家族排除。','- SECOND-CC和WHU-CDC结果来自各目标数据集独立监督训练。','']
 (out/'main_completion_summary.md').write_text('\n'.join(lines),encoding='utf-8');print('SUMMARY_OK')
if __name__=='__main__':main()
