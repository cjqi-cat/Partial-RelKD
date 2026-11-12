#!/usr/bin/env python3
from __future__ import annotations
import importlib,json,sys,time
from pathlib import Path
import torch

P=Path('__PROJECT_ROOT__');OUT=P/'stage10_8_main_completion/results/efficiency';OUT.mkdir(parents=True,exist_ok=True)
def one(name):
 runner=P/('stage6_6_m2_backbones/runner' if name=='change_agent' else 'stage6_7_partial_hybrid/runner');sys.path.insert(0,str(runner));common=importlib.import_module('common');data=importlib.import_module('data');models=importlib.import_module('models')
 if name=='change_agent':cfg=common.load_json(P/'stage6_6_m2_backbones/configs/ca_m2_geometry.json');ck=None
 else:path=P/('stage6_7_partial_hybrid/runs/p3_partial_relation/best.pt' if name=='p3' else 'stage10_3_p3_nomask/runs/p3_nomask/best.pt');ck=torch.load(path,map_location='cpu',weights_only=False);cfg=ck['config']
 tok=data.CaptionTokens(cfg['metadata_root'],41);m=models.build_model(cfg,tok.vocab);m.load_state_dict(ck['trainable_state'],strict=False) if ck else None;m=m.cuda().eval();ds=data.OnlineCaptionPairs(cfg,'test',tok,training=False,condition='clean');x=ds[0];a=x['a'][None].cuda();b=x['b'][None].cuda()
 with torch.inference_mode():
  for _ in range(5):m.generate(a,b)
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();st=time.perf_counter()
  for _ in range(20):m.generate(a,b)
  torch.cuda.synchronize();elapsed=(time.perf_counter()-st)/20
 return {'model':name,'parameters_m':sum(x.numel() for x in m.parameters())/1e6,'checkpoint_parameters_m':sum(x.numel() for x in m.parameters() if x.requires_grad)/1e6,'latency_ms_batch1':elapsed*1000,'peak_memory_mb':torch.cuda.max_memory_allocated()/2**20}
def main():
 rows=[]
 for n in ('change_agent','p3','p3_nomask'):
  for key in list(sys.modules):
   if key in {'common','data','models'}:del sys.modules[key]
  rows.append(one(n))
 (OUT/'efficiency.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');lines=['# 推理效率（单卡，batch=1，greedy）','','| Model | Params(M) | Trainable(M) | Latency(ms) | Peak memory(MB) |','|---|---:|---:|---:|---:|'];
 for r in rows:lines.append(f"| {r['model']} | {r['parameters_m']:.2f} | {r['checkpoint_parameters_m']:.2f} | {r['latency_ms_batch1']:.2f} | {r['peak_memory_mb']:.1f} |")
 (OUT/'efficiency.md').write_text('\n'.join(lines),encoding='utf-8');print('EFFICIENCY_OK')
if __name__=='__main__':main()
