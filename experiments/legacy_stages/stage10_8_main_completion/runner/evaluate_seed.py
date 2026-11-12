#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib,json,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ALL=('cloud','haze','gaussian_noise','blur','jpeg','low_resolution','pixel_dropout','misregistration','radiometric_shift','mixed')
def clean_ids(xs,t): return [int(x) for x in xs if int(x) not in {t.start,t.end,t.pad}]
def save(x,p):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def score(root,refs,hyps):
 sys.path.insert(0,str(root));from eval_func.bleu.bleu import Bleu;from eval_func.rouge.rouge import Rouge;from eval_func.cider.cider import Cider
 g={i:[' '.join(map(str,x)) for x in r] for i,r in enumerate(refs)};h={i:[' '.join(map(str,x))] for i,x in enumerate(hyps)}
 b,_=Bleu(4).compute_score(g,h);r,_=Rouge().compute_score(g,h);c,_=Cider().compute_score(g,h)
 return {'BLEU-4':float(b[3]),'ROUGE-L':float(r),'CIDEr':float(c)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['change_agent','p3','p3_nomask'],required=True);p.add_argument('--degradation-seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 project=Path('__PROJECT_ROOT__');runner=project/('stage6_6_m2_backbones/runner' if a.model=='change_agent' else 'stage6_7_partial_hybrid/runner');sys.path.insert(0,str(runner));common=importlib.import_module('common');data=importlib.import_module('data');models=importlib.import_module('models')
 checkpoint=None
 if a.model=='change_agent':cfg=common.load_json(project/'stage6_6_m2_backbones/configs/ca_m2_geometry.json')
 else:
  path=project/('stage6_7_partial_hybrid/runs/p3_partial_relation/best.pt' if a.model=='p3' else 'stage10_3_p3_nomask/runs/p3_nomask/best.pt');checkpoint=torch.load(path,map_location='cpu',weights_only=False);cfg=dict(checkpoint['config'])
 cfg['severity']=2;cfg['degradation_seed']=a.degradation_seed;cfg['degradation_script']=str(project/'stage10_7_severity_curves/degradation/calibrated_v3_curve.py')
 common.seed_everything(int(cfg['seed']));device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');tokens=data.CaptionTokens(cfg['metadata_root'],41);model=models.build_model(cfg,tokens.vocab)
 if checkpoint:model.load_state_dict(checkpoint['trainable_state'],strict=False)
 model=model.to(device).eval();a.output.mkdir(parents=True,exist_ok=True);allm={}
 for kind in ALL:
  for side in ('A','B','both'):
   cond=f'{kind}__{side}';mp=a.output/f'{cond}.metrics.json';pp=a.output/f'{cond}.predictions.json'
   if mp.exists() and pp.exists():allm[cond]=json.loads(mp.read_text());continue
   ds=data.OnlineCaptionPairs(cfg,'test',tokens,training=False,condition=cond);loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda',collate_fn=data.collate_eval);refs=[];hyps=[];rows=[];st=time.time()
   with torch.inference_mode():
    for batch in loader:
     seqs=model.generate(batch['a'].to(device),batch['b'].to(device))
     for name,seq,rr in zip(batch['name'],seqs,batch['references']):
      pred=clean_ids(seq,tokens);rid=[clean_ids(tokens.encode(x)[0].tolist(),tokens) for x in rr];hyps.append(pred);refs.append(rid);rows.append({'name':name,'prediction_ids':pred,'reference_ids':rid})
   payload={'model':a.model,'condition':cond,'samples':len(rows),'severity':2,'degradation_seed':a.degradation_seed,**score(cfg['change_agent_root'],refs,hyps)};save(payload,mp);save(rows,pp);allm[cond]=payload;print(cond,f'{time.time()-st:.1f}s',flush=True)
 save(allm,a.output/'all_metrics.json')
if __name__=='__main__':main()
