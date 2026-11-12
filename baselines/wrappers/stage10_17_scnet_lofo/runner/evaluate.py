from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from common import conditions,save_json,seed_everything
from data import CaptionPairs,CaptionTokens,collate_eval
from metrics import score
from model import build_model
def clean(ids,tok):
    out=[]
    for x in ids:
        x=int(x)
        if x==tok.end: break
        if x not in (tok.start,tok.pad): out.append(x)
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True); p.add_argument('--conditions',default='all'); a=p.parse_args()
    st=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=st['config']; seed_everything(cfg['seed']); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok=CaptionTokens(cfg['metadata_root'],cfg['max_length']); model=build_model(cfg,tok.vocab); model.load_state_dict(st['model_state']); model.to(dev).eval(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); allm={}
    requested=conditions() if a.conditions=='all' else a.conditions.split(',')
    for cond in requested:
        ds=CaptionPairs(cfg,'test',tok,False,cond); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=cfg['num_workers'],pin_memory=True,collate_fn=collate_eval); refs=[]; hyps=[]; rows=[]
        with torch.inference_mode():
            for i,b in enumerate(dl,1):
                pred=clean(model.greedy(b['a'].to(dev),b['b'].to(dev)).cpu().tolist(),tok); rr=[clean(tok.encode(x)[0],tok) for x in b['references'][0]]; refs.append(rr); hyps.append(pred); rows.append({'name':b['name'][0],'prediction_ids':pred,'reference_ids':rr})
                if i%100==0: print(f'{cond} {i}/{len(dl)}',flush=True)
        payload={'model':'SCNet','protocol':'geometry_quality_LOFO','condition':cond,'samples':len(rows),'severity':cfg['severity'],**score(cfg['change_agent_root'],refs,hyps)}; save_json(payload,out/f'{cond}.metrics.json'); save_json(rows,out/f'{cond}.predictions.json'); allm[cond]=payload; print(json.dumps(payload),flush=True)
    save_json(allm,out/'all_metrics.json'); print('EVALUATION_OK')
if __name__=='__main__': main()
