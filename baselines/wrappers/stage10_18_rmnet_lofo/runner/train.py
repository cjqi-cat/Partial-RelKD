from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from common import atomic_save, load_json, seed_everything
from data import CaptionPairs, CaptionTokens, collate_eval
from metrics import score
from model import build_model

def clean(ids,tok):
    out=[]
    for x in ids:
        x=int(x)
        if x==tok.end: break
        if x not in (tok.start,tok.pad): out.append(x)
    return out
def loss_fn(output):
    scores,captions,lengths,_=output; targets=captions[:,1:]
    scores=pack_padded_sequence(scores,lengths,batch_first=True,enforce_sorted=True).data
    targets=pack_padded_sequence(targets,lengths,batch_first=True,enforce_sorted=True).data
    return torch.nn.functional.cross_entropy(scores.float(),targets)
def validate(model,loader,tok,cfg,device):
    model.eval(); refs=[]; hyps=[]
    with torch.inference_mode():
        for batch in loader:
            seq=model.greedy(batch['a'].to(device),batch['b'].to(device)).cpu().tolist()
            hyps.append(clean(seq,tok)); refs.append([clean(tok.encode(x)[0],tok) for x in batch['references'][0]])
    return score(cfg['change_agent_root'],refs,hyps)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--output',required=True); ap.add_argument('--resume'); a=ap.parse_args()
    cfg=load_json(a.config); seed_everything(cfg['seed']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('CUDA required')
    tok=CaptionTokens(cfg['metadata_root'],cfg['max_length']); model=build_model(cfg,tok.vocab).to(device)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=cfg.get('weight_decay',0.0))
    start=0; best=-1.; bad=0
    if a.resume:
        st=torch.load(a.resume,map_location='cpu',weights_only=False); model.load_state_dict(st['model_state']); opt.load_state_dict(st['optimizer_state']); start=st['epoch']+1; best=st['best_b4']; bad=st.get('bad_epochs',0)
    trds=CaptionPairs(cfg,'train',tok,True); vads=CaptionPairs(cfg,'val',tok,False,'clean')
    tr=DataLoader(trds,batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'],pin_memory=True,drop_last=True)
    va=DataLoader(vads,batch_size=1,shuffle=False,num_workers=cfg['num_workers'],pin_memory=True,collate_fn=collate_eval)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for epoch in range(start,cfg['epochs']):
        tic=time.time(); trds.set_epoch(epoch); model.train(); total=0.; count=0
        for batch in tr:
            opt.zero_grad(set_to_none=True); aa=batch['a'].to(device); bb=batch['b'].to(device); tokens=batch['tokens'].to(device); lengths=batch['length'].to(device)
            loss=loss_fn(model(aa,bb,tokens,lengths)); loss.backward(); clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['grad_clip']); opt.step(); total+=float(loss.detach())*aa.size(0); count+=aa.size(0)
        metrics=validate(model,va,tok,cfg,device); improved=metrics['BLEU-4']>best
        if improved: best=metrics['BLEU-4']; bad=0
        else: bad+=1
        rec={'epoch':epoch,'train_loss':total/count,'val':metrics,'seconds':time.time()-tic}; print(rec,flush=True)
        with (out/'run.log').open('a') as f: f.write(json.dumps(rec)+'\n')
        state={'config':cfg,'epoch':epoch,'best_b4':best,'bad_epochs':bad,'model_state':model.state_dict(),'optimizer_state':opt.state_dict()}
        atomic_save(state,out/'last.pt')
        if improved: atomic_save(state,out/'best.pt')
        if bad>=cfg['patience']: break
    print('TRAINING_OK')
if __name__=='__main__': main()
