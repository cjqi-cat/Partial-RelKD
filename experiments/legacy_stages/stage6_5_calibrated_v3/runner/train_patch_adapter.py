#!/usr/bin/env python3
"""Train full/control frozen-DINO patch adapters from one shared feature cache."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

VIEWS = ("clean_a", "deg1_a", "deg2_a", "clean_b", "deg1_b", "deg2_b")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class PatchDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.arrays = [np.load(root / f"{split}_{key}.npy", mmap_mode="r") for key in VIEWS]
        self.masks = np.load(root / f"{split}_masks.npy", mmap_mode="r")
        meta = np.load(root / f"{split}_meta.npz")
        self.labels, self.kinds = meta["labels"], meta["kinds"]

    def __len__(self): return len(self.labels)

    def __getitem__(self, index):
        views = tuple(torch.from_numpy(np.array(x[index], dtype=np.float32, copy=True)) for x in self.arrays)
        return views + (
            torch.from_numpy(np.array(self.masks[index], dtype=np.float32, copy=True)),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.from_numpy(np.array(self.kinds[index], dtype=np.int64, copy=True)),
        )


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale): ctx.scale = scale; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.scale * grad, None


class PatchModel(nn.Module):
    def __init__(self, dim, hidden, scale):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.scale = scale
        nn.init.zeros_(self.adapter[-1].weight); nn.init.zeros_(self.adapter[-1].bias)
        self.change_head = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.degradation_head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 10))

    def encode(self, x): return F.normalize(x + self.scale * self.adapter(self.norm(x)), dim=-1)
    def change_logits(self, a, b): return self.change_head(torch.cat([torch.abs(b-a), a*b], -1)).squeeze(-1)


def nce(a, b, temperature):
    logits = a @ b.T / temperature
    target = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def compute(model, batch, cfg):
    *views, mask, pair_label, kinds = batch
    z = [model.encode(x) for x in views]
    ca, d1a, d2a, cb, d1b, d2b = z
    selected = torch.randperm(ca.shape[1], device=ca.device)[:min(int(cfg["patches_per_image"]), ca.shape[1])]
    pairs = ((ca,d1a),(ca,d2a),(d1a,d2a),(cb,d1b),(cb,d2b),(d1b,d2b))
    patch_inv = sum((1-(x[:,selected]*y[:,selected]).sum(-1)).mean() for x,y in pairs) / len(pairs)
    global_inv = sum(nce(F.normalize(x.mean(1),dim=-1), F.normalize(y.mean(1),dim=-1), float(cfg["temperature"])) for x,y in pairs) / len(pairs)
    invariance = patch_inv + global_inv
    distance = 1 - (ca*cb).sum(-1); changed = mask > 0.5; unchanged = ~changed
    same = distance[unchanged].pow(2).mean() if unchanged.any() else distance.mean()*0
    changed_loss = F.relu(float(cfg["change_margin"])-distance[changed]).pow(2).mean() if changed.any() else distance.mean()*0
    change = 0.5 * (same + changed_loss)
    logits = model.change_logits(ca, cb)
    pos = mask.sum().clamp_min(1); neg = (1-mask).sum().clamp_min(1)
    patch_cls = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=(neg/pos).clamp(max=20))
    distill = sum((1-(adapted*F.normalize(raw,dim=-1)).sum(-1)).mean() for adapted,raw in zip(z,views))/len(z)
    adv = []
    for feature,label in ((d1a,kinds[:,1]),(d2a,kinds[:,2]),(d1b,kinds[:,4]),(d2b,kinds[:,5])):
        adv.append(F.cross_entropy(model.degradation_head(GradientReverse.apply(feature.mean(1),1.0)),label))
    adversarial = sum(adv)/len(adv)
    total = (
        float(cfg.get("lambda_invariance",1.0))*invariance
        + float(cfg["lambda_change"])*change
        + float(cfg["lambda_patch_cls"])*patch_cls
        + float(cfg["lambda_distill"])*distill
        + float(cfg["lambda_adversarial"])*adversarial
    )
    return total, {"total":total,"inv":invariance,"change":change,"patch_cls":patch_cls,"distill":distill,"adversarial":adversarial}, logits, distance, mask, pair_label


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")


def run_epoch(model, loader, cfg, device, optimizer=None, scaler=None):
    training = optimizer is not None; model.train(training)
    sums = {k:0.0 for k in ("total","inv","change","patch_cls","distill","adversarial")}
    count=0; ys=[]; scores=[]; pair_y=[]; pair_scores=[]
    amp = bool(cfg.get("amp",False)) and device.startswith("cuda")
    for batch in loader:
        batch=[x.to(device,non_blocking=True) for x in batch]
        if training: optimizer.zero_grad(set_to_none=True)
        context=torch.enable_grad() if training else torch.no_grad()
        with context:
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                total,parts,logits,distance,mask,pairs=compute(model,batch,cfg)
        if training:
            scaler.scale(total).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(optimizer); scaler.update()
        n=len(pairs); count+=n
        for key in sums: sums[key]+=float(parts[key].detach())*n
        ys.append(mask.detach().cpu().numpy().ravel()); scores.append(logits.detach().sigmoid().cpu().numpy().ravel())
        pair_y.append(pairs.detach().cpu().numpy()); pair_scores.append(distance.detach().topk(max(1,distance.shape[1]//10),dim=1).values.mean(1).cpu().numpy())
    y=np.concatenate(ys); score=np.concatenate(scores); py=np.concatenate(pair_y); ps=np.concatenate(pair_scores)
    out={k:v/count for k,v in sums.items()}; out["patch_auc"]=safe_auc(y,score); out["pair_top10_auc"]=safe_auc(py,ps)
    return out


def save_checkpoint(payload, path):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".pt.tmp"); torch.save(payload,tmp); tmp.replace(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--features",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); cfg=json.loads(a.config.read_text(encoding="utf-8")); device=cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    for d in (a.output/"logs",a.output/"patch_checkpoints",a.output/"patch_results"): d.mkdir(parents=True,exist_ok=True)
    seed_everything(int(cfg["seed"])); train=PatchDataset(a.features,"train"); val=PatchDataset(a.features,"val")
    kw={"batch_size":int(cfg["batch_size_train"]),"num_workers":int(cfg.get("num_workers",0)),"pin_memory":device.startswith("cuda")}
    gen=torch.Generator().manual_seed(int(cfg["seed"])); train_loader=DataLoader(train,shuffle=True,generator=gen,**kw); val_loader=DataLoader(val,shuffle=False,**kw)
    model=PatchModel(768,int(cfg["adapter_hidden"]),float(cfg["adapter_scale"])).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg["learning_rate"]),weight_decay=float(cfg["weight_decay"])); scaler=torch.amp.GradScaler("cuda",enabled=device.startswith("cuda") and bool(cfg.get("amp",False)))
    best=-math.inf; stale=0; rows=[]
    log=(a.output/"logs"/"train.log").open("a",encoding="utf-8",buffering=1)
    for epoch in range(1,int(cfg["epochs"])+1):
        tr=run_epoch(model,train_loader,cfg,device,optimizer,scaler); va=run_epoch(model,val_loader,cfg,device)
        score=.55*va["patch_auc"]+.45*va["pair_top10_auc"]-.02*va["inv"]
        row={"epoch":epoch,"score":score,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}}; rows.append(row)
        line=f"variant={cfg['run_name']} epoch={epoch} train={tr['total']:.5f} val={va['total']:.5f} patch_auc={va['patch_auc']:.5f} top10_auc={va['pair_top10_auc']:.5f} score={score:.5f}"
        print(line,flush=True); log.write(line+"\n")
        if score>best+1e-5:
            best=score; stale=0; save_checkpoint({"model":model.state_dict(),"config":cfg,"epoch":epoch,"score":score},a.output/"patch_checkpoints"/"best.pt")
        else: stale+=1
        with (a.output/"patch_results"/"training_history.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (a.output/"patch_results"/"summary.json").write_text(json.dumps({"status":"running","best_score":best,"latest_epoch":epoch,"stale":stale,"config":cfg},indent=2),encoding="utf-8")
        if stale>=int(cfg["patience"]): break
    (a.output/"patch_results"/"summary.json").write_text(json.dumps({"status":"complete","best_score":best,"epochs":rows[-1]["epoch"],"config":cfg},indent=2),encoding="utf-8")


if __name__=="__main__": main()
