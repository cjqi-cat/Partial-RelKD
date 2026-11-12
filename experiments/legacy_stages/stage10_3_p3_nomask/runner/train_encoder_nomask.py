#!/usr/bin/env python3
"""Strict mask-free DINO robust encoder training.

No mask image is opened, no pixel/patch change target is constructed, and no
change/no-change target enters the loss or checkpoint selector.  The original
balanced image-name subset is retained only to keep the data distribution
matched to the prior experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import DinoRobustModel, GradientReverse


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
ALL = ("cloud", "haze", "gaussian_noise", "blur", "jpeg", "low_resolution",
       "pixel_dropout", "misregistration", "radiometric_shift", "mixed")
HELDOUT = {"blur", "low_resolution", "misregistration", "mixed"}
EXPECTED = tuple(x for x in ALL if x not in HELDOUT)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--smoke-batches", type=int, default=0)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_seed(base, *parts):
    raw = "|".join((str(base), *(str(x) for x in parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def load_operators(path):
    spec = importlib.util.spec_from_file_location("stage103_degradations", path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return dict(module.OPERATORS)


def balanced_names(root, split, limit, seed):
    rows = json.loads((root / "LevirCCcaptions.json").read_text(encoding="utf-8"))["images"]
    groups = {0: [], 1: []}
    for row in rows:
        if row["split"] == split: groups[int(row["changeflag"])].append(row["filename"])
    rng = random.Random(seed + (0 if split == "train" else 1000))
    each = min(limit // 2, len(groups[0]), len(groups[1]))
    names = rng.sample(groups[0], each) + rng.sample(groups[1], each)
    rng.shuffle(names)
    return names


def checked_open(path):
    path = Path(path)
    forbidden = {"label", "labels", "mask", "masks", "sem"}
    if any(part.lower() in forbidden for part in path.parts):
        raise RuntimeError(f"MASK_ACCESS_BLOCKED: {path}")
    return Image.open(path)


def tensor(image):
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32).copy() / 255.0
    return (torch.from_numpy(arr).permute(2, 0, 1) - MEAN) / STD


class MaskFreePairs(Dataset):
    def __init__(self, cfg, split):
        self.root = Path(cfg["dataset_root"]); self.split = split
        self.seed = int(cfg["seed"]) + (0 if split == "train" else 10_000_000)
        self.names = balanced_names(self.root, split, int(cfg[f"{split}_pairs"]), int(cfg["seed"]))
        self.allowed = tuple(cfg["allowed_degradations"])
        if self.allowed != EXPECTED: raise RuntimeError(f"LOFO mismatch: {self.allowed} != {EXPECTED}")
        self.ops = load_operators(Path(cfg["degradation_script"]))

    def __len__(self): return len(self.names)

    def degrade(self, clean, rng):
        index = int(rng.integers(0, len(self.allowed)))
        severity = int(rng.integers(1, 4))
        output, _ = self.ops[self.allowed[index]](clean, severity, rng)
        if not isinstance(output, Image.Image): output = Image.fromarray(np.asarray(output, dtype=np.uint8))
        return output.convert("RGB"), index

    def __getitem__(self, index):
        name = self.names[index]; views, kinds = [], []
        for side_index, side in enumerate(("A", "B")):
            path = self.root / "images" / self.split / side / name
            with checked_open(path) as opened: clean = opened.convert("RGB").copy()
            rng1 = np.random.default_rng(stable_seed(self.seed, index, side_index, 1))
            rng2 = np.random.default_rng(stable_seed(self.seed, index, side_index, 2))
            first, k1 = self.degrade(clean, rng1); second, k2 = self.degrade(clean, rng2)
            views.extend((tensor(clean), tensor(first), tensor(second)))
            kinds.extend((-1, k1, k2))
        return torch.stack(views), torch.tensor(kinds, dtype=torch.long), name


def nce(a, b, temperature):
    logits = a @ b.T / temperature
    target = torch.arange(len(a), device=a.device)
    return .5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def compute(model, images, kinds, cfg):
    batch, views = images.shape[:2]
    raw_all, adapted_all = model.encode_images(images.flatten(0, 1))
    raw = raw_all.view(batch, views, *raw_all.shape[1:]).unbind(1)
    z = adapted_all.view(batch, views, *adapted_all.shape[1:]).unbind(1)
    ca, d1a, d2a, cb, d1b, d2b = z
    count = min(int(cfg["patches_per_image"]), ca.shape[1])
    ids = torch.randperm(ca.shape[1], device=ca.device)[:count]
    pairs = ((ca,d1a),(ca,d2a),(d1a,d2a),(cb,d1b),(cb,d2b),(d1b,d2b))
    patch_inv = sum((1-(x[:,ids]*y[:,ids]).sum(-1)).mean() for x,y in pairs)/len(pairs)
    global_inv = sum(nce(F.normalize(x.mean(1),dim=-1),F.normalize(y.mean(1),dim=-1),float(cfg["temperature"])) for x,y in pairs)/len(pairs)
    invariance = patch_inv + global_inv
    distill = sum((1-(adapted*base.detach()).sum(-1)).mean() for adapted,base in zip(z,raw))/len(z)
    adversarial = sum(F.cross_entropy(model.degradation_head(GradientReverse.apply(feature.mean(1),float(cfg["grl_scale"]))),label) for feature,label in ((d1a,kinds[:,1]),(d2a,kinds[:,2]),(d1b,kinds[:,4]),(d2b,kinds[:,5])))/4
    anchor = model.backbone_anchor_loss()
    total = (float(cfg["lambda_invariance"])*invariance + float(cfg["lambda_distill"])*distill +
             float(cfg["lambda_adversarial"])*adversarial + float(cfg["lambda_backbone_anchor"])*anchor)
    # Selection deliberately excludes the adversarial minimax head. Lower is better.
    selection = invariance + float(cfg["lambda_distill"])*distill + float(cfg["lambda_backbone_anchor"])*anchor
    return total, {"total":total,"inv":invariance,"distill":distill,"adversarial":adversarial,"anchor":anchor,"selection":selection}


def epoch(model, loader, cfg, device, optimizer=None, scaler=None, max_batches=0):
    training = optimizer is not None; model.train(training)
    sums = {k:0.0 for k in ("total","inv","distill","adversarial","anchor","selection")}; seen=0
    accumulation = int(cfg["gradient_accumulation"]) if training else 1
    if training: optimizer.zero_grad(set_to_none=True)
    for step,(images,kinds,_) in enumerate(loader,1):
        images=images.to(device,non_blocking=True); kinds=kinds.to(device,non_blocking=True)
        with (torch.enable_grad() if training else torch.no_grad()):
            with torch.autocast("cuda",dtype=torch.float16,enabled=device.type=="cuda"):
                total,parts=compute(model,images,kinds,cfg); scaled=total/accumulation
        if training:
            scaler.scale(scaled).backward()
            if step%accumulation==0 or step==len(loader) or (max_batches and step==max_batches):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg["grad_clip"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        size=images.size(0); seen+=size
        for key,value in parts.items(): sums[key]+=float(value.detach())*size
        if max_batches and step>=max_batches: break
    return {key:value/max(seen,1) for key,value in sums.items()}


def optimizer_for(model,cfg):
    lora=[]; backbone=[]; task=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if name.startswith("backbone."):
            (lora if ".lora_a" in name or ".lora_b" in name else backbone).append(p)
        else: task.append(p)
    groups=[]
    if lora: groups.append({"params":lora,"lr":float(cfg.get("lora_learning_rate",cfg["backbone_learning_rate"]))})
    if backbone: groups.append({"params":backbone,"lr":float(cfg["backbone_learning_rate"])})
    if task: groups.append({"params":task,"lr":float(cfg["task_learning_rate"])})
    return torch.optim.AdamW(groups,weight_decay=float(cfg["weight_decay"]))


def trainable_state(model):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    return {n:v.detach().cpu() for n,v in model.state_dict().items() if n in names}


def atomic_save(payload,path):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,temp); temp.replace(path)


def main():
    args=args_parser(); cfg=json.loads(args.config.read_text(encoding="utf-8")); seed_all(int(cfg["seed"]))
    if cfg.get("mask_supervision") is not False: raise RuntimeError("mask_supervision must be false")
    out=args.output; (out/"checkpoints").mkdir(parents=True,exist_ok=True); (out/"results").mkdir(exist_ok=True)
    device=torch.device(cfg.get("device","cuda") if torch.cuda.is_available() else "cpu")
    model=DinoRobustModel(cfg).to(device); optimizer=optimizer_for(model,cfg)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start=1; best=math.inf; stale=0; history=[]
    if args.resume:
        ck=torch.load(args.resume,map_location="cpu",weights_only=False); model.load_state_dict(ck["trainable_state"],strict=False)
        optimizer.load_state_dict(ck["optimizer"]); start=int(ck["epoch"])+1; best=float(ck["best_score"]); stale=int(ck.get("stale",0)); history=ck.get("history",[])
    train_set=MaskFreePairs(cfg,"train"); val_set=MaskFreePairs(cfg,"val")
    kwargs={"batch_size":int(cfg["batch_size_pairs"]),"num_workers":int(cfg["num_workers"]),"pin_memory":device.type=="cuda"}
    generator=torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader=DataLoader(train_set,shuffle=True,drop_last=True,generator=generator,**kwargs)
    val_loader=DataLoader(val_set,shuffle=False,**kwargs)
    for number in range(start,int(cfg["epochs"])+1):
        began=time.time(); train=epoch(model,train_loader,cfg,device,optimizer,scaler,args.smoke_batches); val=epoch(model,val_loader,cfg,device,max_batches=args.smoke_batches)
        row={"epoch":number,**{f"train_{k}":v for k,v in train.items()},**{f"val_{k}":v for k,v in val.items()},"seconds":time.time()-began}; history.append(row); print(row,flush=True)
        improved=val["selection"]<best-float(cfg.get("min_delta",1e-5)); best=min(best,val["selection"]); stale=0 if improved else stale+1
        payload={"architecture":"dino_robust_nomask","variant":cfg["run_name"],"epoch":number,"best_score":best,"stale":stale,"config":cfg,"trainable_state":trainable_state(model),"optimizer":optimizer.state_dict(),"history":history,"provenance":{"mask_used":False,"mask_files_opened":0,"selection":"mask-free invariance+distill+anchor"}}
        atomic_save(payload,out/"checkpoints"/"last.pt")
        if improved: atomic_save(payload,out/"checkpoints"/"best.pt")
        with (out/"results"/"training_history.csv").open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        if args.smoke_batches or stale>=int(cfg["patience"]): break
    (out/"results"/"summary.json").write_text(json.dumps({"status":"complete","best_score":best,"mask_used":False,"epochs":history[-1]["epoch"]},indent=2),encoding="utf-8")


if __name__ == "__main__": main()
