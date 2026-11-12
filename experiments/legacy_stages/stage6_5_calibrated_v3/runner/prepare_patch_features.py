#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from PIL import Image

from common import balanced_names,load_config,load_degradations,seed_everything

MEAN=torch.tensor((0.485,0.456,0.406)).view(3,1,1)
STD=torch.tensor((0.229,0.224,0.225)).view(3,1,1)
CORRUPTIONS=("cloud","haze","gaussian_noise","blur","jpeg","low_resolution","pixel_dropout","misregistration","radiometric_shift","mixed")
VIEWS=("clean_a","deg1_a","deg2_a","clean_b","deg1_b","deg2_b")


def to_tensor(image):
    image=image.convert("RGB").resize((224,224),Image.Resampling.BICUBIC)
    arr=np.asarray(image,dtype=np.float32).copy()/255.0
    return (torch.from_numpy(arr).permute(2,0,1)-MEAN)/STD


def make_degraded(image,operators,rng):
    kind=int(rng.integers(0,len(CORRUPTIONS))); severity=int(rng.integers(1,4))
    out,_=operators[CORRUPTIONS[kind]](image,severity,rng)
    return out,kind,severity


def load_views(root,split,name,operators,seed):
    tensors=[]; kinds=[]; severities=[]
    for side in ("A","B"):
        with Image.open(root/"images"/split/side/name) as opened: clean=opened.convert("RGB").copy()
        offset=0 if side=="A" else 37
        rng1=np.random.default_rng(seed+offset); rng2=np.random.default_rng(seed+1_000_003+offset)
        d1,k1,s1=make_degraded(clean,operators,rng1); d2,k2,s2=make_degraded(clean,operators,rng2)
        tensors.extend((to_tensor(clean),to_tensor(d1),to_tensor(d2))); kinds.extend((-1,k1,k2)); severities.extend((0,s1,s2))
    return tensors,kinds,severities


def patch_mask(path,grid=(16,16)):
    with Image.open(path) as im: arr=(np.asarray(im.convert("L"),dtype=np.uint8)>0).astype(np.float32)
    return (F.adaptive_max_pool2d(torch.from_numpy(arr)[None,None],grid).flatten()>0).numpy().astype(np.uint8)


def extract_split(model,root,split,names,labels,operators,out,batch_size,device,seed):
    count=len(names); maps={key:open_memmap(out/f"{split}_{key}.npy",mode="w+",dtype=np.float16,shape=(count,256,768)) for key in VIEWS}
    masks=open_memmap(out/f"{split}_masks.npy",mode="w+",dtype=np.uint8,shape=(count,256)); kinds=np.full((count,6),-1,dtype=np.int16); severities=np.zeros((count,6),dtype=np.int8)
    pending=[]; locations=[]; started=time.time()
    def flush():
        if not pending:return
        x=torch.stack(pending).to(device)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16,enabled=device.startswith("cuda")):
            z=model.forward_features(x)[:,int(model.num_prefix_tokens):]
            z=F.normalize(z.float(),dim=-1).cpu().numpy().astype(np.float16)
        for (row,key),value in zip(locations,z):maps[key][row]=value
        pending.clear();locations.clear()
    for idx,name in enumerate(names):
        views,k,s=load_views(root,split,name,operators,seed+idx*1009);kinds[idx]=k;severities[idx]=s;masks[idx]=patch_mask(root/"images"/split/"label"/name)
        for key,image in zip(VIEWS,views):
            pending.append(image);locations.append((idx,key))
            if len(pending)>=batch_size:flush()
        if (idx+1)%50==0 or idx+1==count:logging.info("split=%s pairs=%d/%d elapsed_min=%.1f",split,idx+1,count,(time.time()-started)/60)
    flush()
    for value in maps.values():value.flush()
    masks.flush();np.savez_compressed(out/f"{split}_meta.npz",names=np.asarray(names),labels=labels,kinds=kinds,severities=severities)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    cfg=load_config(a.config);out=a.output/"patch_features";logs=a.output/"logs";out.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler(logs/"prepare_patch_features.log",encoding="utf-8"),logging.StreamHandler()])
    seed=int(cfg["seed"]);seed_everything(seed);device=cfg["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    root=Path(cfg["dataset_root"]);operators=load_degradations(Path(cfg["degradation_script"]))
    import timm
    model=timm.create_model("vit_base_patch14_dinov2",pretrained=True,pretrained_cfg_overlay={"hf_hub_id":None},img_size=224,num_classes=0).to(device).eval()
    counts={}
    for split,limit in (("train",int(cfg["train_pairs"])),("val",int(cfg["val_pairs"]))):
        names,labels=balanced_names(root,split,limit,seed);extract_split(model,root,split,names,labels,operators,out,int(cfg["batch_size_extract"]),device,seed);counts[split]=len(names)
    (out/"manifest.json").write_text(json.dumps({"status":"complete","profile":"calibrated_v3","seed":seed,"counts":counts,"shape":"N x 256 x 768","dtype":"float16","test_used":False},indent=2),encoding="utf-8")


if __name__=="__main__":main()
