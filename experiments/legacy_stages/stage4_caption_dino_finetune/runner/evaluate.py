from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from caption_model import ChangeCaptioner
from common import caption_metrics, encode_caption, load_encoder, seed_everything, stable_seed, strip_special

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
CORRUPTIONS = ("cloud", "haze", "gaussian_noise", "blur", "jpeg", "low_resolution", "pixel_dropout", "misregistration", "radiometric_shift", "mixed")
CONDITIONS = ["clean"] + [f"{kind}__{target}" for kind in CORRUPTIONS for target in ("A", "B", "both")]


def to_tensor(image):
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy()/255.0
    return (torch.from_numpy(array).permute(2, 0, 1)-MEAN)/STD


class TestPairs(Dataset):
    def __init__(self, dataset_root, list_root, token_root, vocab, condition, degradation, severity=2, seed=20260928):
        self.root, self.token_root, self.vocab = dataset_root, token_root, vocab
        self.names = [line.strip() for line in (list_root/"test.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.condition, self.degradation, self.severity, self.seed = condition, degradation, severity, seed
    def __len__(self): return len(self.names)
    def __getitem__(self, index):
        name=self.names[index]; kind=target=None
        if self.condition!="clean": kind,target=self.condition.rsplit("__",1)
        images={}
        for side in ("A","B"):
            with Image.open(self.root/"images"/"test"/side/name) as opened: image=opened.convert("RGB").copy()
            if kind and (target==side or target=="both"):
                image,_=self.degradation.OPERATORS[kind](image,self.severity,np.random.default_rng(stable_seed(self.seed,kind,target,name,side)))
            images[side]=to_tensor(image)
        refs=json.loads((self.token_root/f"{Path(name).stem}.txt").read_text(encoding="utf-8"))
        refs=np.stack([encode_caption(row,self.vocab)[0] for row in refs])
        return images["A"],images["B"],torch.from_numpy(refs),name


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--variant",required=True); p.add_argument("--stage1-root",type=Path,required=True)
    p.add_argument("--encoder-checkpoint",type=Path,required=True); p.add_argument("--caption-checkpoint",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--list-root",type=Path,required=True); p.add_argument("--token-root",type=Path,required=True)
    p.add_argument("--vocab",type=Path,required=True); p.add_argument("--change-agent-root",type=Path,required=True); p.add_argument("--degradation-script",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--workers",type=int,default=4); p.add_argument("--device",default="cuda")
    p.add_argument("--condition",choices=CONDITIONS); p.add_argument("--limit",type=int); return p.parse_args()


def main():
    args=parse_args(); seed_everything(20261001); args.output.mkdir(parents=True,exist_ok=True); (args.output/"logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler(args.output/"logs"/f"evaluate_{args.variant}.log",encoding="utf-8"),logging.StreamHandler()])
    vocab=json.loads(args.vocab.read_text(encoding="utf-8"))
    import importlib.util
    spec=importlib.util.spec_from_file_location("degradations",args.degradation_script); degradation=importlib.util.module_from_spec(spec); spec.loader.exec_module(degradation)
    encoder,encoder_ckpt,missing=load_encoder(args.stage1_root,args.encoder_checkpoint,args.device)
    captioner=ChangeCaptioner(args.change_agent_root,vocab).to(args.device); caption_ckpt=torch.load(args.caption_checkpoint,map_location="cpu",weights_only=False); captioner.load_state_dict(caption_ckpt["model"]); captioner.eval()
    logging.info("variant=%s encoder_epoch=%s caption_epoch=%s missing_frozen=%d",args.variant,encoder_ckpt.get("epoch"),caption_ckpt.get("epoch"),missing)
    conditions=[args.condition] if args.condition else CONDITIONS
    for number,condition in enumerate(conditions,1):
        target=args.output/"results"/args.variant/condition/"metrics.json"
        if target.exists(): logging.info("skip completed condition=%s",condition); continue
        dataset=TestPairs(args.dataset_root,args.list_root,args.token_root,vocab,condition,degradation)
        if args.limit: dataset.names=dataset.names[:args.limit]
        loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
        references,hypotheses=[],[]; started=time.time()
        for step,(a,b,all_refs,_) in enumerate(loader,1):
            images=torch.cat([a,b]).to(args.device,non_blocking=True)
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16,enabled=args.device.startswith("cuda")):
                _,features=encoder.encode_images(images); fa,fb=features.chunk(2); prediction=captioner.greedy(fa,fb).cpu().tolist()
            references.extend([[strip_special(row,vocab) for row in group] for group in all_refs.tolist()]); hypotheses.extend(strip_special(row,vocab) for row in prediction)
            if step%25==0: logging.info("condition=%s %d/%d elapsed_min=%.1f",condition,min(step*args.batch_size,len(dataset)),len(dataset),(time.time()-started)/60)
        score=caption_metrics(args.change_agent_root,references,hypotheses); payload={"variant":args.variant,"condition":condition,"samples":len(references),**score}
        target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        logging.info("done %d/%d variant=%s condition=%s BLEU4=%.5f ROUGE_L=%.5f CIDEr=%.5f",number,len(conditions),args.variant,condition,score["Bleu_4"],score["ROUGE_L"],score["CIDEr"])


if __name__=="__main__": main()
