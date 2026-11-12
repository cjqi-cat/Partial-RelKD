import argparse
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from common import load_json, seed_everything
from data import CaptionPairs, CaptionTokens
from models import build_model


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    cfg=load_json(args.config); seed_everything(cfg["seed"]); device=torch.device("cuda")
    tok=CaptionTokens(cfg["metadata_root"],cfg["max_length"]); ds=CaptionPairs(cfg,"train",tok,True)
    batch=next(iter(DataLoader(ds,batch_size=1,num_workers=0)))
    model=build_model(cfg,tok.vocab).to(device).train()
    out=model(batch["a"].to(device),batch["b"].to(device),batch["tokens"].to(device),batch["length"].to(device))
    score,cap,lens,_=out; target=cap[:,1:]
    loss=torch.nn.functional.cross_entropy(pack_padded_sequence(score,lens,True).data.float(),pack_padded_sequence(target,lens,True).data)
    loss.backward()
    print({"model":cfg["model"],"scores":tuple(score.shape),"loss":float(loss),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad)})
    print("SMOKE_OK")
if __name__=="__main__": main()
