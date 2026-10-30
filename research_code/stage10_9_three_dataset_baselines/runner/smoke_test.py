import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import CaptionPairs, CaptionTokens
from models import build_model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text()); tok = CaptionTokens(cfg["metadata_root"], cfg.get("max_length", 41))
    batch = next(iter(DataLoader(CaptionPairs(cfg, "train", tok, True), batch_size=1, num_workers=0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = build_model(cfg, tok.vocab).to(device).train()
    a, b, ids, lengths = (batch[x].to(device) for x in ("a", "b", "tokens", "length"))
    out = model(a, b, ids, lengths); loss = out[0].float().mean(); loss.backward()
    print({"model": cfg["model"], "scores": tuple(out[0].shape), "probe": float(loss)})
    print("SMOKE_OK")


if __name__ == "__main__": main()
