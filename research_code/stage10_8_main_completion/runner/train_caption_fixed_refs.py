#!/usr/bin/env python3
"""Stage4 caption teacher with reference sets padded/truncated to five for batching."""
from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import numpy as np, torch

root=Path('__PROJECT_ROOT__/stage4_caption_dino_finetune/runner')
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('stage108_caption_source',root/'train_caption.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class FixedDataset(mod.CaptionFeatureDataset):
 def __getitem__(self,index):
  a,b,tokens,length,all_refs,name=super().__getitem__(index)
  rows=list(all_refs)
  if not rows: raise RuntimeError(f'No references for {name}')
  rows=(rows+[rows[-1]]*5)[:5]
  return a,b,tokens,length,torch.stack(rows),name
mod.CaptionFeatureDataset=FixedDataset
mod.main()
