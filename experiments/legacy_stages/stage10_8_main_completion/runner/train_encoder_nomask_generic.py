#!/usr/bin/env python3
"""Adapts the audited Stage10.3 mask-free encoder trainer to any declared LOFO family."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path

source=Path('__PROJECT_ROOT__/stage10_3_p3_nomask/runner/train_encoder_nomask.py')
cfg_path=Path(sys.argv[sys.argv.index('--config')+1])
cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
if cfg.get('mask_supervision') is not False: raise RuntimeError('mask_supervision must be false')
all_types=('cloud','haze','gaussian_noise','blur','jpeg','low_resolution','pixel_dropout','misregistration','radiometric_shift','mixed')
held=set(cfg['heldout_degradations']); expected=tuple(x for x in all_types if x not in held)
if tuple(cfg['allowed_degradations']) != expected: raise RuntimeError(f'LOFO mismatch {cfg["allowed_degradations"]} != {expected}')
sys.path.insert(0,str(source.parent))
spec=importlib.util.spec_from_file_location('stage108_nomask_source',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.ALL=all_types;mod.HELDOUT=held;mod.EXPECTED=expected
print({'mask_used':False,'heldout':sorted(held),'allowed':list(expected)},flush=True)
mod.main()
