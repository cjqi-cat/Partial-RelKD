#!/usr/bin/env python3
from pathlib import Path
import json
P=Path('__PROJECT_ROOT__');R=P/'stage10_8_main_completion'
required=[P/'stage10_3_p3_nomask/runs/p3_nomask/best.pt',P/'stage6_7_partial_hybrid/runs/p3_partial_relation/best.pt',P/'stage10_7_severity_curves/degradation/calibrated_v3_curve.py',P/'stage3_caption_retrain/vendor/Change-Agent/Multi_change/models_ckpt/MCI_model.pth',P/'stage8_1_multidataset_full/data_views/SECOND_CC',P/'stage8_1_multidataset_full/data_views/WHU_CDC']
missing=[str(x) for x in required if not x.exists()]
if missing:raise SystemExit('MISSING:\n'+'\n'.join(missing))
for p in R.glob('configs/*/*.json'):
 c=json.loads(p.read_text());leak=set(c.get('allowed_degradations',[]))&set(c.get('heldout_degradations',[]));
 if leak:raise RuntimeError(f'{p}: leakage {leak}')
print('PREFLIGHT_OK')
