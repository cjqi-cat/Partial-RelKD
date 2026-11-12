import argparse
from pathlib import Path
from common import load_json
p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/LEVIR_MCI/rmnet.json'); a=p.parse_args(); c=load_json(a.config)
for k in ('dataset_root','metadata_root','change_agent_root','degradation_script','rmnet_root','vmamba_checkpoint'):
    if not Path(c[k]).exists(): raise FileNotFoundError(f'{k}: {c[k]}')
if set(c['allowed_degradations']) & set(c['heldout_degradations']): raise RuntimeError('LOFO leakage')
for split in ('train','val','test'):
    n=sum(1 for x in open(Path(c['metadata_root'])/f'{split}.txt') if x.strip()); print(split,n)
print('PREFLIGHT_OK')
