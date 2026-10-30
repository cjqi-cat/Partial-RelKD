#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

PROJECT=Path('__PROJECT_ROOT__')
ROOT=PROJECT/'stage10_8_main_completion'
ALL=['cloud','haze','gaussian_noise','blur','jpeg','low_resolution','pixel_dropout','misregistration','radiometric_shift','mixed']
FAMILIES={
 'atmospheric':['cloud','haze','mixed'],
 'geometry_quality':['blur','low_resolution','misregistration','mixed'],
 'noise_loss':['gaussian_noise','jpeg','pixel_dropout','mixed'],
}
CA=PROJECT/'stage3_caption_retrain/vendor/Change-Agent/Multi_change'
DATA=Path('__DATA_ROOT__/LEVIR-MCI-dataset')
DEG=PROJECT/'stage6_5_calibrated_v3/degradation/calibrated_v3.py'

def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(x,p): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def family_configs(family,seed=20261008,tag=None):
 held=FAMILIES[family]; allowed=[x for x in ALL if x not in held]; tag=tag or family
 base=ROOT/'runs'/tag
 enc=load(PROJECT/'stage6_6_m2_backbones/configs/dino_partial_encoder_geometry.json')
 enc.update(run_name=f'{tag}_masked_encoder',seed=seed,allowed_degradations=allowed,degradation_script=str(DEG))
 enc_out=base/'masked_encoder'
 teacher=base/'masked_teacher/checkpoints/partial_last2_best.pt'
 common=dict(family=family,allowed_degradations=allowed,heldout_degradations=held,seed=seed,
             degradation_script=str(DEG),dino_encoder_checkpoint=str(enc_out/'checkpoints/best.pt'),dino_caption_checkpoint=str(teacher))
 aug=load(PROJECT/'stage6_6_m2_backbones/configs/dino_partial_m2_geometry.json');aug.update(common);aug['run_name']=f'{tag}_dino_aug'
 p3=load(PROJECT/'stage6_7_partial_hybrid/configs/p3_partial_relation.json');p3.update(common);p3['run_name']=f'{tag}_p3'
 nenc=load(PROJECT/'stage10_3_p3_nomask/configs/partial_encoder_nomask.json')
 nenc.update(run_name=f'{tag}_nomask_encoder',seed=seed,allowed_degradations=allowed,heldout_degradations=held,degradation_script=str(DEG),mask_supervision=False)
 nbase=base/'nomask_encoder'; nteacher=base/'nomask_teacher/checkpoints/partial_last2_best.pt'
 np3=load(PROJECT/'stage10_3_p3_nomask/configs/p3_nomask.json')
 np3.update(family=family,allowed_degradations=allowed,heldout_degradations=held,seed=seed,degradation_script=str(DEG),
            dino_encoder_checkpoint=str(nbase/'checkpoints/best.pt'),dino_caption_checkpoint=str(nteacher),run_name=f'{tag}_p3_nomask')
 out=ROOT/'configs'/tag
 for name,obj in [('masked_encoder',enc),('dino_aug',aug),('p3',p3),('nomask_encoder',nenc),('p3_nomask',np3)]:dump(obj,out/f'{name}.json')

def cross_configs(dataset):
 spec={'SECOND_CC':(8438,1190),'WHU_CDC':(5947,743)}[dataset];tag=f'cross_{dataset.lower()}'
 family_configs('geometry_quality',20261008,tag)
 out=ROOT/'configs'/tag;base=ROOT/'runs'/tag
 meta=PROJECT/'stage8_1_multidataset_full/metadata'/dataset; data=PROJECT/'stage8_1_multidataset_full/data_views'/dataset
 nenc=load(out/'nomask_encoder.json');nenc.update(dataset_root=str(data),metadata_root=str(meta),train_pairs=spec[0],val_pairs=spec[1],name_selection='balanced_changeflag')
 np3=load(out/'p3_nomask.json');np3.update(dataset_root=str(data),metadata_root=str(meta),dino_encoder_checkpoint=str(base/'nomask_encoder/checkpoints/best.pt'),dino_caption_checkpoint=str(base/'nomask_teacher/checkpoints/partial_last2_best.pt'))
 dump(nenc,out/'nomask_encoder.json');dump(np3,out/'p3_nomask.json')

def main():
 for f in FAMILIES: family_configs(f)
 for seed in (20261018,20261028): family_configs('geometry_quality',seed,f'geometry_seed{seed}')
 for d in ('SECOND_CC','WHU_CDC'):cross_configs(d)
 print('CONFIGS_OK')
if __name__=='__main__':main()
