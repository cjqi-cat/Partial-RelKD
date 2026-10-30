import json
from pathlib import Path

ROOT=Path("__PROJECT_ROOT__/stage10_9_three_dataset_baselines"); PROJECT=Path("__PROJECT_ROOT__")
STAGE10=PROJECT/"stage10_authoritative_baselines"; CA=Path("__CHANGE_AGENT_ROOT__")
DEG=PROJECT/"stage6_5_calibrated_v3/degradation/calibrated_v3.py"
ALLOWED=["cloud","haze","gaussian_noise","jpeg","pixel_dropout","radiometric_shift"]
HELDOUT=["blur","low_resolution","misregistration","mixed"]
SPECS={"chg2cap":dict(batch_size=6,eval_batch_size=12),"mccformer_d":dict(batch_size=8,eval_batch_size=16),
       "rsiccformer":dict(batch_size=8,eval_batch_size=16),
       "sen":dict(batch_size=8,eval_batch_size=16,model_stage=4,sen_ft_layer=7,sen_weight_path="resnet50_B6_imagenet")}
def dump(x,p): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def main():
 for dataset in ("SECOND_CC","WHU_CDC"):
  for model,extra in SPECS.items():
   cfg={"dataset":dataset,"model":model,"family":"geometry_quality","seed":20261008,
        "dataset_root":str(PROJECT/f"stage8_1_multidataset_full/data_views/{dataset}"),
        "metadata_root":str(PROJECT/f"stage8_1_multidataset_full/metadata/{dataset}"),
        "change_agent_root":str(CA),"degradation_script":str(DEG),"stage10_root":str(STAGE10),
        "allowed_degradations":ALLOWED,"heldout_degradations":HELDOUT,"severity":2,"degradation_seed":20260928,
        "clean_probability":.25,"image_size":256,"max_length":41,"encoder_layers":3,"decoder_layers":3,
        "heads":8,"dropout":.3,"batch_size":8,"eval_batch_size":16,"num_workers":4,"lr":5e-5,
        "weight_decay":.01,"epochs":30,"patience":8,"label_smoothing":.05,**extra}
   dump(cfg,ROOT/f"configs/{dataset}/{model}.json")
 print("CONFIGS_OK")
if __name__=="__main__":main()
