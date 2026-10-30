#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("__PROJECT_ROOT__/stage8_1_multidataset_full")
PROJECT = Path("__PROJECT_ROOT__")
CA = PROJECT / "stage3_caption_retrain/vendor/Change-Agent/Multi_change"
DEG = PROJECT / "stage6_5_calibrated_v3/degradation/calibrated_v3.py"
STAGE67 = PROJECT / "stage6_7_partial_hybrid"
STAGE4 = PROJECT / "stage4_caption_dino_finetune"
ALLOWED = ["cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift"]
HELDOUT = ["blur", "low_resolution", "misregistration", "mixed"]
SPECS = {
    "SECOND_CC": {"train": 8438, "val": 1190},
    "WHU_CDC": {"train": 5947, "val": 743},
}


def dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def encoder(dataset, mode):
    spec = SPECS[dataset]
    cfg = {
        "run_name": f"{dataset.lower()}_{mode}_encoder",
        "seed": 20261008,
        "dataset_root": str(ROOT / "data_views" / dataset),
        "degradation_script": str(DEG),
        "allowed_degradations": ALLOWED,
        "finetune_mode": "partial_last2" if mode == "partial" else "hybrid_lora_partial",
        "unfreeze_last_blocks": 2,
        "gradient_checkpointing": True,
        "train_pairs": spec["train"], "val_pairs": spec["val"],
        "batch_size_pairs": 4 if mode == "partial" else 2,
        "gradient_accumulation": 2 if mode == "partial" else 4,
        "epochs": 40, "patience": 8,
        "backbone_learning_rate": 5e-6, "task_learning_rate": 3e-4,
        "weight_decay": 1e-4, "adapter_hidden": 256, "adapter_scale": 0.35,
        "temperature": 0.1, "patches_per_image": 64, "change_margin": 0.35,
        "lambda_invariance": 1.0, "lambda_change": 1.0,
        "lambda_patch_cls": 0.5, "lambda_distill": 0.1,
        "lambda_adversarial": 0.15, "lambda_backbone_anchor": 0.001,
        "grl_scale": 1.0, "grad_clip": 5.0, "device": "cuda", "amp": True,
        "num_workers": 4,
    }
    if mode == "hybrid":
        cfg.update({"lora_frozen_blocks": 10, "lora_rank": 8, "lora_alpha": 16.0,
                    "lora_dropout": 0.05, "lora_learning_rate": 1e-4})
    return cfg


def caption(dataset, architecture, encoder_ckpt, teacher_ckpt, name):
    cfg = {
        "run_name": name, "architecture": architecture, "family": "geometry_quality",
        "allowed_degradations": ALLOWED, "heldout_degradations": HELDOUT,
        "dataset_root": str(ROOT / "data_views" / dataset),
        "metadata_root": str(ROOT / "metadata" / dataset),
        "change_agent_root": str(CA), "degradation_script": str(DEG),
        "dino_model_path": str(STAGE67 / "runner/model.py"),
        "dino_caption_model_path": str(STAGE4 / "runner/caption_model.py"),
        "dino_encoder_checkpoint": str(encoder_ckpt),
        "dino_caption_checkpoint": str(teacher_ckpt),
        "dino_train_encoder": True, "dino_decoder_layers": 3, "heads": 8,
        "dropout": 0.1, "max_length": 41, "batch_size": 4,
        "val_batch_size": 16, "num_workers": 4, "epochs": 30, "patience": 8,
        "main_lr": 5e-5, "backbone_lr": 1e-6, "weight_decay": 1e-4,
        "grad_clip": 5.0, "clean_probability": 0.25, "label_smoothing": 0.05,
        "lambda_token_kd": 0.5, "lambda_clean": 0.5, "lambda_relation": 0.1,
        "lambda_backbone_anchor": 0.001, "selection_clean_weight": 0.5,
        "kd_temperature": 2.0, "severity": 2, "seed": 20261008,
        "degradation_seed": 20260928, "device": "cuda",
    }
    return cfg


def main():
    for dataset in SPECS:
        base = ROOT / "runs" / dataset
        config_root = ROOT / "configs" / dataset
        partial_encoder = base / "partial_encoder/checkpoints/best.pt"
        partial_teacher = base / "partial_teacher/checkpoints/partial_last2_best.pt"
        dump(encoder(dataset, "partial"), config_root / "partial_encoder.json")
        m2 = caption(dataset, "dino_partial_last2", partial_encoder, partial_teacher, f"{dataset.lower()}_m2")
        m2.update({"epochs": 20, "patience": 6, "lambda_clean": 0.0, "lambda_relation": 0.0})
        dump(m2, config_root / "m2.json")
        dump(caption(dataset, "dino_partial_last2", partial_encoder, partial_teacher, f"{dataset.lower()}_p3"), config_root / "p3.json")
        ca = {
            "run_name": f"{dataset.lower()}_change_agent_target", "architecture": "change_agent",
            "family": "geometry_quality", "allowed_degradations": ALLOWED,
            "heldout_degradations": HELDOUT,
            "dataset_root": str(ROOT / "data_views" / dataset),
            "metadata_root": str(ROOT / "metadata" / dataset),
            "change_agent_root": str(CA), "ca_checkpoint": str(CA / "models_ckpt/MCI_model.pth"),
            "degradation_script": str(DEG), "ca_network": "segformer-mit_b1",
            "ca_train_encoder": True, "ca_neck_layers": 3, "ca_decoder_layers": 1,
            "heads": 8, "dropout": 0.1, "max_length": 41,
            "batch_size": 8, "val_batch_size": 8, "num_workers": 4,
            "epochs": 30, "patience": 8, "main_lr": 5e-5, "backbone_lr": 1e-6,
            "weight_decay": 1e-4, "grad_clip": 5.0, "clean_probability": 1.0,
            "label_smoothing": 0.05, "severity": 2, "seed": 20261008,
            "degradation_seed": 20260928, "device": "cuda",
        }
        dump(ca, config_root / "change_agent.json")
    print("CONFIGS_OK", flush=True)


if __name__ == "__main__":
    main()
