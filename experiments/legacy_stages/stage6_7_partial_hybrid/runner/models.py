from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import torch
from torch import nn

from common import load_module


def _greedy(model, a, b, start_id, end_id, pad_id, max_length):
    memory = model.memory(a, b)
    batch = a.size(0)
    sequence = torch.full((batch, 1), start_id, dtype=torch.long, device=a.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=a.device)
    for _ in range(max_length - 1):
        token = model.decode_memory(memory, sequence)[:, -1].argmax(-1)
        token = torch.where(finished, torch.full_like(token, pad_id), token)
        sequence = torch.cat([sequence, token[:, None]], dim=1)
        finished |= token.eq(end_id)
        if finished.all():
            break
    return sequence.tolist()


class DinoCaptionM2(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        dino_module = load_module(cfg["dino_model_path"], "stage67_dino_model")
        encoder_checkpoint = torch.load(
            cfg["dino_encoder_checkpoint"], map_location="cpu", weights_only=False
        )
        self.encoder = dino_module.DinoRobustModel(encoder_checkpoint["config"])
        missing, unexpected = self.encoder.load_state_dict(
            encoder_checkpoint["trainable_state"], strict=False
        )
        if unexpected:
            raise RuntimeError(f"Unexpected DINO encoder keys: {unexpected}")
        # Anchors created by __init__ point to raw pretrained DINO.  Stage-2
        # must instead preserve the robust Stage-1 checkpoint just loaded.
        self.encoder.reset_backbone_anchor()

        # This stage intentionally contains runner/model.py for compatibility
        # with the Stage-4 feature extractor.  Python may cache that file as
        # the top-level module ``model``, which would shadow Change-Agent's
        # real ``model`` package.  Remove only the non-package collision and
        # prioritize the Change-Agent root before constructing its decoder.
        change_agent_root = str(Path(cfg["change_agent_root"]))
        model_package_path = str(Path(change_agent_root) / "model")
        cached_model = sys.modules.get("model")
        cached_paths = list(getattr(cached_model, "__path__", [])) if cached_model else []
        if model_package_path not in cached_paths:
            # Change-Agent's model directory has no __init__.py.  Without an
            # explicit namespace package, runner/model.py wins import
            # resolution as a regular module and breaks model.model_decoder.
            model_package = types.ModuleType("model")
            model_package.__package__ = "model"
            model_package.__path__ = [model_package_path]
            sys.modules["model"] = model_package
        if change_agent_root in sys.path:
            sys.path.remove(change_agent_root)
        sys.path.insert(0, change_agent_root)
        caption_module = load_module(cfg["dino_caption_model_path"], "stage67_caption_model")
        self.captioner = caption_module.ChangeCaptioner(
            Path(change_agent_root),
            vocab,
            max_length=int(cfg.get("max_length", 41)),
            feature_dim=512,
            heads=int(cfg.get("heads", 8)),
            layers=int(cfg.get("dino_decoder_layers", 3)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        caption_checkpoint = torch.load(
            cfg["dino_caption_checkpoint"], map_location="cpu", weights_only=False
        )
        self.captioner.load_state_dict(caption_checkpoint["model"], strict=True)

        if not bool(cfg.get("dino_train_encoder", False)):
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.max_length = int(cfg.get("max_length", 41))
        self.vocab = vocab
        self.missing_frozen_keys = len(missing)

    def features(self, a, b):
        images = torch.cat([a, b], dim=0)
        _, features = self.encoder.encode_images(images)
        return features.chunk(2, dim=0)

    def memory(self, a, b):
        fa, fb = self.features(a, b)
        return self.captioner.memory(fa, fb)

    def decode_memory(self, memory, tokens):
        return self.captioner.decode_logits(memory, tokens)

    def forward(self, a, b, tokens):
        return self.decode_memory(self.memory(a, b), tokens)

    @torch.inference_mode()
    def generate(self, a, b):
        return _greedy(
            self,
            a,
            b,
            self.vocab["<START>"],
            self.vocab["<END>"],
            self.vocab["<NULL>"],
            self.max_length,
        )


def build_model(cfg, vocab):
    if not str(cfg["architecture"]).startswith("dino_"):
        raise ValueError(f"Stage6.7 supports DINO caption models only: {cfg['architecture']}")
    return DinoCaptionM2(cfg, vocab)


def frozen_teacher(student):
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher
