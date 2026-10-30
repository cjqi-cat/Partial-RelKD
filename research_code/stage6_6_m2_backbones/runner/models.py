from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import torch
from torch import nn

from common import load_module


def _decoder_logits(decoder, memory, tokens):
    length = tokens.size(1)
    mask = torch.triu(
        torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1
    )
    padding = tokens.eq(decoder.word_vocab["<NULL>"])
    words = decoder.position_encoding(decoder.vocab_embedding(tokens).transpose(0, 1))
    prediction = decoder.transformer(words, memory, tgt_mask=mask, tgt_key_padding_mask=padding)
    return decoder.wdc(decoder.dropout(prediction)).permute(1, 0, 2)


def _greedy(model, a, b, start_id, end_id, pad_id, max_length):
    memory = model.memory(a, b)
    batch = a.size(0)
    sequence = torch.full((batch, 1), start_id, dtype=torch.long, device=a.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=a.device)
    for _ in range(max_length - 1):
        logits = model.decode_memory(memory, sequence)
        token = logits[:, -1].argmax(-1)
        token = torch.where(finished, torch.full_like(token, pad_id), token)
        sequence = torch.cat([sequence, token[:, None]], dim=1)
        finished |= token.eq(end_id)
        if finished.all():
            break
    return sequence.tolist()


class ChangeAgentM2(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        root = str(Path(cfg["change_agent_root"]))
        if root not in sys.path:
            sys.path.insert(0, root)
        encoder_module = importlib.import_module("model.model_encoder_att")
        decoder_module = importlib.import_module("model.model_decoder")
        self.encoder = encoder_module.Encoder(cfg.get("ca_network", "segformer-mit_b1"))
        self.neck = encoder_module.AttentiveEncoder(
            train_stage=None,
            n_layers=int(cfg.get("ca_neck_layers", 3)),
            feature_size=[16, 16, 512],
            heads=int(cfg.get("heads", 8)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        self.decoder = decoder_module.DecoderTransformer(
            encoder_dim=512,
            feature_dim=512,
            vocab_size=len(vocab),
            max_lengths=int(cfg.get("max_length", 41)),
            word_vocab=vocab,
            n_head=int(cfg.get("heads", 8)),
            n_layers=int(cfg.get("ca_decoder_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
        )
        checkpoint = torch.load(cfg["ca_checkpoint"], map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder_dict"], strict=True)
        # The released MCI checkpoint stores a superset of neck branches.  The
        # official Change-Agent inference path deliberately loads this block
        # with strict=False and ignores branches that are absent from the
        # selected AttentiveEncoder implementation.  Preserve that behavior,
        # but fail if a parameter required by the selected implementation is
        # missing (unexpected superset keys are safe to ignore).
        neck_state = self.neck.load_state_dict(
            checkpoint["encoder_trans_dict"], strict=False
        )
        if neck_state.missing_keys:
            raise RuntimeError(
                "MCI checkpoint is missing required AttentiveEncoder keys: "
                f"{neck_state.missing_keys}"
            )
        self.ca_ignored_neck_keys = len(neck_state.unexpected_keys)
        self.decoder.load_state_dict(checkpoint["decoder_dict"], strict=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = bool(cfg.get("ca_train_encoder", False))
        self.neck.fine_tune(1, True)
        self.decoder.fine_tune(True)
        self.max_length = int(cfg.get("max_length", 41))
        self.vocab = vocab

    def visual(self, a, b):
        list_a, list_b = self.encoder(a, b)
        a, b, _ = self.neck.CC_neck_s0(list_a[-1], list_b[-1])
        a, b, _ = self.neck.prepare_caption(a, b, None)
        return a, b

    def memory(self, a, b):
        a, b = self.visual(a, b)
        fused = self.decoder.LN(self.decoder.Conv1(torch.cat([a, b], dim=1)))
        return fused.flatten(2).permute(2, 0, 1)

    def decode_memory(self, memory, tokens):
        return _decoder_logits(self.decoder, memory, tokens)

    def forward(self, a, b, tokens):
        return self.decode_memory(self.memory(a, b), tokens)

    @torch.inference_mode()
    def generate(self, a, b):
        return _greedy(
            self, a, b, self.vocab["<START>"], self.vocab["<END>"],
            self.vocab["<NULL>"], self.max_length,
        )


class DinoM2(nn.Module):
    def __init__(self, cfg, vocab):
        super().__init__()
        dino_module = load_module(cfg["dino_model_path"], "stage66_dino_model")
        encoder_checkpoint = torch.load(
            cfg["dino_encoder_checkpoint"], map_location="cpu", weights_only=False
        )
        self.encoder = dino_module.DinoRobustModel(encoder_checkpoint["config"])
        missing, unexpected = self.encoder.load_state_dict(
            encoder_checkpoint["trainable_state"], strict=False
        )
        if unexpected:
            raise RuntimeError(f"Unexpected DINO encoder keys: {unexpected}")

        caption_module = load_module(cfg["dino_caption_model_path"], "stage66_caption_model")
        self.captioner = caption_module.ChangeCaptioner(
            Path(cfg["change_agent_root"]), vocab,
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
            self, a, b, self.vocab["<START>"], self.vocab["<END>"],
            self.vocab["<NULL>"], self.max_length,
        )


def build_model(cfg, vocab):
    architecture = cfg["architecture"]
    if architecture == "change_agent":
        return ChangeAgentM2(cfg, vocab)
    if architecture in {"dino_lora", "dino_partial_last2"}:
        return DinoM2(cfg, vocab)
    raise ValueError(f"Unsupported architecture: {architecture}")


def frozen_teacher(student):
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher
