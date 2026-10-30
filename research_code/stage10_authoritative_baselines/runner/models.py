from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn


def _vendor(cfg, name):
    root = str(Path(cfg["project_root"]) / "vendor" / name)
    if root not in sys.path: sys.path.insert(0, root)
    return root


def _log_file(cfg, name):
    path = Path(cfg["output_root"]) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


class CaptionBaseline(nn.Module):
    length_column = False

    def forward(self, a, b, tokens, lengths):
        memory = self.encode(a, b)
        return self.decode(memory, tokens, lengths)

    def greedy(self, a, b, start, end, pad, max_length):
        memory = self.encode(a, b)
        batch = a.size(0)
        seq = torch.full((batch, max_length), pad, dtype=torch.long, device=a.device)
        seq[:, 0] = start
        finished = torch.zeros(batch, dtype=torch.bool, device=a.device)
        for step in range(max_length - 1):
            lengths = torch.full((batch,), step + 2, dtype=torch.long, device=a.device)
            scores, _, _, order = self.decode(memory, seq, lengths)
            # Equal lengths make the sort stable on supported PyTorch versions; invert defensively.
            inv = torch.argsort(order)
            nxt = scores[inv, step].argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
            seq[:, step + 1] = nxt
            finished |= nxt.eq(end)
            if finished.all(): break
        return seq


class SENBaseline(CaptionBaseline):
    def __init__(self, cfg, vocab):
        super().__init__()
        _vendor(cfg, "sen")
        os.environ["model_stage"] = str(cfg.get("model_stage", 4))
        os.environ["log_path"] = _log_file(cfg, "sen_backbone.log")
        from modules.SEN import SEN
        weight = cfg.get("sen_weight_path", "resnet50_B6_imagenet")
        args = Namespace(weight_path=weight, model_stage=cfg.get("model_stage", 4),
                         proj_channel=512, ft_layer=cfg.get("sen_ft_layer", 7),
                         encoder_n_layers=cfg.get("encoder_layers", 3), vocab_size=len(vocab),
                         n_heads=cfg.get("heads", 8), dropout=cfg.get("dropout", .3),
                         decoder_n_layers=cfg.get("decoder_layers", 3))
        self.net = SEN(args)
        self.length_column = True

    def encode(self, a, b):
        images = torch.cat((a, b), 1)
        return self.net.encoder(images, self.net.extractor(images))

    def decode(self, memory, tokens, lengths):
        return self.net.decoder(memory, tokens, lengths[:, None])


class RSICCFamily(CaptionBaseline):
    def __init__(self, cfg, vocab, mcc=False):
        super().__init__()
        _vendor(cfg, "sen")
        os.environ["log_path"] = _log_file(cfg, "cnn_backbone.log")
        from modules.RSICCformer import CNN_Encoder, MCCFormers_diff_as_Q, TransDecoder
        from modules.MCCFormer import MCCFormers_D
        self.backbone = CNN_Encoder("resnet101")
        for p in self.backbone.parameters(): p.requires_grad = False
        common = dict(feature_dim=1024, dropout=cfg.get("dropout", .3), h=14, w=14,
                      d_model=512, n_head=cfg.get("heads", 8), n_layers=cfg.get("encoder_layers", 3))
        self.fusion = MCCFormers_D(**common) if mcc else MCCFormers_diff_as_Q(**common)
        self.decoder = TransDecoder(feature_dim=1024, vocab_size=len(vocab),
                                    n_head=cfg.get("heads", 8), n_layers=cfg.get("decoder_layers", 3),
                                    dropout=cfg.get("dropout", .3))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, a, b):
        with torch.no_grad(): fa, fb = self.backbone(a), self.backbone(b)
        return self.fusion(fa, fb)

    def decode(self, memory, tokens, lengths):
        return self.decoder(memory, tokens, lengths[:, None])


class Chg2CapBaseline(CaptionBaseline):
    def __init__(self, cfg, vocab):
        super().__init__()
        _vendor(cfg, "chg2cap")
        from model.model_encoder import Encoder, AttentiveEncoder
        from model.model_decoder import DecoderTransformer
        self.backbone = Encoder("resnet101")
        self.fusion = AttentiveEncoder(cfg.get("encoder_layers", 3), [8, 8, 2048],
                                       cfg.get("heads", 8), 2048, 512, cfg.get("dropout", .3))
        self.decoder = DecoderTransformer(2048, 512, len(vocab), cfg.get("max_length", 41), vocab,
                                          cfg.get("heads", 8), cfg.get("decoder_layers", 3),
                                          cfg.get("dropout", .3))

    def encode(self, a, b):
        fa, fb = self.backbone(a, b)
        return self.fusion(fa, fb)

    def decode(self, memory, tokens, lengths):
        return self.decoder(memory[0], memory[1], tokens, lengths)


class CLIPPatchEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        import clip
        model, _ = clip.load(model_name, jit=False)
        self.visual = model.visual.float()
        for p in self.visual.parameters(): p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def one(self, x):
        v = self.visual
        x = v.conv1(x).reshape(x.shape[0], v.conv1.out_channels, -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device, dtype=x.dtype)
        x = torch.cat((cls, x), 1) + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x).permute(1, 0, 2)
        x = v.transformer(x).permute(1, 0, 2)
        return v.ln_post(x[:, 1:])

    def forward(self, a, b):
        with torch.no_grad(): return self.one(a), self.one(b)


class RSCaMaBaseline(CaptionBaseline):
    def __init__(self, cfg, vocab):
        super().__init__()
        _vendor(cfg, "rscama")
        from model.model_encoder_attMamba import AttentiveEncoder
        from model.model_decoder import DecoderTransformer
        self.backbone = CLIPPatchEncoder(cfg.get("clip_model", "ViT-B/32"))
        self.fusion = AttentiveEncoder(cfg.get("encoder_layers", 3), [7, 7, 768], cfg.get("heads", 8),
                                       cfg.get("dropout", .3))
        self.decoder = DecoderTransformer("transformer_decoder", 768, len(vocab), cfg.get("max_length", 41),
                                          vocab, cfg.get("heads", 8), cfg.get("decoder_layers", 3),
                                          cfg.get("dropout", .3))

    def encode(self, a, b):
        fa, fb = self.backbone(a, b)
        return self.fusion(fa, fb)

    def decode(self, memory, tokens, lengths):
        return self.decoder(memory, tokens, lengths)


def build_model(cfg, vocab):
    name = cfg["model"]
    if name == "sen": return SENBaseline(cfg, vocab)
    if name == "rsiccformer": return RSICCFamily(cfg, vocab, False)
    if name == "mccformer_d": return RSICCFamily(cfg, vocab, True)
    if name == "chg2cap": return Chg2CapBaseline(cfg, vocab)
    if name == "rscama": return RSCaMaBaseline(cfg, vocab)
    raise ValueError(name)
