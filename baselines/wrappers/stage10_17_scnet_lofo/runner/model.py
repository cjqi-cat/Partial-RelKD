from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


class SCNet(nn.Module):
    """Thin protocol adapter around the official SCNet implementation."""

    def __init__(self, cfg, vocab):
        super().__init__()
        root = Path(cfg["scnet_root"])
        # This adapter itself is imported as ``model`` by the runner.  Loading
        # the upstream ``model`` package normally would therefore collide with
        # this module.  Give the two official files private module names.
        def load_source(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        enc = load_source("scnet_official_encoder", root / "model" / "model_encoder.py")
        dec = load_source("scnet_official_decoder", root / "model" / "model_decoder.py")
        Encoder, AttentiveEncoder = enc.Encoder, enc.AttentiveEncoder
        DecoderTransformer = dec.DecoderTransformer

        self.encoder = Encoder(cfg.get("network", "resnet101"))
        self.encoder.fine_tune(bool(cfg.get("fine_tune_encoder", True)))
        self.relation = AttentiveEncoder(
            n_layers=int(cfg.get("relation_layers", 3)),
            feature_size=[int(cfg["feature_size"]), int(cfg["feature_size"]), int(cfg.get("encoder_dim", 2048))],
            heads=int(cfg.get("heads", 8)), hidden_dim=int(cfg.get("hidden_dim", 2048)),
            attention_dim=int(cfg.get("feature_dim", 512)), dropout=float(cfg.get("dropout", .1)))
        self.decoder = DecoderTransformer(
            encoder_dim=int(cfg.get("encoder_dim", 2048)), feature_dim=int(cfg.get("feature_dim", 512)),
            vocab_size=len(vocab), max_lengths=int(cfg.get("max_length", 41)), word_vocab=vocab,
            n_head=int(cfg.get("heads", 8)), n_layers=int(cfg.get("decoder_layers", 1)),
            dropout=float(cfg.get("dropout", .1)))

    def memory(self, a, b):
        fa, fb = self.encoder(a, b)
        return self.relation(fa, fb)

    def forward(self, a, b, tokens, lengths):
        fa, fb = self.memory(a, b)
        return self.decoder(fa, fb, tokens, lengths)

    @torch.no_grad()
    def greedy(self, a, b):
        # The upstream sample routine is faithful but only safe for batch size 1.
        if a.size(0) != 1:
            return torch.stack([self.greedy(a[i:i+1], b[i:i+1]) for i in range(a.size(0))])
        fa, fb = self.memory(a, b)
        # The upstream ``sample`` converts its output to a Python list, whereas
        # ``sample_fast`` implements the same greedy policy and keeps it on GPU.
        return self.decoder.sample_fast(fa, fb, k=1)[0]


def build_model(cfg, vocab):
    return SCNet(cfg, vocab)
