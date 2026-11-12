from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn


class RMNet(nn.Module):
    """Protocol adapter retaining the official CNN-VMamba/recalibration network."""

    def __init__(self, cfg, vocab):
        super().__init__()
        root = Path(cfg["rmnet_root"])
        sys.path.insert(0, str(root))
        from vmamba.fusion_backbone import FusionModule, fusion_both_3stage_concat_with_sc_config
        from models_RSICCformerDfusion import MCCFormers_diff_as_Q, DecoderTransformer

        backbone = copy.deepcopy(fusion_both_3stage_concat_with_sc_config)
        backbone["vmamba_config"]["model_config"]["pretrained"] = cfg["vmamba_checkpoint"]
        # Upstream hard-codes a private ResNet path. ImageNet initialization is the
        # public equivalent implemented by the same official extractor.
        backbone["resnet_config"]["pretrained"] = True
        self.image_encoder = FusionModule(**backbone)
        self.image_encoder.fine_tune(bool(cfg.get("fine_tune_encoder", False)))
        self.relation = MCCFormers_diff_as_Q(
            feature_dim=1024, dropout=float(cfg.get("dropout", .5)), h=14, w=14, d_model=512,
            n_head=int(cfg.get("heads", 8)), n_layers=int(cfg.get("relation_layers", 3)),
            cross_trans_type=cfg.get("cross_trans_type", "bidi"))
        self.decoder = DecoderTransformer(feature_dim=1024, vocab_size=len(vocab),
                                          n_head=int(cfg.get("heads", 8)),
                                          n_layers=int(cfg.get("decoder_layers", 1)),
                                          dropout=float(cfg.get("dropout", .5)))
        self.vocab = vocab
        self.max_length = int(cfg.get("max_length", 41))

    def memory(self, a, b):
        return self.relation(self.image_encoder(a), self.image_encoder(b))

    def forward(self, a, b, tokens, lengths):
        return self.decoder(self.memory(a, b), tokens, lengths[:, None])

    @torch.no_grad()
    def greedy(self, a, b):
        if a.size(0) != 1:
            return torch.stack([self.greedy(a[i:i+1], b[i:i+1]) for i in range(a.size(0))])
        memory = self.memory(a, b)
        pad, start, end = self.vocab["<NULL>"], self.vocab["<START>"], self.vocab["<END>"]
        tokens = torch.full((1, self.max_length), pad, dtype=torch.long, device=a.device)
        tokens[:, 0] = start
        for step in range(self.max_length - 1):
            tgt = tokens.T
            length = tgt.size(0)
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=a.device), diagonal=1)
            emb = self.decoder.position_encoding(self.decoder.vocab_embedding(tgt))
            pred = self.decoder.transformer(emb, memory, tgt_mask=mask)
            nxt = self.decoder.wdc(pred)[step, 0].argmax(-1)
            tokens[0, step + 1] = nxt
            if int(nxt) == end: break
        return tokens[0]


def build_model(cfg, vocab):
    return RMNet(cfg, vocab)
