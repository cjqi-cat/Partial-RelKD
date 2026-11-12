from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import nn


def decoder_logits(decoder, memory, tokens):
    length = tokens.size(1)
    mask = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1)
    padding = tokens.eq(decoder.word_vocab["<NULL>"])
    words = decoder.position_encoding(decoder.vocab_embedding(tokens).transpose(0, 1))
    prediction = decoder.transformer(words, memory, tgt_mask=mask, tgt_key_padding_mask=padding)
    return decoder.wdc(decoder.dropout(prediction)).permute(1, 0, 2)


class TargetChangeAgent(nn.Module):
    """Change-Agent visual initialization with a target-vocabulary decoder."""

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
        # Reuse only the released visual representation.  The decoder is
        # deliberately new because every target dataset owns its vocabulary.
        checkpoint = torch.load(cfg["ca_checkpoint"], map_location="cpu", weights_only=False)
        self.encoder.load_state_dict(checkpoint["encoder_dict"], strict=True)
        neck_state = self.neck.load_state_dict(checkpoint["encoder_trans_dict"], strict=False)
        if neck_state.missing_keys:
            raise RuntimeError(f"Missing Change-Agent neck keys: {neck_state.missing_keys}")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = bool(cfg.get("ca_train_encoder", True))
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
        return decoder_logits(self.decoder, memory, tokens)

    def forward(self, a, b, tokens):
        return self.decode_memory(self.memory(a, b), tokens)

    @torch.inference_mode()
    def generate(self, a, b):
        memory = self.memory(a, b)
        batch = a.size(0)
        sequence = torch.full((batch, 1), self.vocab["<START>"], dtype=torch.long, device=a.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=a.device)
        for _ in range(self.max_length - 1):
            token = self.decode_memory(memory, sequence)[:, -1].argmax(-1)
            token = torch.where(finished, torch.full_like(token, self.vocab["<NULL>"]), token)
            sequence = torch.cat([sequence, token[:, None]], dim=1)
            finished |= token.eq(self.vocab["<END>"])
            if finished.all():
                break
        return sequence.tolist()


def build_model(cfg, vocab):
    return TargetChangeAgent(cfg, vocab)
