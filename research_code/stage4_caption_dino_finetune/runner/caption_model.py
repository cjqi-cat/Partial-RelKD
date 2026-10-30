from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


class ChangeCaptioner(nn.Module):
    """Projection plus the original Change-Agent caption decoder."""

    def __init__(self, change_agent_root: Path, vocab: dict[str, int], max_length=41, feature_dim=512, heads=8, layers=3, dropout=0.1):
        super().__init__()
        sys.path.insert(0, str(change_agent_root))
        from model.model_decoder import DecoderTransformer
        self.projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.decoder = DecoderTransformer(feature_dim, feature_dim, len(vocab), max_length, vocab, heads, layers, dropout)
        self.vocab = vocab
        self.max_length = max_length

    def visual_maps(self, a: torch.Tensor, b: torch.Tensor):
        a, b = self.projection(a), self.projection(b)
        batch, patches, channels = a.shape
        side = int(patches ** 0.5)
        if side * side != patches:
            raise ValueError(f"Expected square patch grid, got {patches}")
        return a.transpose(1, 2).reshape(batch, channels, side, side), b.transpose(1, 2).reshape(batch, channels, side, side)

    def memory(self, a: torch.Tensor, b: torch.Tensor):
        a, b = self.visual_maps(a, b)
        return self.decoder.LN(self.decoder.Conv1(torch.cat([a, b], dim=1))).flatten(2).permute(2, 0, 1)

    def decode_logits(self, memory: torch.Tensor, tokens: torch.Tensor):
        length = tokens.shape[1]
        mask = torch.triu(torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1)
        pad = tokens.eq(self.vocab["<NULL>"])
        word = self.decoder.position_encoding(self.decoder.vocab_embedding(tokens).transpose(0, 1))
        prediction = self.decoder.transformer(word, memory, tgt_mask=mask, tgt_key_padding_mask=pad)
        return self.decoder.wdc(self.decoder.dropout(prediction)).permute(1, 0, 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor, tokens_in: torch.Tensor):
        return self.decode_logits(self.memory(a, b), tokens_in)

    @torch.inference_mode()
    def greedy(self, a: torch.Tensor, b: torch.Tensor):
        memory = self.memory(a, b)
        batch = a.shape[0]
        sequence = torch.full((batch, 1), self.vocab["<START>"], dtype=torch.long, device=a.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=a.device)
        for _ in range(self.max_length - 1):
            next_token = self.decode_logits(memory, sequence)[:, -1].argmax(-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.vocab["<NULL>"]), next_token)
            sequence = torch.cat([sequence, next_token[:, None]], dim=1)
            finished |= next_token.eq(self.vocab["<END>"])
            if finished.all():
                break
        return sequence
