from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
        return self.base(x) + residual * self.scale


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.scale * grad, None


class DinoRobustModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        import timm

        self.cfg = cfg
        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=True,
            pretrained_cfg_overlay={"hf_hub_id": None},
            img_size=224,
            num_classes=0,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        mode = cfg["finetune_mode"]
        if mode == "partial_last2":
            count = int(cfg.get("unfreeze_last_blocks", 2))
            if not 1 <= count <= len(self.backbone.blocks):
                raise ValueError(f"Invalid unfreeze_last_blocks={count}")
            for block in self.backbone.blocks[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True
        elif mode == "lora":
            count = int(cfg.get("lora_last_blocks", len(self.backbone.blocks)))
            selected = self.backbone.blocks[-count:]
            for block in selected:
                block.attn.qkv = LoRALinear(
                    block.attn.qkv,
                    rank=int(cfg.get("lora_rank", 8)),
                    alpha=float(cfg.get("lora_alpha", 16.0)),
                    dropout=float(cfg.get("lora_dropout", 0.05)),
                )
        else:
            raise ValueError(f"Unsupported finetune_mode={mode!r}")

        if bool(cfg.get("gradient_checkpointing", True)) and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)

        dim = 768
        hidden = int(cfg["adapter_hidden"])
        self.adapter_norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.adapter_scale = float(cfg["adapter_scale"])
        self.change_head = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.degradation_head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 10))

        self._anchor_names = []
        for index, (name, parameter) in enumerate(
            (item for item in self.backbone.named_parameters() if item[1].requires_grad)
        ):
            buffer_name = f"_anchor_{index}"
            self.register_buffer(buffer_name, parameter.detach().clone(), persistent=False)
            self._anchor_names.append((name, buffer_name))

    def encode_images(self, images: torch.Tensor):
        features = self.backbone.forward_features(images)
        prefix = int(self.backbone.num_prefix_tokens)
        raw = F.normalize(features[:, prefix:].float(), dim=-1)
        adapted = F.normalize(raw + self.adapter_scale * self.adapter(self.adapter_norm(raw)), dim=-1)
        return raw, adapted

    def change_logits(self, a: torch.Tensor, b: torch.Tensor):
        return self.change_head(torch.cat([torch.abs(b - a), a * b], dim=-1)).squeeze(-1)

    def backbone_anchor_loss(self):
        losses = []
        parameters = dict(self.backbone.named_parameters())
        for name, buffer_name in self._anchor_names:
            parameter = parameters[name]
            reference = getattr(self, buffer_name)
            losses.append((parameter.float() - reference.float()).pow(2).mean())
        if not losses:
            return next(self.parameters()).new_zeros(())
        return torch.stack(losses).mean()

    def parameter_report(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        backbone_trainable = sum(parameter.numel() for parameter in self.backbone.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable, "backbone_trainable": backbone_trainable}
