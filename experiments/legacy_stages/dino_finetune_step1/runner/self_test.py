#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn

from model import GradientReverse, LoRALinear


def main():
    torch.manual_seed(7)
    base = nn.Linear(5, 9)
    layer = LoRALinear(base, rank=2, alpha=4.0)
    x = torch.randn(3, 5, requires_grad=True)
    before = layer(x)
    assert torch.allclose(before, base(x), atol=1e-6), "zero-init LoRA must initially match base"
    before.sum().backward()
    assert layer.lora_b.grad is not None
    assert all(not p.requires_grad for p in layer.base.parameters())
    y = torch.randn(2, 3, requires_grad=True)
    GradientReverse.apply(y, 0.5).sum().backward()
    assert torch.allclose(y.grad, torch.full_like(y, -0.5))
    print("SELF_TEST_PASS")


if __name__ == "__main__":
    main()
