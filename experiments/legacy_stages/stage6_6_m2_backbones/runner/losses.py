from __future__ import annotations

import torch
from torch.nn import functional as F


def caption_loss(logits, targets, pad_id, label_smoothing=0.05):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
    )


def token_kd(student_logits, teacher_logits, targets, pad_id, temperature=2.0):
    mask = targets.ne(pad_id)
    student = F.log_softmax(student_logits / temperature, dim=-1)
    teacher = F.softmax(teacher_logits / temperature, dim=-1)
    per_token = F.kl_div(student, teacher, reduction="none").sum(-1)
    return (per_token * mask).sum() / mask.sum().clamp_min(1) * temperature**2
