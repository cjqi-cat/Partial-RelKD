# Core package

This package exposes the method components that are shared by the validated
experiment stages:

- `models.visual_adapter`: partial DINOv2 adaptation, residual adapter, GRL,
  and Stage-1/Stage-2 parameter anchors;
- `data.degradations`: the nine calibrated individual degradations and mixed
  composition used by training and evaluation;
- `losses.distillation`: caption cross-entropy, token KD, and relation-memory
  KD;
- `protocols`: the canonical atmospheric, geometry-quality, and noise-loss
  LOFO definitions.

The complete caption model still depends on the separately obtained
Change-Agent decoder. Validated end-to-end experiment drivers are retained
under `experiments/legacy_stages` and are restored to their original server
layout by `tools/materialize_workspace.py`.
