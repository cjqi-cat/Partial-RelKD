# Third-party components

Third-party repositories, datasets, and pretrained weights are not vendored in
this release. Clone them separately and comply with their own licenses and
dataset terms. The license eventually selected for Partial-RelKD will cover
only original code in this repository.

## Required for the main method

| Component | Upstream | Pinned revision/version | License status |
|---|---|---|---|
| Change-Agent | https://github.com/Chen-Yang-Liu/Change-Agent | `68cbaa7f388b36e4fc10872f7a2911482d26ae5b` | MIT license in upstream repository |
| DINOv2 | https://github.com/facebookresearch/dinov2 | ViT-B/14 pretrained weights used through timm | Apache-2.0 upstream; pretrained-weight terms also apply |
| timm | https://github.com/huggingface/pytorch-image-models | `timm>=1.0,<1.1` | Apache-2.0 |

Change-Agent supplies the caption decoder, BLEU/ROUGE/CIDEr implementation,
and LEVIR-MCI token/list interface expected by the experiment wrappers.

## Optional authoritative-baseline reproduction

| Model/codebase | Upstream | Pinned commit | License status at audit time |
|---|---|---|---|
| Chg2Cap | https://github.com/ShizhenChang/Chg2Cap | `7b8cda937002e614d51a6dab3d949aa7de77c176` | MIT |
| MCCFormer-D and RSICCFormer | https://github.com/Chen-Yang-Liu/RSICC | `d1505e514c450c3728782ca723e82761e70bafd3` | No license file detected; do not redistribute without permission |
| SEN | https://github.com/mrazhou/SEN | `42b97b62ff8ccd57938d371ff0affec59fc13410` | No license file detected; do not redistribute without permission |
| RSCaMa (optional wrapper retained for ablation compatibility) | https://github.com/Chen-Yang-Liu/RSCaMa | `aa68e4c6933c7dd0f2e2f9cbd7642d61929c86a5` | No license file detected; do not redistribute without permission |
| OpenAI CLIP | https://github.com/openai/CLIP | install from upstream when required | MIT |

The `stage10_authoritative_baselines` and
`stage10_9_three_dataset_baselines` directories contain only local experiment
wrappers and configuration logic. They do not grant permission to redistribute
the external implementations or checkpoints. For repositories without an
explicit license, users must obtain the code directly from the upstream owner
and use it only under permissions they have independently secured.

## Data

LEVIR-MCI, SECOND-CC, and WHU-CDC are excluded. Users must obtain each dataset
from its official source and follow its license or access terms. See
`DATASETS.md` for the expected local layouts.
