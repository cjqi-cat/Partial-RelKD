# Dataset setup

Datasets are not redistributed by this repository. Obtain them from their
official providers and comply with their licenses.

## LEVIR-MCI

Expected image layout after materialization:

```text
<data-root>/LEVIR-MCI-dataset/
  images/
    train/A/*.png
    train/B/*.png
    val/A/*.png
    val/B/*.png
    test/A/*.png
    test/B/*.png
  label/*.png                 # Stage-1 training masks
```

The Change-Agent metadata directory must provide:

```text
<change-agent-root>/data/LEVIR_MCI/
  train.txt
  val.txt
  test.txt
  tokens/
  vocab.json
```

The release does not alter the official splits, captions, vocabulary, masks,
or change/no-change definition.

## SECOND-CC and WHU-CDC

The scripts under `stage8_1_multidataset_full` build dataset-specific views and
metadata. Each experiment uses the target dataset's own training images,
captions, vocabulary, and train/validation/test split. These are
target-supervised reproductions, not zero-shot transfer experiments.

Because local extractions may use different folder names, inspect
`runner/prepare_views.py` before materializing and pass the official dataset
roots used on your machine.

## Generated degradations

Degradations are generated online from clean images. The deterministic test
seed is `20260928`; the seed additionally includes degradation type, target
side, image name, and temporal side. Do not commit generated degraded images or
feature caches.
