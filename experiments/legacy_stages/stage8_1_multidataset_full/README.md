# Target-supervised reproduction on SECOND-CC and WHU-CDC

This stage independently trains Change-Agent, M2, and Partial-RelKD using each
target dataset's own training images, captions, vocabulary, and official
splits. It evaluates method-level reproducibility under the same
geometry-quality LOFO protocol; it is not a zero-shot transfer experiment.

Raw data are expected at `__DATA_ROOT__/SECOND-CC-AUG` and
`__DATA_ROOT__/WHU_CDC/whu_CDC_dataset`. First generate list/token/vocabulary
metadata with `runner/prepare_metadata.py`, then construct the common data
views:

```bash
python runner/prepare_metadata.py --name SECOND_CC \
  --dataset-root __DATA_ROOT__/SECOND-CC-AUG \
  --annotation __DATA_ROOT__/SECOND-CC-AUG/SECOND-CC-AUG.json \
  --layout secondcc --output metadata/SECOND_CC --expected 8438,1190,1227

python runner/prepare_metadata.py --name WHU_CDC \
  --dataset-root __DATA_ROOT__/WHU_CDC/whu_CDC_dataset \
  --annotation __DATA_ROOT__/WHU_CDC/whu_CDC_dataset/WHU_CDC.json \
  --layout standard --output metadata/WHU_CDC --expected 5947,743,744 \
  --require-label

python runner/prepare_views.py
python runner/make_configs.py
python runner/preflight.py
bash run_all.sh
```

Adjust only the WHU annotation filename if the distributed archive uses a
different name; do not change its official split assignments.
