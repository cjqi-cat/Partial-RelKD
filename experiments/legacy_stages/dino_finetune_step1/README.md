# DINOv2 真正微调：Step 1.5

本目录独立于旧的 `stage2_contrastive_dino`，不会覆盖原始 Adapter 结果。本轮只完成导师工作流的第一步扩展：让梯度进入 DINOv2 主干内部。

## 实验组

1. `dino_lora_qkv_all`：冻结原始权重，在12个Transformer Block的Attention QKV层加入rank=8 LoRA，同时训练原有Residual Adapter和辅助头。
2. `dino_partial_last2`：训练DINOv2最后2个Transformer Block及最终LayerNorm，同时训练Residual Adapter和辅助头。

两组均保留旧实验的数据与目标：LEVIR-MCI train 4,000对、val 800对，每对图像生成clean/deg1/deg2六视图，使用10类退化和1–3级随机强度。

## 损失

`L = L_inv + L_change + 0.5 L_patch_cls + 0.1 L_distill + 0.15 L_adv + 0.001 L_anchor`

- `L_inv`：patch余弦不变性与global InfoNCE；
- `L_change`：无变化patch拉近、变化patch至少保持margin；
- `L_patch_cls`：变化掩码二分类；
- `L_distill`：Adapter输出保持DINO语义；
- `L_adv`：梯度反转，压低退化类型信息；
- `L_anchor`：可训练DINO/LoRA参数不要过度偏离初始化。

## 服务器要求

推荐使用48GB显存的CUDA GPU。24GB显存可以尝试，但LoRA配置可能需要把 `batch_size_pairs` 从2降到1；由于InfoNCE在batch=1时失效，更推荐保持batch≥2。

全新服务器只需要准备：

- 数据：`__DATA_ROOT__/LEVIR-MCI-dataset`
- 退化脚本已包含在本工程的 `runner/degradations.py`，不依赖旧服务器工程；
- DINOv2权重缓存：`~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`

## 执行

```bash
cd __PROJECT_ROOT__/dino_finetune_step1
chmod +x *.sh
python runner/self_test.py
bash start_server.sh
bash status_server.sh
```

断线不会终止训练。再次执行 `start_server.sh` 不会重复启动已有进程；已有 `last.pt` 时会从断点续跑；已经完成的实验会自动跳过。

## 输出

- `runs/<variant>/checkpoints/best.pt`：最优可训练参数；
- `runs/<variant>/checkpoints/last.pt`：安全续跑断点；
- `runs/<variant>/results/training_history.csv`：逐epoch损失和AUC；
- `runs/<variant>/results/summary.json`：完成状态和最优分数；
- `logs/pipeline.log`：总进度。

训练完成后再运行统一的固定退化评测，并与旧的Raw/Control/Full Adapter比较。不要依据test结果反复调参。
