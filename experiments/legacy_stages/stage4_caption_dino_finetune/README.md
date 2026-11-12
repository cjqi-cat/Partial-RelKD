# Stage 4：LoRA与Partial-last2变化描述重训

本阶段冻结已经训练好的两种视觉编码器，分别提取clean训练/验证特征，使用与旧Stage 3完全相同的Change-Agent Decoder、词表、数据划分、交叉熵目标和早停规则训练两个Caption模型，最后在clean+30个退化条件上评测BLEU-4、ROUGE-L和CIDEr。

## 两个模型

- `lora`：DINOv2 12层QKV-LoRA + Residual Adapter；
- `partial_last2`：DINOv2最后2个Block + Residual Adapter。

视觉编码器在Caption训练期间完全冻结。缓存中已经是适配后的patch特征，Decoder前不再重复套Adapter。

## 数据与输出规模

- train 6,815对，val 1,333对，test 1,929对；
- 每种视觉缓存约6–7GB，两种约12GB；
- Caption检查点约75–80MB/模型；
- 评测31条件/模型，支持按条件续跑。

## 服务器依赖

服务器必须已有：

- `__DATA_ROOT__/LEVIR-MCI-dataset`；
- `__PROJECT_ROOT__/dino_finetune_step1`及两个best checkpoint；
- `__PROJECT_ROOT__/stage3_caption_retrain/vendor/Change-Agent`。

## 启动

先按操作说明做两种编码器各8对图像的特征冒烟测试，再执行：

```bash
chmod +x *.sh
bash start_server.sh
bash status_server.sh
```

断线不会停止，重复启动不会创建第二个pipeline。特征缓存、Caption检查点和逐条件评测均会被检测并跳过已经完成的部分。
