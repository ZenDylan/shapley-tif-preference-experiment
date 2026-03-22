# Shapley Value vs TIF 偏好数据估值对比实验

## 项目概述

本项目复现并扩展论文《Towards Understanding Valuable Preference Data for LLM Alignment》的实验。使用 Shapley Value（通过 Stratified SVARM 算法近似）替代 Truncated Influence Function (TIF) 对偏好数据进行估值，并对比两种方法的排序结果。

## 实验阶段

| 阶段 | 说明 | 运行方式 |
|------|------|---------|
| 阶段一 | 环境搭建与数据准备 | 实际运行 |
| 阶段二 | 计算 TIF/LossDiff/IRM 分数 | 实际运行 |
| 阶段三 | 计算 Shapley Value | 仅写代码 |
| 阶段四 | 对比分析与可视化 | 仅写代码 |

## 硬件要求

- **GPU**: NVIDIA RTX 4060 (8GB VRAM)
- **CPU RAM**: 16GB（硬约束）
- 训练使用 LoRA (rank=16, alpha=32)，batch_size=1，bf16 精度

## 环境安装

**Windows (RTX 4060):**
双击运行 `run.bat`，或手动依次执行：

```bash
cd project
pip install -r requirements.txt
python stage1_setup/prepare_data.py
python stage1_setup/train_sft.py
python stage2_tif/compute_tif.py
python stage2_tif/compute_lossdiff_irm.py
```

**Linux/macOS:**
```bash
bash run.sh
```

## 各阶段运行顺序

```bash
# 阶段一：数据准备
python stage1_setup/prepare_data.py
python stage1_setup/train_sft.py

# 阶段二：TIF/LossDiff/IRM 计算
python stage2_tif/compute_tif.py
python stage2_tif/compute_lossdiff_irm.py
```

## 文件结构

```
project/
├── requirements.txt
├── README.md
├── configs/                    # 配置文件
│   ├── config.py
│   └── lora_config.py
├── utils/                      # 工具函数
│   ├── memory.py
│   └── io.py
├── data/                       # 预处理后的数据
├── checkpoints/                # 训练产物（SFT checkpoint 等）
├── outputs/                    # 阶段输出（IF 分数、Shapley Value 等）
├── stage1_setup/               # 阶段一
│   ├── prepare_data.py
│   └── train_sft.py
├── stage2_tif/                 # 阶段二
│   ├── compute_tif.py
│   └── compute_lossdiff_irm.py
├── stage3_shapley/             # 阶段三（仅写代码）
│   ├── preference_game.py
│   └── compute_shapley.py
└── stage4_analysis/             # 阶段四（仅写代码）
    └── compare.py
```

## 数据集

- **SFT 数据**: HuggingFaceH4/ultrachat_200k (约 10k 条子集)
- **偏好数据**: HuggingFaceH4/ultrafeedback_binarized (训练 200 / 验证 100 / 测试 100)

## 模型

- **基础模型**: EleutherAI/pythia-410m
- **训练方式**: LoRA (rank=16, alpha=32) + bf16 + gradient_checkpointing

## 注意事项

1. 16GB RAM 是硬约束，所有梯度计算必须只对 LoRA 参数进行
2. 每条数据的全参数梯度 = 1.6GB，LoRA 梯度 = 8MB
3. 内存紧张时立即写磁盘、清空缓存
4. 所有随机种子固定为 42，确保可复现
