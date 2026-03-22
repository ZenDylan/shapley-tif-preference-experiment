"""
全局配置文件。
所有可调参数集中在本文件，方便修改。
"""

import os

# ===== 项目根目录 =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 模型配置 =====
MODEL_NAME = "EleutherAI/pythia-410m"
MODEL_DTYPE = "bfloat16"   # bf16 精度

# ===== 数据集配置 =====
SFT_DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
PREF_DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

# 偏好数据划分（均为分层采样）
N_TRAIN = 200    # D_train
N_VAL   = 100    # D_val
N_TEST  = 100    # D_test

# SFT 数据子集大小
N_SFT_SAMPLES = 10_000

# ===== 训练超参数 =====
SFT_TRAINING_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "max_seq_length": 512,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 50,
    "save_steps": 1000,
    "output_dir": os.path.join(CHECKPOINT_DIR, "sft_pythia410m"),
}

DPO_TRAINING_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "max_seq_length": 512,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
}

# ===== 分层采样配置 =====
STRATA_COUNT = 5   # margin 分数划分的档位数

# ===== 随机种子 =====
RANDOM_SEED = 42

# ===== 进度条配置 =====
TQDM_CONFIG = {
    "bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
}

# ===== 16GB RAM 告警阈值 =====
RAM_WARN_GB = 13.0   # RAM 使用超过此值打印警告
