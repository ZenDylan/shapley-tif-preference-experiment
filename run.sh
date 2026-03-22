#!/bin/bash
# 运行脚本：自动安装依赖 + 依次执行阶段一和阶段二
# 用法: bash run.sh

set -e  # 遇到错误立即退出

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================="
echo "Shapley vs TIF 实验 - 自动运行脚本"
echo "项目目录: $PROJECT_DIR"
echo "=============================================="

# ---- 检查 GPU ----
echo ""
echo "[环境检查]"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "CUDA 可用: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
else
    echo "警告: 未检测到 NVIDIA GPU 或 nvidia-smi！"
    echo "RTX 4060 实验必须在有 NVIDIA GPU 的机器上运行。"
    exit 1
fi

# ---- 安装依赖 ----
echo ""
echo "[1/4] 安装 Python 依赖..."
pip install -r requirements.txt

# ---- 阶段一：数据准备 ----
echo ""
echo "[2/4] 阶段一：数据准备..."
echo "----------------------------------------------"
python stage1_setup/prepare_data.py
echo "----------------------------------------------"
echo "阶段一-数据准备完成!"

# ---- 阶段一：SFT 训练 ----
echo ""
echo "[3/4] 阶段一：SFT 训练..."
echo "（这可能需要 30-120 分钟，取决于 GPU 速度）"
echo "----------------------------------------------"
python stage1_setup/train_sft.py
echo "----------------------------------------------"
echo "阶段一-SFT训练完成!"

# ---- 阶段二：TIF 计算 ----
echo ""
echo "[4/4] 阶段二：TIF/LossDiff/IRM 计算..."
echo "（这可能需要 1-3 小时，取决于 GPU 速度）"
echo "----------------------------------------------"
python stage2_tif/compute_tif.py
echo "----------------------------------------------"
echo "阶段二-TIF计算完成!"

# ---- 阶段二：LossDiff/IRM ----
echo ""
echo "[4/4 续] 阶段二：LossDiff 和 IRM 计算..."
echo "----------------------------------------------"
python stage2_tif/compute_lossdiff_irm.py
echo "----------------------------------------------"
echo "阶段二-LossDiff/IRM完成!"

echo ""
echo "=============================================="
echo "阶段一和阶段二全部完成!"
echo "阶段三（Shapley）和阶段四（分析）的代码已就绪。"
echo "你可以手动运行:"
echo "  python stage3_shapley/compute_shapley.py --budget 200"
echo "  python stage4_analysis/compare.py"
echo "=============================================="
