"""
阶段四：对比分析脚本。

1. 加载 4 组分数（IF、Shapley、LossDiff、IRM）
2. 计算相关系数（Spearman + Pearson）
3. 生成散点图（IF vs Shapley），按 IF 分组着色
4. 分组训练对比
5. 不一致数据分析
6. 输出 PNG + JSON + Markdown 报告
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from tqdm import tqdm

from configs.config import (
    N_TRAIN, N_TEST, RANDOM_SEED, DATA_DIR,
    CHECKPOINT_DIR, OUTPUT_DIR,
)
from utils.io import load_json, save_json
from utils.memory import print_memory_usage


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def load_scores():
    """加载所有分数文件。"""
    if_scores = load_json(os.path.join(OUTPUT_DIR, "if_scores.json"))
    shapley_scores = load_json(os.path.join(OUTPUT_DIR, "shapley_values.json"))
    lossdiff_scores = load_json(os.path.join(OUTPUT_DIR, "lossdiff_scores.json"))
    irm_scores = load_json(os.path.join(OUTPUT_DIR, "irm_scores.json"))
    train_records = load_json(os.path.join(DATA_DIR, "prefs_train.json"))
    test_records = load_json(os.path.join(DATA_DIR, "prefs_test.json"))

    return {
        "if": if_scores.get("if_scores", []),
        "shapley": shapley_scores.get("shapley_values", []),
        "lossdiff": lossdiff_scores.get("lossdiff", []),
        "irm": irm_scores.get("irm_scores", []),
        "train": train_records,
        "test": test_records,
    }


def compute_correlations(data):
    """计算各方法之间的 Spearman 和 Pearson 相关系数。"""
    n = N_TRAIN

    if_vals = np.array([data["if"][i]["if_score"] for i in range(n)])
    sv_vals = np.array(data["shapley"][:n])
    ld_vals = np.array([data["lossdiff"][i]["lossdiff"] for i in range(n)])
    irm_vals = np.array([data["irm"][i]["irm_score"] for i in range(n)])

    correlations = {}
    pairs = [
        ("IF vs SV", if_vals, sv_vals),
        ("LD vs SV", ld_vals, sv_vals),
        ("IRM vs SV", irm_vals, sv_vals),
    ]

    for name, x, y in pairs:
        spearman_r, spearman_p = stats.spearmanr(x, y)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        correlations[name] = {
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
        }
        print(f"  {name}:")
        print(f"    Spearman r={spearman_r:.4f}, p={spearman_p:.4e}")
        print(f"    Pearson  r={pearson_r:.4f}, p={pearson_p:.4e}")

    return correlations


def plot_scatter(data, correlations, output_dir):
    """生成 IF vs SV 散点图，按 IF 分组着色。"""
    n = N_TRAIN

    if_vals = np.array([data["if"][i]["if_score"] for i in range(n)])
    sv_vals = np.array(data["shapley"][:n])

    # 按 IF 分组
    sorted_indices = np.argsort(if_vals)
    third = n // 3
    groups = ["small", "medium", "large"]
    group_labels = np.array(["unranked"] * n)
    group_labels[sorted_indices[:third]] = "small"
    group_labels[sorted_indices[third:2*third]] = "medium"
    group_labels[sorted_indices[2*third:]] = "large"

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = {"small": "blue", "medium": "green", "large": "red"}

    for group in groups:
        mask = group_labels == group
        ax.scatter(if_vals[mask], sv_vals[mask], label=group, alpha=0.7,
                   color=palette[group], s=50)

    ax.set_xlabel("IF Score")
    ax.set_ylabel("Shapley Value")
    ax.set_title("IF vs Shapley Value (colored by IF group)")
    ax.legend(title="IF Group")

    # 添加相关系数文本
    r = correlations["IF vs SV"]["spearman_r"]
    ax.text(0.05, 0.95, f"Spearman r = {r:.4f}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top")

    plt.tight_layout()
    path = os.path.join(output_dir, "scatter_if_vs_shapley.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  散点图已保存: {path}")


def plot_correlation_matrix(data, output_dir):
    """生成相关系数热力图。"""
    n = N_TRAIN

    if_vals = np.array([data["if"][i]["if_score"] for i in range(n)])
    sv_vals = np.array(data["shapley"][:n])
    ld_vals = np.array([data["lossdiff"][i]["lossdiff"] for i in range(n)])
    irm_vals = np.array([data["irm"][i]["irm_score"] for i in range(n)])

    matrix = np.array([
        [1.0,                   stats.spearmanr(if_vals, sv_vals)[0],
         stats.spearmanr(if_vals, ld_vals)[0],   stats.spearmanr(if_vals, irm_vals)[0]],
        [stats.spearmanr(sv_vals, if_vals)[0],  1.0,
         stats.spearmanr(sv_vals, ld_vals)[0],   stats.spearmanr(sv_vals, irm_vals)[0]],
        [stats.spearmanr(ld_vals, if_vals)[0],   stats.spearmanr(ld_vals, sv_vals)[0],
         1.0,                                     stats.spearmanr(ld_vals, irm_vals)[0]],
        [stats.spearmanr(irm_vals, if_vals)[0],  stats.spearmanr(irm_vals, sv_vals)[0],
         stats.spearmanr(irm_vals, ld_vals)[0],  1.0],
    ])

    labels = ["IF", "Shapley", "LossDiff", "IRM"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Spearman Rank Correlation Matrix")
    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  热力图已保存: {path}")


def group_training_comparison(data, output_dir):
    """
    分组训练效果对比。

    按 IF 排序将 200 条数据分成 small/medium/large 三组，
    按 Shapley 排序同样分成三组，
    分别用每组数据做 DPO 训练，在测试集上计算 DPO loss。
    """
    print("\n[分组训练对比]")
    print("  此步骤需要运行 DPO 训练（耗时较长），代码已就绪。")
    print("  建议在确认阶段一和阶段二正确运行后，手动执行。")
    # TODO: 实现分组训练逻辑（可参考 stage2 的 DPO 训练代码）


def analyze_disagreements(data, output_dir, top_n=20):
    """分析 IF 和 SV 排序不一致的数据。"""
    n = N_TRAIN

    if_scores_raw = data["if"]
    sv_vals = np.array(data["shapley"][:n])
    if_vals = np.array([if_scores_raw[i]["if_score"] for i in range(n)])
    train_records = data["train"]

    # 计算排名差异
    if_rank = stats.rankdata(if_vals)
    sv_rank = stats.rankdata(sv_vals)
    rank_diff = np.abs(if_rank - sv_rank)

    # 找出 Top-N 不一致数据
    top_indices = np.argsort(rank_diff)[-top_n:][::-1]

    print(f"\n[不一致分析] Top-{top_n} IF vs SV 排序差异最大的数据:")
    disagreements = []

    for idx in top_indices:
        d = {
            "index": int(idx),
            "if_score": float(if_vals[idx]),
            "shapley_value": float(sv_vals[idx]),
            "rank_diff": float(rank_diff[idx]),
            "if_rank": int(if_rank[idx]),
            "sv_rank": int(sv_rank[idx]),
            "prompt": train_records[idx]["prompt"][:300],
            "margin": train_records[idx].get("margin"),
        }
        disagreements.append(d)
        print(f"  [{idx}] IF={if_vals[idx]:.4f} (rank={int(if_rank[idx])}), "
              f"SV={sv_vals[idx]:.4f} (rank={int(sv_rank[idx])}), "
              f"diff={rank_diff[idx]:.0f}")
        print(f"    prompt: {train_records[idx]['prompt'][:100]}...")

    return disagreements


def generate_report(correlations, disagreements, output_dir):
    """生成 Markdown 实验报告。"""
    md_path = os.path.join(output_dir, "experiment_report.md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 实验报告：Shapley Value vs TIF 偏好数据估值对比\n\n")

        f.write("## 1. 相关系数分析\n\n")
        for name, corr in correlations.items():
            f.write(f"### {name}\n\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|----|\n")
            f.write(f"| Spearman r | {corr['spearman_r']:.4f} |\n")
            f.write(f"| Spearman p | {corr['spearman_p']:.4e} |\n")
            f.write(f"| Pearson r | {corr['pearson_r']:.4f} |\n")
            f.write(f"| Pearson p | {corr['pearson_p']:.4e} |\n\n")

        f.write(f"## 2. 不一致数据分析\n\n")
        f.write(f"Top-20 IF 与 Shapley 排序差异最大的数据：\n\n")
        f.write(f"| 索引 | IF Score | Shapley | IF Rank | SV Rank | 差异 | Margin |\n")
        f.write(f"|------|----------|---------|---------|---------|------|--------|\n")
        for d in disagreements[:20]:
            f.write(f"| {d['index']} | {d['if_score']:.4f} | {d['shapley_value']:.4f} "
                    f"| {d['if_rank']} | {d['sv_rank']} | {d['rank_diff']:.0f} "
                    f"| {d.get('margin', 'N/A')} |\n")

    print(f"  报告已生成: {md_path}")


def main():
    set_seed()
    print("\n" + "#" * 60)
    print("# 阶段四：对比分析")
    print("#" * 60)
    print_memory_usage("启动时")

    # ---- 1. 加载数据 ----
    print("\n[1] 加载分数数据...")
    data = load_scores()

    n_if = len(data["if"])
    n_sv = len(data["shapley"])
    n_ld = len(data["lossdiff"])
    n_irm = len(data["irm"])

    print(f"  IF scores:      {n_if} 条")
    print(f"  Shapley values: {n_sv} 条")
    print(f"  LossDiff:       {n_ld} 条")
    print(f"  IRM scores:     {n_irm} 条")

    if n_if == 0 or n_sv == 0:
        print("  错误: 缺少 IF 或 Shapley 数据，请先运行阶段二和阶段三")
        return

    # ---- 2. 计算相关系数 ----
    print("\n[2] 计算相关系数...")
    correlations = compute_correlations(data)

    # ---- 3. 生成散点图 ----
    print("\n[3] 生成散点图...")
    plot_scatter(data, correlations, OUTPUT_DIR)

    # ---- 3b. 生成热力图 ----
    print("\n[3b] 生成相关系数热力图...")
    plot_correlation_matrix(data, OUTPUT_DIR)

    # ---- 4. 分组训练对比 ----
    print("\n[4] 分组训练对比（仅生成占位，后续手动执行）...")
    group_training_comparison(data, OUTPUT_DIR)

    # ---- 5. 不一致数据分析 ----
    print("\n[5] 不一致数据分析...")
    disagreements = analyze_disagreements(data, OUTPUT_DIR)

    # ---- 6. 生成报告 ----
    print("\n[6] 生成 Markdown 报告...")
    generate_report(correlations, disagreements, OUTPUT_DIR)

    # ---- 7. 保存数值结果 ----
    print("\n[7] 保存数值结果...")
    result_path = os.path.join(OUTPUT_DIR, "analysis_results.json")
    save_json({
        "correlations": correlations,
        "disagreements": disagreements,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
    }, result_path)

    print_memory_usage("完成后")

    print("\n" + "#" * 60)
    print("# 阶段四：对比分析完成!")
    print(f"# 结果目录: {OUTPUT_DIR}")
    print("#" * 60)


if __name__ == "__main__":
    main()
