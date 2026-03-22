"""
阶段一：数据准备脚本。
- 下载 HuggingFaceH4/ultrachat_200k（采样 10k 条用于 SFT）
- 下载 HuggingFaceH4/ultrafeedback_binarized（偏好数据，分层采样 200/100/100）
- 按 GPT-4 score margin 做分层采样（确保高质量和低质量偏好对都有覆盖）
- 保存为本地 JSON 文件
- 打印数据集统计信息
"""

import random
import numpy as np
import psutil
from datasets import load_dataset
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    SFT_DATASET_NAME, PREF_DATASET_NAME,
    N_SFT_SAMPLES, N_TRAIN, N_VAL, N_TEST,
    STRATA_COUNT, RANDOM_SEED, DATA_DIR,
)
from utils.io import save_json
from utils.memory import print_memory_usage


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)


def extract_text_from_messages(messages: list) -> str:
    """
    将 messages 列表拼接成纯文本字符串（prompt + assistant response）。
    用于 SFT 训练时的文本拼接。
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|> {content}")
    return "\n".join(parts)


def prepare_sft_data() -> list:
    """下载并采样 SFT 数据。"""
    print("\n" + "=" * 60)
    print("[准备] SFT 数据: HuggingFaceH4/ultrachat_200k")
    print("=" * 60)

    print(f"  下载数据集（这可能需要几分钟）...")
    ds = load_dataset(SFT_DATASET_NAME, split="train_sft")
    print(f"  原始数据量: {len(ds)} 条")

    total = min(N_SFT_SAMPLES, len(ds))
    sampled = ds.select(range(total))

    print(f"  采样后: {len(sampled)} 条")
    print_memory_usage("SFT 数据加载后")

    records = []
    for item in sampled:
        messages = item.get("messages", [])
        prompt_parts = []
        response_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                response_parts.append(content)

        prompt = "\n".join(prompt_parts)
        response = "\n".join(response_parts)

        records.append({
            "prompt": prompt,
            "response": response,
            "messages": messages,
        })

    return records


def prepare_pref_data() -> tuple[list, list, list]:
    """下载偏好数据并做分层采样。"""
    print("\n" + "=" * 60)
    print("[准备] 偏好数据: HuggingFaceH4/ultrafeedback_binarized")
    print("=" * 60)

    print(f"  下载数据集（这可能需要几分钟）...")
    ds = load_dataset(PREF_DATASET_NAME, split="train_prefs")
    print(f"  原始数据量: {len(ds)} 条")
    print_memory_usage("偏好数据加载后")

    records = []
    for item in tqdm(ds, desc="[Prepare] 处理偏好数据", unit="条"):
        prompt_raw = item.get("prompt", "")
        if isinstance(prompt_raw, list):
            prompt = "\n".join(m.get("content", "") for m in prompt_raw if isinstance(m, dict))
        else:
            prompt = str(prompt_raw)

        chosen_raw = item.get("chosen", [])
        if isinstance(chosen_raw, list):
            chosen = "\n".join(m.get("content", "") for m in chosen_raw if isinstance(m, dict))
        else:
            chosen = str(chosen_raw)

        rejected_raw = item.get("rejected", [])
        if isinstance(rejected_raw, list):
            rejected = "\n".join(m.get("content", "") for m in rejected_raw if isinstance(m, dict))
        else:
            rejected = str(rejected_raw)

        score_chosen = float(item.get("score_chosen", 0))
        score_rejected = float(item.get("score_rejected", 0))
        margin = score_chosen - score_rejected

        records.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "score_chosen": score_chosen,
            "score_rejected": score_rejected,
            "margin": margin,
        })

    print(f"  处理完毕: {len(records)} 条")

    margins = [r["margin"] for r in records]
    print(f"\n  [统计] Margin 分布:")
    print(f"    最小值={min(margins):.2f}, 最大值={max(margins):.2f}, "
          f"均值={np.mean(margins):.2f}, 标准差={np.std(margins):.2f}")

    percentiles = [25, 50, 75, 90, 95]
    print(f"    分位数: " + ", ".join(f"p{p}={np.percentile(margins, p):.2f}" for p in percentiles))

    # ---- 分层采样 ----
    print(f"\n  [分层采样] 将 margin 分为 {STRATA_COUNT} 档，按比例分配 train:val:test = {N_TRAIN}:{N_VAL}:{N_TEST}...")

    sorted_records = sorted(records, key=lambda r: r["margin"])

    # 将数据分成 STRATA_COUNT 档
    strata_size = len(sorted_records) // STRATA_COUNT
    strata = []
    for i in range(STRATA_COUNT):
        start = i * strata_size
        if i == STRATA_COUNT - 1:
            end = len(sorted_records)
        else:
            end = (i + 1) * strata_size
        strata.append(sorted_records[start:end])

    # 初始化三个集合
    train_set: list = []
    val_set: list = []
    test_set: list = []

    # 剩余配额（整数减法，无浮点误差）
    remaining_train = N_TRAIN
    remaining_val   = N_VAL
    remaining_test  = N_TEST
    total_remaining = N_TRAIN + N_VAL + N_TEST  # 400

    for stratum in strata:
        random.shuffle(stratum)
        stratum_len = len(stratum)

        # 按档内人数比例分配配额（整数，避免浮点误差）
        quota_train = round(stratum_len * N_TRAIN / total_remaining)
        quota_val   = round(stratum_len * N_VAL  / total_remaining)
        quota_test  = stratum_len - quota_train - quota_val  # 本档剩余全给 test

        # 防止超额：若 train 配额超过剩余量，用尽剩余量并把多出的退回 val
        if quota_train > remaining_train:
            excess = quota_train - remaining_train
            quota_train = remaining_train
            quota_val = min(stratum_len - quota_train, quota_val + excess)

        # 防止超额：val 配额
        if quota_val > remaining_val:
            excess = quota_val - remaining_val
            quota_val = remaining_val
            quota_test = stratum_len - quota_train - quota_val

        # 防止超额：test 配额
        quota_test = min(quota_test, remaining_test)
        # 若 test 不够，从 val 补（极端情况很少触发）
        if quota_train + quota_val + quota_test > stratum_len:
            quota_test = stratum_len - quota_train - quota_val

        train_set.extend(stratum[:quota_train])
        val_set.extend(stratum[quota_train:quota_train + quota_val])
        test_set.extend(stratum[quota_train + quota_val:quota_train + quota_val + quota_test])

        remaining_train -= quota_train
        remaining_val   -= quota_val
        remaining_test  -= quota_test
        total_remaining -= (quota_train + quota_val + quota_test)

    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    print(f"\n  [采样结果]")
    print(f"    训练集 D_train: {len(train_set)} 条")
    print(f"    验证集 D_val:   {len(val_set)} 条")
    print(f"    测试集 D_test:  {len(test_set)} 条")

    return train_set, val_set, test_set


def print_pref_stats(dataset: list, name: str):
    """打印偏好数据集的统计信息。"""
    if not dataset:
        return
    margins = [r["margin"] for r in dataset]
    prompt_lens = [len(r["prompt"]) for r in dataset]
    chosen_lens = [len(r["chosen"]) for r in dataset]
    rejected_lens = [len(r["rejected"]) for r in dataset]

    print(f"\n  [{name}] 统计信息:")
    print(f"    Margin:      min={min(margins):.2f}, max={max(margins):.2f}, "
          f"mean={np.mean(margins):.2f}, std={np.std(margins):.2f}")
    print(f"    Prompt 长度: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"median={np.median(prompt_lens):.0f}")
    print(f"    Chosen 长度: min={min(chosen_lens)}, max={max(chosen_lens)}, "
          f"median={np.median(chosen_lens):.0f}")
    print(f"    Rejected 长度: min={min(rejected_lens)}, max={max(rejected_lens)}, "
          f"median={np.median(rejected_lens):.0f}")


def main():
    set_seed()
    print("\n" + "#" * 60)
    print("# 阶段一：数据准备")
    print("#" * 60)
    print(f"  SFT 采样量:   {N_SFT_SAMPLES}")
    print(f"  偏好数据:     训练 {N_TRAIN} / 验证 {N_VAL} / 测试 {N_TEST}")
    print(f"  分层数:       {STRATA_COUNT}")
    print(f"  数据目录:     {DATA_DIR}")
    print_memory_usage("启动时")

    # 1. SFT 数据
    sft_records = prepare_sft_data()
    sft_path = os.path.join(DATA_DIR, "sft_train.json")
    save_json(sft_records, sft_path)

    # 2. 偏好数据
    train_set, val_set, test_set = prepare_pref_data()

    prefs_train_path = os.path.join(DATA_DIR, "prefs_train.json")
    prefs_val_path   = os.path.join(DATA_DIR, "prefs_val.json")
    prefs_test_path  = os.path.join(DATA_DIR, "prefs_test.json")

    save_json(train_set, prefs_train_path)
    save_json(val_set,   prefs_val_path)
    save_json(test_set,  prefs_test_path)

    # 打印统计
    print_pref_stats(train_set, "D_train")
    print_pref_stats(val_set,   "D_val")
    print_pref_stats(test_set,  "D_test")

    print("\n" + "#" * 60)
    print("# 数据准备完成!")
    print(f"#  文件已保存至 {DATA_DIR}")
    print("#" * 60)
    print_memory_usage("数据准备完成")


if __name__ == "__main__":
    main()
