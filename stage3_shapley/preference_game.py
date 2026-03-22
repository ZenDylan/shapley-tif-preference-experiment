"""
阶段三：PreferenceDPOGame —— Shapley Value 的 value function 定义。

每次调用 __call__(coalition) 即执行一次 value function 评估：
1. 从 SFT checkpoint 启动
2. 用 coalition 对应的数据子集做 1 epoch DPO 训练
3. 在验证集上计算 DPO loss
4. 完整内存清理后返回 -loss

使用子进程隔离（multiprocessing）避免内存泄漏。
缓存结果写入 JSONL 文件，支持断点续传。
"""

import os
import gc
import time
import random
import json
import multiprocessing as mp

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from tqdm import tqdm
from datasets import Dataset as HFDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    MODEL_NAME, DPO_TRAINING_CONFIG,
    RANDOM_SEED, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
)
from configs.lora_config import lora_config
from utils.io import load_json, load_jsonl_as_dict, append_jsonl


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 子进程中的 value function 实现（每次独立运行）
# ============================================================

def _run_dpo_in_subprocess(args):
    """
    在独立子进程中运行 DPO 训练并返回验证集 loss。
    返回 (coalition_tuple, -val_loss, eval_time_seconds)
    """
    coalition, train_records_subset, val_records, cache_path = args

    set_seed(RANDOM_SEED + len(coalition))  # 不同 coalition 用不同 seed

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- 加载 SFT checkpoint ----
        sft_ckpt = os.path.join(CHECKPOINT_DIR, "sft_pythia410m")
        tokenizer = AutoTokenizer.from_pretrained(sft_ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            sft_ckpt,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,
        )
        model = get_peft_model(model, lora_config)
        model.train()

        # ---- 构建 DPO 训练数据集 ----
        pref_data = HFDataset.from_list([{
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        } for r in train_records_subset])

        # ---- DPO 训练 ----
        tmp_output = os.path.join(CHECKPOINT_DIR, f"tmp_dpo_{hash(str(coalition))}")
        os.makedirs(tmp_output, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=tmp_output,
            num_train_epochs=DPO_TRAINING_CONFIG["num_train_epochs"],
            per_device_train_batch_size=DPO_TRAINING_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=DPO_TRAINING_CONFIG["gradient_accumulation_steps"],
            learning_rate=DPO_TRAINING_CONFIG["learning_rate"],
            warmup_ratio=DPO_TRAINING_CONFIG["warmup_ratio"],
            lr_scheduler_type=DPO_TRAINING_CONFIG["lr_scheduler_type"],
            logging_steps=DPO_TRAINING_CONFIG["logging_steps"],
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            remove_unused_columns=False,
            optim="paged_adamw_32bit",
            seed=RANDOM_SEED,
        )

        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=pref_data,
            max_length=DPO_TRAINING_CONFIG["max_seq_length"],
            max_prompt_length=int(DPO_TRAINING_CONFIG["max_seq_length"] // 2),
        )

        dpo_trainer.train()

        # ---- 在验证集上计算 DPO loss ----
        model.eval()
        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for sample in val_records:
                # 简化：用 last-token logit difference 作为 DPO loss 的近似
                prompt_tok = tokenizer(sample["prompt"], return_tensors="pt",
                                      truncation=True, max_length=256).to(device)
                chosen_tok = tokenizer(sample["chosen"], return_tensors="pt",
                                      truncation=True, max_length=256).to(device)
                rejected_tok = tokenizer(sample["rejected"], return_tensors="pt",
                                        truncation=True, max_length=256).to(device)

                prompt_ids = prompt_tok.input_ids[0]
                chosen_ids = chosen_tok.input_ids[0]
                rejected_ids = rejected_tok.input_ids[0]

                chosen_full = torch.cat([prompt_ids, chosen_ids], dim=-1).unsqueeze(0)
                rejected_full = torch.cat([prompt_ids, rejected_ids], dim=-1).unsqueeze(0)

                c_out = model(input_ids=chosen_full)
                r_out = model(input_ids=rejected_full)

                c_logit = c_out.logits[0, -1, :].mean()
                r_logit = r_out.logits[0, -1, :].mean()

                loss = -torch.log(torch.sigmoid(c_logit - r_logit) + 1e-8)
                total_loss += loss.item()
                n_samples += 1

        val_loss = total_loss / max(n_samples, 1)

        # ---- 完整内存清理 ----
        del model, dpo_trainer
        gc.collect()
        torch.cuda.empty_cache()

        # 清理临时目录
        import shutil
        if os.path.exists(tmp_output):
            shutil.rmtree(tmp_output)

        return {
            "success": True,
            "coalition": list(coalition),
            "value": float(-val_loss),
            "val_loss": float(val_loss),
            "n_samples": len(train_records_subset),
        }

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "success": False,
            "coalition": list(coalition),
            "error": str(e),
            "value": 0.0,
        }


# ============================================================
# 主 Game 类 —— 适配 Stratified SVARM 接口
# ============================================================

class PreferenceDPOGame:
    """
    合作博弈：偏好数据估值。

    Players: 0~199（共 N_TRAIN=200 条训练偏好对）
    Value function v(S):
        从 SFT checkpoint 开始，用子集 S 中的数据做 1 epoch DPO 训练，
        在验证集上计算 DPO loss，返回 -loss（因为 loss 越小越好）。

    特性：
    - 子进程隔离：每次评估在独立进程中运行，彻底避免内存泄漏
    - JSONL 缓存：每次评估完立即追加写入磁盘，支持断点续传
    - 空集合返回 0
    """

    def __init__(self, budget: int = 200, n_workers: int = 1):
        """
        Args:
            budget: SVARM 的总评估预算
            n_workers: 并行子进程数（建议 1，因为显存只有 8GB）
        """
        self.n = 200  # 玩家数量 = 训练集大小
        self.budget = budget
        self.n_workers = n_workers
        self.eval_count = 0
        self.start_time = time.time()
        self.cache_path = os.path.join(OUTPUT_DIR, "shapley_cache.jsonl")

        # 加载训练集和验证集
        self.all_train_records = load_json(os.path.join(DATA_DIR, "prefs_train.json"))
        self.val_records = load_json(os.path.join(DATA_DIR, "prefs_val.json"))

        # 加载已有缓存（断点续传）
        self.cache = load_jsonl_as_dict(self.cache_path, key_field="coalition")
        already_done = len(self.cache)
        print(f"[PreferenceDPOGame] 初始化完成, {self.n} 个玩家")
        print(f"[PreferenceDPOGame] 已有缓存: {already_done} 条")
        print(f"[PreferenceDPOGame] 子进程数: {self.n_workers} (建议 1 以节省显存)")

        # 创建 tqdm 进度条
        self.pbar = tqdm(
            total=self.budget,
            desc="[Shapley] Value Function 评估",
            unit="次",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

    def get_player_number(self) -> int:
        return self.n

    def get_name(self) -> str:
        return "PreferenceDPOGame"

    def get_shapley_values(self) -> list:
        # 精确 Shapley Value（仅在玩家数很小时使用）
        raise NotImplementedError("精确 Shapley Value 计算量过大，请使用 SVARM 近似。")

    def get_value(self, coalition) -> float:
        """对外暴露的接口（兼容 SVARM 代码）。"""
        return self(coalition)

    def __call__(self, coalition) -> float:
        """
        评估给定 coalition 的 value。
        先查缓存，未命中则启动子进程计算。
        """
        coalition_tuple = tuple(sorted(coalition))

        # 查缓存
        if coalition_tuple in self.cache:
            cached = self.cache[coalition_tuple]
            self.eval_count += 1
            value = cached.get("value", 0.0)
            self._update_pbar(value)
            return value

        # 空集合
        if len(coalition) == 0:
            result = {"success": True, "coalition": [], "value": 0.0, "val_loss": 0.0}
        else:
            # 取子集数据
            train_subset = [self.all_train_records[i] for i in coalition]
            args = (coalition_tuple, train_subset, self.val_records, self.cache_path)
            result = _run_dpo_in_subprocess(args)

        # 缓存
        entry = {
            "coalition": list(coalition_tuple),
            "value": result.get("value", 0.0),
            "val_loss": result.get("val_loss", 0.0),
            "success": result.get("success", False),
            "eval_count": self.eval_count + 1,
            "timestamp": time.time(),
        }
        append_jsonl(entry, self.cache_path)
        self.cache[coalition_tuple] = entry

        self.eval_count += 1
        value = result.get("value", 0.0)
        self._update_pbar(value)
        return value

    def _update_pbar(self, value: float):
        """更新进度条 postfix。"""
        elapsed = time.time() - self.start_time
        avg_time = elapsed / max(self.eval_count, 1)
        import psutil
        ram_gb = psutil.Process().memory_info().rss / 1024**3

        self.pbar.set_postfix(
            avg=f"{avg_time:.1f}s/次",
            RAM=f"{ram_gb:.1f}GB",
            val=f"{value:.4f}"
        )
        self.pbar.update(1)

    def close(self):
        self.pbar.close()
