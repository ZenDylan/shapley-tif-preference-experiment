"""
阶段二：LossDiff 和 IRM 指标计算脚本。

LossDiff(d) = ℓ(θ; d) - ℓ(θ_val; d)
- 需要额外在验证集上做 1 epoch DPO 训练，得到 θ_val
- 然后计算每条训练数据在 θ 和 θ_val 上的 DPO loss 之差
- 只需要前向传播，不需要梯度

IRM(d) = β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x))
- 只需要前向传播，计算 chosen 和 rejected 的 log probability
"""

import os
import gc
import time
import random

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
from tqdm import tqdm
from datasets import Dataset as HFDataset
from transformers import TrainingArguments

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    MODEL_NAME, DPO_TRAINING_CONFIG,
    RANDOM_SEED, DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
)
from configs.lora_config import lora_config
from utils.io import load_json, save_json
from utils.memory import (
    print_memory_usage, get_ram_gb, get_vram_gb,
    reset_peak_vram, peak_vram_gb,
)


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_log_prob(model, tokenizer, text, device, max_len=512):
    """
    计算给定文本在模型下的 log probability。
    返回每个 token 的 log prob 之和。
    """
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                   max_length=max_len).input_ids[0].to(device)

    if ids.size(0) == 0:
        return 0.0

    with torch.no_grad():
        outputs = model(input_ids=ids.unsqueeze(0))
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # 交叉熵：shift one step
    log_probs = torch.log_softmax(logits, dim=-1)
    # 目标 token = input_ids[1:]
    target = ids[1:]
    # 对应位置的 log prob
    token_log_probs = log_probs[:-1, :].gather(dim=1, index=target.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum().item()


def compute_dpo_loss(model, tokenizer, prompt, chosen, rejected, device, max_len=512):
    """
    计算单条偏好数据的 DPO-style loss（不需要梯度）。
    DPO loss = -log σ(log_ratio) 近似为:
        -log σ(β * (log_prob_chosen - log_prob_rejected))
    这里 β=1。
    """
    with torch.no_grad():
        lp_chosen = compute_log_prob(model, tokenizer, prompt + "\n" + chosen,
                                    device, max_len)
        lp_rejected = compute_log_prob(model, tokenizer, prompt + "\n" + rejected,
                                      device, max_len)
        diff = lp_chosen - lp_rejected
        loss = -np.log(np.exp(diff) / (1 + np.exp(diff)) + 1e-8)
    return loss, lp_chosen, lp_rejected


def main():
    set_seed()
    print("\n" + "#" * 60)
    print("# 阶段二：LossDiff 和 IRM 指标计算")
    print("#" * 60)
    print_memory_usage("启动时")
    reset_peak_vram()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")

    # ---- 1. 加载数据 ----
    print("\n[1] 加载数据...")
    train_records = load_json(os.path.join(DATA_DIR, "prefs_train.json"))
    val_records   = load_json(os.path.join(DATA_DIR, "prefs_val.json"))
    print(f"  训练集: {len(train_records)} 条")
    print(f"  验证集: {len(val_records)} 条")

    # ---- 2. 加载 θ（SFT checkpoint 上的 DPO checkpoint）----
    print("\n[2] 加载 θ (DPO checkpoint)...")
    theta_ckpt = os.path.join(CHECKPOINT_DIR, "dpo_theta")
    if not os.path.exists(theta_ckpt):
        raise FileNotFoundError(
            f"θ checkpoint 不存在: {theta_ckpt}\n"
            "请先运行: python stage2_tif/compute_tif.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(theta_ckpt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    theta_model = AutoModelForCausalLM.from_pretrained(
        theta_ckpt, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True
    )
    theta_model = get_peft_model(theta_model, lora_config)
    theta_model.eval()
    print_memory_usage("θ 加载后")

    # ---- 3. 训练 θ_val（验证集上 1 epoch DPO）----
    print("\n[3] 训练 θ_val (验证集 1 epoch DPO)...")

    val_pref = HFDataset.from_list([{
        "prompt": r["prompt"],
        "chosen": r["chosen"],
        "rejected": r["rejected"],
    } for r in val_records])

    val_output_dir = os.path.join(CHECKPOINT_DIR, "dpo_theta_val")
    val_training_args = TrainingArguments(
        output_dir=val_output_dir,
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

    theta_val_model = AutoModelForCausalLM.from_pretrained(
        theta_ckpt, torch_dtype=torch.bfloat16, device_map="auto", use_cache=False
    )
    theta_val_model = get_peft_model(theta_val_model, lora_config)
    theta_val_model.train()

    val_trainer = DPOTrainer(
        model=theta_val_model,
        ref_model=None,
        args=val_training_args,
        processing_class=tokenizer,
        train_dataset=val_pref,
        max_length=DPO_TRAINING_CONFIG["max_seq_length"],
        max_prompt_length=int(DPO_TRAINING_CONFIG["max_seq_length"] // 2),
    )

    t_val_start = time.time()
    val_trainer.train()
    t_val_end = time.time()
    print(f"  θ_val 训练完成! 耗时: {(t_val_end - t_val_start) / 60:.1f} 分钟")
    theta_val_model.eval()
    print_memory_usage("θ_val 训练后")

    # ---- 4. 加载 reference model (π_ref = SFT checkpoint) ----
    print("\n[4] 加载 reference model (π_ref = SFT)...")
    sft_ckpt = os.path.join(CHECKPOINT_DIR, "sft_pythia410m")
    ref_model = AutoModelForCausalLM.from_pretrained(
        sft_ckpt, torch_dtype=torch.bfloat16, device_map="auto", use_cache=True
    )
    ref_model.eval()
    print_memory_usage("π_ref 加载后")

    # ---- 5. 计算 LossDiff ----
    print("\n[5] 计算 LossDiff...")
    print_memory_usage("LossDiff 计算前")

    lossdiff_results = []
    t_ld_start = time.time()

    with tqdm(total=len(train_records), desc="[LossDiff] 训练集指标", unit="条",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for idx, sample in enumerate(train_records):
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            # θ 上的 DPO loss
            loss_theta, _, _ = compute_dpo_loss(
                theta_model, tokenizer, prompt, chosen, rejected, device
            )

            # θ_val 上的 DPO loss
            loss_theta_val, _, _ = compute_dpo_loss(
                theta_val_model, tokenizer, prompt, chosen, rejected, device
            )

            lossdiff = loss_theta - loss_theta_val

            lossdiff_results.append({
                "index": idx,
                "loss_theta": float(loss_theta),
                "loss_theta_val": float(loss_theta_val),
                "lossdiff": float(lossdiff),
            })

            elapsed = time.time() - t_ld_start
            avg_time = elapsed / (idx + 1)
            ram_gb = get_ram_gb()
            vram_gb = get_vram_gb()
            pbar.set_postfix(LD=f"{lossdiff:.4f}", RAM=f"{ram_gb:.1f}GB",
                             VRAM=f"{vram_gb:.1f}GB", avg=f"{avg_time:.1f}s/条")
            pbar.update(1)

    print(f"  LossDiff 计算完成! 耗时: {(time.time() - t_ld_start) / 60:.1f} 分钟")

    # ---- 6. 计算 IRM ----
    print("\n[6] 计算 IRM (β=0.5)...")
    BETA = 0.5
   irm_results = []
    t_irm_start = time.time()

    with tqdm(total=len(train_records), desc="[IRM] 训练集IRM分数", unit="条",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        for idx, sample in enumerate(train_records):
            prompt = sample["prompt"]
            chosen = sample["chosen"]
            rejected = sample["rejected"]

            full_prompt_chosen = prompt + "\n" + chosen
            full_prompt_rejected = prompt + "\n" + rejected

            with torch.no_grad():
                # θ
                lp_c_theta = compute_log_prob(theta_model, tokenizer,
                                              full_prompt_chosen, device)
                lp_r_theta = compute_log_prob(theta_model, tokenizer,
                                              full_prompt_rejected, device)

                # π_ref
                lp_c_ref = compute_log_prob(ref_model, tokenizer,
                                             full_prompt_chosen, device)
                lp_r_ref = compute_log_prob(ref_model, tokenizer,
                                            full_prompt_rejected, device)

            # IRM: β * log(π_θ(y_w|x) / π_ref(y_w|x)) - β * log(π_θ(y_l|x) / π_ref(y_l|x))
            irm_score = BETA * (lp_c_theta - lp_c_ref) - BETA * (lp_r_theta - lp_r_ref)

            irm_results.append({
                "index": idx,
                "irm_score": float(irm_score),
                "beta": BETA,
            })

            elapsed = time.time() - t_irm_start
            avg_time = elapsed / (idx + 1)
            ram_gb = get_ram_gb()
            vram_gb = get_vram_gb()
            pbar.set_postfix(IRM=f"{irm_score:.4f}", RAM=f"{ram_gb:.1f}GB",
                             VRAM=f"{vram_gb:.1f}GB", avg=f"{avg_time:.1f}s/条")
            pbar.update(1)

    print(f"  IRM 计算完成! 耗时: {(time.time() - t_irm_start) / 60:.1f} 分钟")

    # ---- 7. 保存结果 ----
    print("\n[7] 保存结果...")

    lossdiff_path = os.path.join(OUTPUT_DIR, "lossdiff_scores.json")
    save_json({"lossdiff": lossdiff_results, "n_train": len(train_records)},
              lossdiff_path)

    irm_path = os.path.join(OUTPUT_DIR, "irm_scores.json")
    save_json({"irm_scores": irm_results, "n_train": len(train_records), "beta": BETA},
              irm_path)

    # 打印统计
    lds = [r["lossdiff"] for r in lossdiff_results]
    irms = [r["irm_score"] for r in irm_results]
    print(f"\n  LossDiff 统计:")
    print(f"    min={min(lds):.4f}, max={max(lds):.4f}, mean={np.mean(lds):.4f}, std={np.std(lds):.4f}")
    print(f"\n  IRM 统计:")
    print(f"    min={min(irms):.4f}, max={max(irms):.4f}, mean={np.mean(irms):.4f}, std={np.std(irms):.4f}")

    # ---- 8. 清理 ----
    print("\n[8] 清理内存...")
    del theta_model, theta_val_model, ref_model, val_trainer
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_usage("完成后")

    print("\n" + "#" * 60)
    print("# LossDiff 和 IRM 计算完成!")
    print(f"# LossDiff: {lossdiff_path}")
    print(f"# IRM:      {irm_path}")
    print("#" * 60)


if __name__ == "__main__":
    main()
