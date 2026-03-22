"""
阶段一：SFT 训练脚本。
在 ultrachat_200k 子集（约 10k 条）上做 1 epoch SFT 训练。
使用 LoRA (rank=16, alpha=32)，batch_size=1，gradient_checkpointing，bf16。
保存 checkpoint 到 checkpoints/sft_pythia410m/，作为后续 DPO 训练的起点。
"""

import os
import gc
import time
import random

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import (
    MODEL_NAME, MODEL_DTYPE,
    SFT_TRAINING_CONFIG, RANDOM_SEED,
    DATA_DIR,
)
from configs.lora_config import lora_config
from utils.memory import (
    print_memory_usage, format_params,
    get_vram_gb, get_ram_gb, reset_peak_vram,
    peak_vram_gb,
)
from utils.io import load_json


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    print("\n" + "#" * 60)
    print("# 阶段一：SFT 训练")
    print("#" * 60)
    print(f"  模型:        {MODEL_NAME}")
    print(f"  精度:        {MODEL_DTYPE}")
    print(f"  LoRA rank:   {lora_config.r}, alpha: {lora_config.lora_alpha}")
    print(f"  输出目录:    {SFT_TRAINING_CONFIG['output_dir']}")
    print_memory_usage("启动时")
    reset_peak_vram()

    # ---- 1. 加载 tokenizer 和模型 ----
    print("\n[1] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[2] 加载模型 (bf16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )
    print(f"    模型加载耗时: {time.time() - t0:.1f}s")
    print_memory_usage("模型加载后")

    # ---- 2. 注入 LoRA ----
    print("\n[3] 注入 LoRA...")
    model = get_peft_model(model, lora_config)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_numels = sum(p.numel() for p in trainable_params)

    print(f"  总参数量:   {format_params(total_params)} ({total_params:,})")
    print(f"  可训练参数: {format_params(trainable_numels)} ({trainable_numels:,})")
    print(f"  可训练比例: {trainable_numels / total_params * 100:.2f}%")
    lora_vram_mb = trainable_numels * 4 / 1024**2
    grad_mb = trainable_numels * 4 / 1024**2
    opt_mb = trainable_numels * 4 * 3 / 1024**2
    act_mb = 500
    total_est_mb = lora_vram_mb + grad_mb + opt_mb + act_mb
    print(f"  预估 VRAM 占用: ~{total_est_mb:.0f} MB "
          f"(权重{lora_vram_mb:.0f} + 梯度{grad_mb:.0f} + 优化器{opt_mb:.0f} + 激活值~{act_mb:.0f})")
    print_memory_usage("LoRA 注入后")

    # ---- 3. 加载 SFT 数据 ----
    print("\n[4] 加载 SFT 训练数据...")
    sft_path = os.path.join(DATA_DIR, "sft_train.json")
    if not os.path.exists(sft_path):
        raise FileNotFoundError(
            f"SFT 数据文件不存在: {sft_path}\n"
            "请先运行: python stage1_setup/prepare_data.py"
        )

    sft_records = load_json(sft_path)
    print(f"  已加载 {len(sft_records)} 条 SFT 训练数据")

    sft_dataset = Dataset.from_list(sft_records)

    def formatting_prompts_func(example):
        text = example["prompt"] + "\n" + example["response"]
        return {"text": text}

    sft_dataset = sft_dataset.map(
        formatting_prompts_func,
        remove_columns=sft_dataset.column_names,
    )
    print(f"  数据集格式化为 {len(sft_dataset)} 条")

    # ---- 4. 配置 SFTTrainer ----
    print("\n[5] 配置 SFTTrainer...")

    training_args = SFTConfig(
        output_dir=SFT_TRAINING_CONFIG["output_dir"],
        num_train_epochs=SFT_TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=SFT_TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=SFT_TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=SFT_TRAINING_CONFIG["learning_rate"],
        warmup_ratio=SFT_TRAINING_CONFIG["warmup_ratio"],
        lr_scheduler_type=SFT_TRAINING_CONFIG["lr_scheduler_type"],
        logging_steps=SFT_TRAINING_CONFIG["logging_steps"],
        save_steps=SFT_TRAINING_CONFIG["save_steps"],
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        optim="paged_adamw_32bit",
        seed=RANDOM_SEED,
        max_seq_length=SFT_TRAINING_CONFIG["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    print(f"  总训练步数: {trainer.get_train_dataloader().__len__() * training_args.num_train_epochs // training_args.gradient_accumulation_steps}")

    # ---- 5. 开始训练 ----
    print("\n[6] 开始 SFT 训练 (1 epoch)...")
    print_memory_usage("训练开始前")
    t_train_start = time.time()

    train_result = trainer.train()

    t_train_end = time.time()
    print(f"\n  训练完成! 总耗时: {(t_train_end - t_train_start) / 60:.1f} 分钟")
    print(f"  最终 train loss: {train_result.training_loss:.4f}")

    # ---- 6. 保存 checkpoint ----
    print("\n[7] 保存 SFT checkpoint...")
    trainer.save_model(SFT_TRAINING_CONFIG["output_dir"])
    trainer.save_state()
    print(f"  checkpoint 已保存至: {SFT_TRAINING_CONFIG['output_dir']}")

    # ---- 7. 清理 ----
    print("\n[8] 清理内存...")
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print_memory_usage("训练完成后")

    print("\n" + "#" * 60)
    print("# SFT 训练完成!")
    print(f"# checkpoint: {SFT_TRAINING_CONFIG['output_dir']}")
    print("#" * 60)


if __name__ == "__main__":
    main()
