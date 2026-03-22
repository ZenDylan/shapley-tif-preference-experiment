"""
LoRA 配置。
与 prompt.md 保持一致：rank=16, alpha=32。
"""

from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Pythia-410m 使用 GPT-NeoX 架构，attention 层名称为 query_key_value（QKV 合并投影）和 dense（输出投影）
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
