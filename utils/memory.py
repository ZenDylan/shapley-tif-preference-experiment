"""
内存 / CPU / GPU 监控工具。
提供格式化输出和 tqdm postfix 字符串，供各阶段脚本复用。
"""

import psutil
import torch
from configs.config import RAM_WARN_GB


def get_ram_gb() -> float:
    """返回当前进程 RSS 内存（GB）。"""
    return psutil.Process().memory_info().rss / 1024**3


def get_vram_gb() -> float:
    """返回当前 GPU 显存占用（GB）。"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def format_bytes(n_bytes: int) -> str:
    """将字节数格式化为易读字符串。"""
    if n_bytes >= 1024**3:
        return f"{n_bytes / 1024**3:.1f} GB"
    elif n_bytes >= 1024**2:
        return f"{n_bytes / 1024**2:.1f} MB"
    else:
        return f"{n_bytes / 1024:.1f} KB"


def format_params(n_params: int) -> str:
    """将参数量格式化为易读字符串。"""
    if n_params >= 1e9:
        return f"{n_params / 1e9:.2f}B"
    elif n_params >= 1e6:
        return f"{n_params / 1e6:.2f}M"
    else:
        return f"{n_params / 1e3:.2f}K"


def print_memory_usage(label: str = ""):
    """打印当前内存使用情况。"""
    ram_gb = get_ram_gb()
    vram_gb = get_vram_gb()

    if ram_gb > RAM_WARN_GB:
        ram_str = f"**{ram_gb:.2f} GB** (超过警告阈值 {RAM_WARN_GB} GB!)"
    else:
        ram_str = f"{ram_gb:.2f} GB"

    print(f"[Memory {label}] RAM={ram_str}, VRAM={vram_gb:.2f} GB")


def mem_postfix() -> dict:
    """
    返回一个 dict，适合直接传给 tqdm.set_postfix()。
    """
    ram_gb = get_ram_gb()
    vram_gb = get_vram_gb()
    if ram_gb > RAM_WARN_GB:
        ram_str = f"**{ram_gb:.1f}GB"
    else:
        ram_str = f"{ram_gb:.1f}GB"
    return {"RAM": ram_str, "VRAM": f"{vram_gb:.1f}GB"}


def peak_vram_gb() -> float:
    """返回迄今为止 GPU 显存峰值（GB）。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def reset_peak_vram():
    """重置 GPU 显存峰值记录。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
