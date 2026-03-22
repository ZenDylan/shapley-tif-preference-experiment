"""
JSON / JSONL 文件读写工具。
封装常见操作，支持追加写 JSONL（用于 Shapley 缓存）和断点续传。
"""

import json
import os


def save_json(data, path: str):
    """将 data 写入 JSON 文件（indent=2）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [IO] 保存到 {path}")


def load_json(path: str):
    """从 JSON 文件读取并返回 Python 对象。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(entry: dict, path: str):
    """
    向 JSONL 文件追加一条记录（每行一个 JSON）。
    目录不存在时自动创建。
    """
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    """
    读取整个 JSONL 文件，返回 list[dict]。
    文件不存在时返回空列表。
    """
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_jsonl_as_dict(path: str, key_field: str = "coalition") -> dict:
    """
    读取 JSONL 文件，将每条记录转为 dict，
    key 为指定字段（coalition 转成 tuple 作为 key）。
    用于断点续传：快速查找已计算的 coalition。
    """
    entries = load_jsonl(path)
    result = {}
    for e in entries:
        k = e.get(key_field)
        if isinstance(k, list):
            k = tuple(sorted(k))
        result[k] = e
    return result
