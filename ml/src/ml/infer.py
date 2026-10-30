"""推理入口：``uv run ml-infer`` / ``python -m ml.infer``。

对一段或多段文本做分类，输出**全部分类**的 softmax 概率，并按概率从高到低排序。
输入按 UTF-8 编码成字节（训练时读的就是 ``.txt`` 的原始字节），截到 ``[data] max_length``
（``-1`` 表示不截断），其余与训练一致：右侧补 0、``padding mask`` 挡住 pad 位置、模型切到 eval。

示例::

    uv run ml-infer "商务部：前三季度社会消费品零售总额同比增长 8.2%"
    uv run ml-infer --file data/THUCNews/体育/1.txt
    Get-Content 一篇新闻.txt | uv run ml-infer
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .compilation import configure_compilation
from .config import Config, load_config
from .data import INFO_NAME, PAD_TOKEN_ID, DatasetInfo, fixed_padding_length, pad_byte_sequences, truncation_limit
from .module import ByteClassifierModule
from .train import DEFAULT_CONFIG_PATH, build_module_from_checkpoint, find_checkpoint

LOGGER = logging.getLogger(__name__)

# 预览文本的最大字符数（只影响显示）。
PREVIEW_LENGTH = 60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-infer",
        description="用 checkpoint 给文本分类，输出全部分类的 softmax 概率（降序）。",
    )
    parser.add_argument("text", nargs="*", metavar="TEXT", help="要分类的文本（可给多段）；不给则读 --file 或标准输入")
    parser.add_argument(
        "--file", type=Path, action="append", default=None, metavar="PATH", help="从文件读原始字节（可重复）"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML 配置文件路径（必须与训练时一致）"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="检查点路径；省略时取最新一次训练的 best.ckpt（要最后一个 epoch 就传 last.ckpt）",
    )
    parser.add_argument("--device", default="auto", help="推理设备：auto（有 CUDA 就用）/ cpu / cuda")
    parser.add_argument("--json", action="store_true", help="输出 JSON（便于脚本消费）")
    return parser


def load_class_names(cfg: Config) -> list[str]:
    """从 ``[data] root`` 的 ``dataset_info.json`` 读分类名（下标即标签编号）。

    读不到时回落到 ``class_<编号>``：只影响展示，不影响概率本身。

    Raises:
        SystemExit: 文件里的分类数与配置的 ``[model] num_classes`` 不一致。
    """
    root = Path(cfg.data.root)
    try:
        info = DatasetInfo.load(root)
    except FileNotFoundError:
        LOGGER.warning("找不到 %s，类别只能用 class_<编号> 显示（不影响概率）", root / INFO_NAME)
        return [f"class_{index}" for index in range(cfg.model.num_classes)]
    if len(info.class_names) != cfg.model.num_classes:
        raise SystemExit(
            f"{root / INFO_NAME} 里有 {len(info.class_names)} 个分类，"
            f"与配置的 [model] num_classes={cfg.model.num_classes} 不一致"
        )
    return info.class_names


def read_inputs(texts: Sequence[str], files: Sequence[Path] | None) -> list[tuple[str, bytes]]:
    """把命令行文本 / 文件 / 标准输入统一成 ``(来源说明, 原始字节)``。

    命令行文本与标准输入按 UTF-8 编码；文件直接读原始字节，和训练时 ``Path.read_bytes``
    完全一致（先解码再编码可能改变字节）。

    Raises:
        SystemExit: 指定的文件不存在，或根本没有输入。
    """
    items: list[tuple[str, bytes]] = [
        (f"arg{index}", text.encode("utf-8")) for index, text in enumerate(texts, start=1)
    ]
    for path in files or ():
        if not path.is_file():
            raise SystemExit(f"文件不存在: {path}")
        items.append((str(path), path.read_bytes()))
    if items:
        return items
    if sys.stdin is None or sys.stdin.isatty():
        raise SystemExit("没有输入：给一段文本、加 --file，或从标准输入喂文本")
    raw = sys.stdin.buffer.read()
    if not raw.strip():
        raise SystemExit("标准输入是空的")
    return [("<stdin>", raw)]


def resolve_device(name: str) -> torch.device:
    """把 ``--device`` 解析成具体设备；``auto`` 时有 CUDA 就用 CUDA。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(name)
    except RuntimeError as exc:
        raise SystemExit(f"无法识别的 --device {name}: {exc}") from exc


def predict_logits(
    module: ByteClassifierModule,
    cfg: Config,
    documents: Sequence[bytes],
    device: torch.device,
) -> Tensor:
    """对 ``documents``（已按 ``max_length`` 截断）做一次前向，返回 CPU 上的 ``(B, num_classes)`` logits。"""
    # 补齐规则与 ByteClassificationDataModule._loader 一致：max_length=-1 不截断 → 只能变长补齐；
    # 否则 pad_to_max_length 时补到 max_length，关闭时补到 batch 内最长。截断在调用方做。
    input_ids = pad_byte_sequences(
        [np.frombuffer(document, dtype=np.uint8) for document in documents],
        fixed_padding_length(cfg.data),
    ).to(device)
    padding_mask = input_ids != PAD_TOKEN_ID if cfg.model.use_padding_mask else None
    with torch.inference_mode():
        logits = module(input_ids, padding_mask)
    return logits.float().cpu()


def _ranking(probabilities: Sequence[float]) -> list[int]:
    """按概率降序返回类别下标（稳定排序：概率相同时保持标签编号顺序）。"""
    return sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)


def _display_width(text: str) -> int:
    """终端显示宽度：东亚宽字符（汉字、全角标点）占 2 列，用来对齐中英混排的表格。"""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _pad(text: str, width: int, *, right: bool = False) -> str:
    """按显示宽度补空格到 ``width`` 列；``right=True`` 靠右（数字列）。"""
    spaces = " " * max(0, width - _display_width(text))
    return spaces + text if right else text + spaces


def _table_row(rank: str, label: str, probability: str, logit: str, label_width: int) -> str:
    """表格的一行：排名 / 分类 / 概率 / logit，四列宽度固定（分类列随名字宽度自适应）。"""
    cells = (_pad(rank, 4, right=True), _pad(label, label_width), _pad(probability, 8, right=True))
    return "  " + "  ".join(cells) + "  " + _pad(logit, 9, right=True)


def _preview(document: bytes, limit: int = PREVIEW_LENGTH) -> str:
    """把文档字节解码成一行预览；坏字节换成 ``?``，过长截断。"""
    text = document.decode("utf-8", errors="replace").replace("\ufffd", "?")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def format_ranking(
    source: str,
    document: bytes,
    probabilities: Sequence[float],
    logits: Sequence[float],
    class_names: Sequence[str],
) -> str:
    """把一条输入的**全部**分类排成表格：softmax 概率降序，概率与 logit 一起给出。"""
    label_width = max(_display_width(name) for name in class_names)
    lines = [
        f"输入 {source}: {len(document)} 字节",
        f"内容: {_preview(document)}",
        _table_row("排名", "分类", "概率", "logit", label_width),
    ]
    for rank, class_id in enumerate(_ranking(probabilities), start=1):
        lines.append(
            _table_row(
                str(rank),
                class_names[class_id],
                f"{probabilities[class_id]:.4f}",
                f"{logits[class_id]:.4f}",
                label_width,
            )
        )
    return "\n".join(lines)


def ranking_payload(
    source: str,
    document: bytes,
    probabilities: Sequence[float],
    logits: Sequence[float],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """``--json`` 用的结构：来源、字节数、全部分类的概率与 logit（降序）。"""
    return {
        "source": source,
        "bytes": len(document),
        "preview": _preview(document),
        "predictions": [
            {
                "rank": rank,
                "class_id": class_id,
                "label": class_names[class_id],
                "prob": round(float(probabilities[class_id]), 6),
                "logit": round(float(logits[class_id]), 4),
            }
            for rank, class_id in enumerate(_ranking(probabilities), start=1)
        ],
    }


def infer(argv: list[str] | None = None) -> None:
    """解析参数、载入 checkpoint，对输入逐条输出全部分类的 softmax 概率。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    # 单条 / 少量文本不值得走 torch.compile：省掉首次调用的编译时间（CPU 上还要本地 C++ 编译器）。
    cfg = replace(cfg, compile=replace(cfg.compile, enabled=False))

    # 先读输入并截断到训练时的 max_length（-1 表示不截断）：输入有问题就直接退出，不用白载一遍模型。
    # 训练时 THUCNewsDataset 也按同一规则截断（pad_to_max_length 只影响补齐，不影响截断）。
    limit = truncation_limit(cfg.data.max_length)
    pairs = [
        (source, document if limit is None else document[:limit])
        for source, document in read_inputs(args.text, args.file)
    ]
    empty = [source for source, document in pairs if not document]
    if empty:
        raise SystemExit(f"输入为空，无法分类: {', '.join(empty)}")

    configure_compilation(cfg.compile)
    torch.set_float32_matmul_precision(cfg.trainer.float32_matmul_precision)
    checkpoint = find_checkpoint(cfg, args.checkpoint)
    class_names = load_class_names(cfg)
    device = resolve_device(args.device)
    module = build_module_from_checkpoint(cfg, checkpoint).to(device)
    module.eval()

    logits = predict_logits(module, cfg, [document for _, document in pairs], device)
    probabilities = logits.softmax(dim=-1).tolist()
    logit_rows = logits.tolist()

    if args.json:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "results": [
                        ranking_payload(source, document, probabilities[row], logit_rows[row], class_names)
                        for row, (source, document) in enumerate(pairs)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    blocks = [
        format_ranking(source, document, probabilities[row], logit_rows[row], class_names)
        for row, (source, document) in enumerate(pairs)
    ]
    print(f"checkpoint: {checkpoint}")
    print("\n\n".join(blocks))


if __name__ == "__main__":
    infer()
