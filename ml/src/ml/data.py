"""THUCNews 字节级数据集：直接读取预处理生成的顺序字节 pack。

``just prep``（纯 Python）或 ``just prep-dedup``（.NET，先去重再划分）的产物就是一份 pack：每个
split 一段连续 UTF-8 字节 + 偏移 / 标签两个 int64 索引数组。读取时 ``mmap`` 一次、按偏移切片，
避免每轮打开几十万个小文件。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from .config import DataConfig

LOGGER = logging.getLogger(__name__)

# 0 号 token 直接作为 padding：正常文本的 UTF-8 编码几乎不会出现 0x00。
PAD_TOKEN_ID = 0
# [data] max_length 取这个值表示不截断（整篇喂给模型）。
NO_TRUNCATION = -1
INFO_NAME = "dataset_info.json"
PACK_DIR = "packs"


def truncation_limit(max_length: int) -> int | None:
    """把 ``[data] max_length`` 解释成截断长度：正数是保留的字节数，``NO_TRUNCATION`` 表示不截断。

    Returns:
        截断长度；``None`` 表示不截断。

    Raises:
        ValueError: 取值既不是正数也不是 ``NO_TRUNCATION``（``0`` 与更小的负数没有意义）。
    """
    if max_length == NO_TRUNCATION:
        return None
    if max_length <= 0:
        raise ValueError(f"[data] max_length 必须是正数或 {NO_TRUNCATION}（表示不截断），收到 {max_length}")
    return max_length


def fixed_padding_length(cfg: DataConfig) -> int | None:
    """固定补齐长度（即 ``collate_byte_batch`` 的 ``pad_to``）；``None`` 表示按 batch 变长补齐。

    只有「会截断 + ``pad_to_max_length``」才有恒定形状可以补：``max_length = -1`` 时不截断，
    序列长度没有上界，只能变长补齐（``pad_to_max_length`` 在这种情况下不生效）。
    """
    if not cfg.pad_to_max_length:
        return None
    return truncation_limit(cfg.max_length)


@dataclass(frozen=True)
class DatasetInfo:
    """``dataset_info.json`` 里数据集模块需要的那部分。"""

    class_names: list[str]
    split_totals: dict[str, int]

    @classmethod
    def load(cls, root: Path) -> DatasetInfo:
        """读取 ``root/dataset_info.json``。

        Raises:
            FileNotFoundError: 目录里没有 ``dataset_info.json``（还没跑预处理）。
        """
        path = root / INFO_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到 {path}：请先运行 `just prep`（纯 Python，不含去重）或 `just prep-dedup`（.NET，先去重再划分）"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            class_names=list(raw["class_names"]),
            split_totals={name: int(section.get("total", 0)) for name, section in raw.get("splits", {}).items()},
        )


@dataclass(frozen=True)
class _Pack:
    """一个 split 的顺序字节 pack：``data[offsets[i]:offsets[i+1]]`` 是第 i 篇的 UTF-8 字节。"""

    data: np.ndarray
    offsets: np.ndarray
    labels: np.ndarray


def _load_pack(pack_dir: Path, split: str) -> _Pack:
    """mmap 一个 split 的 pack，并校验三个文件互相对得上。

    Raises:
        FileNotFoundError: pack 文件不全（还没跑预处理）。
        ValueError: 偏移 / 标签 / 字节长度不一致（pack 损坏或来自不同版本）。
    """
    bin_path = pack_dir / f"{split}.bin"
    offsets_path = pack_dir / f"{split}.offsets.npy"
    labels_path = pack_dir / f"{split}.labels.npy"
    missing = [path.name for path in (bin_path, offsets_path, labels_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"缺少 {split} 的 pack（{', '.join(missing)}）：请先运行 `just prep` 或 `just prep-dedup`"
            f"（输出到 {pack_dir.parent}）"
        )
    offsets = np.load(offsets_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    if len(offsets) != len(labels) + 1:
        raise ValueError(f"pack 索引损坏：{offsets_path} 有 {len(offsets)} 个偏移，但标签有 {len(labels)} 个")
    if len(labels) == 0:
        raise ValueError(f"{split} 的 pack 是空的：{labels_path}")
    data = np.memmap(bin_path, dtype=np.uint8, mode="r")
    if len(data) != int(offsets[-1]):
        raise ValueError(f"pack 数据损坏：{bin_path} 长 {len(data)} 字节，与索引终点 {int(offsets[-1])} 不一致")
    return _Pack(data=data, offsets=offsets, labels=labels)


class THUCNewsDataset(TorchDataset[dict[str, Any]]):
    """``packs/{split}.*`` 形式的字节级分类数据集。

    Args:
        pack_dir: pack 目录，里面有 ``{split}.bin`` / ``.offsets.npy`` / ``.labels.npy``。
        split: split 名，用作 pack 文件名。
        max_length: 每篇最多保留的 UTF-8 字节数，超出部分截断；``-1`` 表示不截断（整篇喂给模型）。

    Raises:
        ValueError: ``max_length`` 既不是正数也不是 ``-1``。
    """

    def __init__(self, pack_dir: Path, split: str, max_length: int) -> None:
        self.pack_dir = Path(pack_dir)
        self.split = split
        self.max_length = max_length
        # 截断长度；None 表示不截断。与 pad_to_max_length 无关：那个开关只管补齐。
        self._limit = truncation_limit(max_length)
        self._pack = _load_pack(self.pack_dir, split)

    def class_counts(self, num_classes: int) -> np.ndarray:
        """每个类别的文档数（下标即标签编号）；标签越界时会比 ``num_classes`` 长。"""
        return np.bincount(np.asarray(self._pack.labels), minlength=num_classes)

    def __len__(self) -> int:
        return len(self._pack.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self._pack.offsets[index])
        stop = int(self._pack.offsets[index + 1])
        raw = self._pack.data[start:stop]
        if self._limit is not None:
            raw = raw[: self._limit]
        return {"input_ids": np.asarray(raw, dtype=np.int64), "label": int(self._pack.labels[index])}


def pad_byte_sequences(sequences: Sequence[np.ndarray], pad_to: int | None = None) -> Tensor:
    """把若干条字节 / token 序列右侧补 0，堆成 ``(B, L)`` 的 ``int64`` 张量。

    Args:
        sequences: 每条样本的 token id 序列（一维数组或 list，会转成 ``int64``）。
            空序列会补成 ``(1,)``，所以张量至少有一列。
        pad_to: 固定补齐长度。给出时所有 batch 形状一致，可以避免
            ``torch.compile`` 因序列长度变化而反复重编译；``None`` 时补到
            batch 内最长序列。
    """
    longest = max(1, max(len(sequence) for sequence in sequences))
    if pad_to is not None:
        longest = max(pad_to, longest)
    input_ids = np.zeros((len(sequences), longest), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        tokens = np.asarray(sequence, dtype=np.int64)[:longest]
        input_ids[row, : len(tokens)] = tokens
    return torch.from_numpy(input_ids)


def collate_byte_batch(batch: list[dict[str, Any]], pad_to: int | None = None) -> dict[str, Tensor]:
    """把 batch 堆叠成 ``(B, L)`` 的 ``int64`` 张量，右侧补 0。

    Args:
        batch: 单样本字典列表，需含 ``input_ids``（numpy 或 list）与 ``label``。
        pad_to: 固定补齐长度，见 :func:`pad_byte_sequences`。
    """
    input_ids = pad_byte_sequences([sample["input_ids"] for sample in batch], pad_to)
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


class ByteClassificationDataModule(pl.LightningDataModule):
    """把预处理好的 THUCNews pack 转成字节级分类 DataLoader。

    Args:
        cfg: 数据集相关配置。
        num_classes: 期望的类别数；若与 ``dataset_info.json`` 里的分类数不一致会直接报错。
        seed: 随机种子（通常传 ``[trainer] seed``）。三个 split 各自派生一个
            ``torch.Generator``：``seed + split 顺序``，让 shuffle 顺序可复现且互不相同。
    """

    def __init__(self, cfg: DataConfig, num_classes: int | None = None, seed: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.expected_num_classes = num_classes
        self.root = Path(cfg.root)
        self.class_names: list[str] = []
        self._splits: dict[str, THUCNewsDataset] = {}

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def setup(self, stage: str | None = None) -> None:
        """加载三个 split；重复调用时直接复用。"""
        if self._splits:
            return

        if self.cfg.pad_to_max_length and truncation_limit(self.cfg.max_length) is None:
            LOGGER.warning(
                "[data] max_length=%d（不截断）时序列长度没有上界，pad_to_max_length 不生效，将按 batch 变长补齐",
                NO_TRUNCATION,
            )

        info = DatasetInfo.load(self.root)
        self.class_names = info.class_names
        if self.expected_num_classes is not None and self.expected_num_classes != len(self.class_names):
            raise ValueError(
                f"配置里的 num_classes={self.expected_num_classes} "
                f"与数据集类别数 {len(self.class_names)} 不一致: {self.class_names}"
            )

        splits: dict[str, THUCNewsDataset] = {}
        for split in (self.cfg.train_split, self.cfg.val_split, self.cfg.test_split):
            dataset = THUCNewsDataset(self.root / PACK_DIR, split, self.cfg.max_length)
            counts = dataset.class_counts(len(self.class_names))
            if len(counts) != len(self.class_names):
                raise ValueError(
                    f"{split} 的标签越界（出现 {len(counts) - 1}，但只有 {len(self.class_names)} 个分类）："
                    "pack 与 dataset_info.json 不一致，请重新运行 `just prep`"
                )
            missing = [name for name, count in zip(self.class_names, counts) if count == 0]
            if missing:
                raise ValueError(f"{split} 集合缺少分类 {missing}：预处理时每个分类都做了 8:1:1 划分吗？")
            expected_total = info.split_totals.get(split)
            if expected_total is not None and expected_total != len(dataset):
                raise ValueError(
                    f"{split} 实际有 {len(dataset)} 篇，但 {INFO_NAME} 记录 {expected_total} 篇："
                    "pack 与索引不一致，请重新运行 `just prep`"
                )
            splits[split] = dataset
        self._splits = splits

    def _split_seed(self, split: str) -> int:
        """split 的随机种子：配置种子按 ``train/val/test`` 顺序线性错开。"""
        order = (self.cfg.train_split, self.cfg.val_split, self.cfg.test_split)
        offset = order.index(split) if split in order else len(order)
        return self.seed + offset

    def _generator(self, split: str) -> torch.Generator:
        """每个 split 一个手动播种的 ``torch.Generator``，让 shuffle / worker 播种可复现。

        显式传 ``generator`` 后，DataLoader 不再消耗全局 RNG，训练顺序由配置种子决定。
        """
        generator = torch.Generator()
        generator.manual_seed(self._split_seed(split))
        return generator

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        # pad_to 只控制补齐长度：None 表示补到 batch 内最长序列；序列本身已经在
        # THUCNewsDataset.__getitem__ 里按 max_length 截断过了（max_length=-1 不截断时只能变长补齐）。
        pad_to = fixed_padding_length(self.cfg)
        return DataLoader(
            self._splits[split],
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=partial(collate_byte_batch, pad_to=pad_to),
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=self.cfg.pin_memory,
            generator=self._generator(split),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.cfg.train_split, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.cfg.val_split, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.cfg.test_split, shuffle=False)
