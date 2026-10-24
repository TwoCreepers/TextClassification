"""UTF-8 字节级数据集：把文本编码成 byte token，0 号 token 兼作 padding。"""

from typing import Any

import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from .config import DataConfig

# 0 号 token 直接作为 padding：正常文本的 UTF-8 编码几乎不会出现 0x00。
PAD_TOKEN_ID = 0


def encode_bytes(text: str, max_length: int) -> list[int]:
    """把文本编码成 UTF-8 字节序列（token id 即字节值），并按 ``max_length`` 截断。

    Args:
        text: 原始文本。
        max_length: 最大 token 数，超出部分丢弃。

    Returns:
        字节值组成的 token id 列表；空文本返回空列表。
    """
    return list(text.encode("utf-8"))[:max_length]


def collate_byte_batch(batch: list[dict[str, Any]]) -> dict[str, Tensor]:
    """按 batch 内最长序列右侧补 0，并堆叠成 ``(B, L)`` 的 ``int64`` 张量。"""
    encoded = [sample["input_ids"] for sample in batch]
    longest = max(1, max(len(ids) for ids in encoded))
    input_ids = torch.tensor(
        [ids + [PAD_TOKEN_ID] * (longest - len(ids)) for ids in encoded],
        dtype=torch.long,
    )
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


class _ByteSplitDataset(TorchDataset[dict[str, Any]]):
    """把 ``datasets.Dataset`` 适配成 ``torch.utils.data.Dataset`` 接口。"""

    def __init__(self, split: Dataset) -> None:
        self._split = split

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._split[index]


class ByteClassificationDataModule(pl.LightningDataModule):
    """把 ``seamew/THUCNews`` 转成字节级分类数据集。

    Args:
        cfg: 数据集相关配置。
        num_classes: 期望的类别数；若与数据集标签集不一致会直接报错。
    """

    def __init__(self, cfg: DataConfig, num_classes: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.expected_num_classes = num_classes
        self.class_names: list[str] = []
        self._splits: dict[str, _ByteSplitDataset] = {}

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def setup(self, stage: str | None = None) -> None:
        """加载并编码三个 split；重复调用时直接复用。"""
        if self._splits:
            return

        raw = load_dataset(self.cfg.dataset_name, revision=self.cfg.revision)
        train_set = raw[self.cfg.train_split]
        label_feature = train_set.features[self.cfg.label_column]
        self.class_names = list(getattr(label_feature, "names", []) or [])

        if self.expected_num_classes is not None and self.class_names:
            if len(self.class_names) != self.expected_num_classes:
                raise ValueError(
                    f"配置里的 num_classes={self.expected_num_classes} "
                    f"与数据集类别数 {len(self.class_names)} 不一致: {self.class_names}"
                )

        self._splits = {
            split: self._encode(raw[split_name])
            for split, split_name in (
                ("train", self.cfg.train_split),
                ("val", self.cfg.val_split),
                ("test", self.cfg.test_split),
            )
        }

    def _encode(self, split: Dataset) -> _ByteSplitDataset:
        """把文本列替换成字节 token 列，其余列（含标签）保留。"""

        def encode_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            return {
                "input_ids": [encode_bytes(text, self.cfg.max_length) for text in batch[self.cfg.text_column]],
            }

        return _ByteSplitDataset(split.map(encode_batch, batched=True, desc=f"byte-encode {split.split}"))

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            self._splits[split],
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=collate_byte_batch,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)
