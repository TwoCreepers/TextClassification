"""THUCNews 预处理：按分类 8:1:1 切成 train/val/test，并生成顺序字节 pack。

原始数据（本仓库的 ``data/THUCNews``）每个分类一个目录、每篇文章一个 UTF-8 文本::

    THUCNews/{分类}/{数字id}.txt

输出（训练只读这些）::

    THUCNews/
        dataset_info.json           # 分类顺序、各集合 / 分类的文档数、划分参数
        packs/{split}.bin           # 该集合所有文档按顺序拼接的 UTF-8 字节
        packs/{split}.offsets.npy   # int64[n+1]，第 i 篇的起止偏移
        packs/{split}.labels.npy    # int64[n]，第 i 篇的类别编号

划分是"每个分类各自洗牌后按 8:1:1 切"再合并，所以三个集合都包含全部 14 个分类；洗牌种子
由 ``--seed`` 和分类名派生，同一份数据重复运行结果完全一致（pack 逐字节相同）。

Note:
    83.6 万篇每篇都要读一次，而 Windows 上打开/读取单个小文件受安全软件逐个扫描拖累（约
    3~5 ms），所以读用线程池并发（默认 16 线程，实测快 8 倍），全量约 2 分钟。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("ml.preprocess")

SPLITS: tuple[str, str, str] = ("train", "val", "test")
INFO_NAME = "dataset_info.json"
PACK_DIR = "packs"
PACK_SUFFIXES = (".bin", ".offsets.npy", ".labels.npy")


def _clear_packs(pack_dir: Path) -> None:
    """删掉上一次留下的 pack 文件。

    ``PackWriter`` 用 ``"wb"`` 打开，本来就会截断，这里删一遍只是为了中途失败时不留半成品：
    少了任何一个文件，数据集模块会直接报「缺少 pack」而不是读到错位的数据。
    """
    for split in SPLITS:
        for suffix in PACK_SUFFIXES:
            (pack_dir / f"{split}{suffix}").unlink(missing_ok=True)


def discover_classes(source: Path) -> list[Path]:
    """列出源目录下的分类子目录，按名字排序（类别编号由这个顺序决定）。"""
    if not source.is_dir():
        raise SystemExit(f"源目录不存在: {source}")
    classes = sorted(
        (path for path in source.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name,
    )
    if not classes:
        raise SystemExit(f"{source} 下没有任何分类目录")
    return classes


def _numeric_key(path: Path) -> tuple[int, int, str]:
    """``123.txt`` 按数值排，避免 ``10.txt`` 排在 ``2.txt`` 前面。"""
    stem = path.stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def collect_documents(class_dir: Path, limit: int = 0) -> list[Path]:
    """收集一个分类下的所有 ``*.txt``（按文件名的数值顺序），``limit`` 只用于调试。

    Note:
        用 ``os.scandir`` 而不是 ``Path.iterdir()``：单个分类最多 15 万个文件，
        ``DirEntry.is_file()`` 直接读目录枚举里缓存的属性，不会每篇再 stat 一次
        （15 万次 stat 在 Windows 上要几十秒，是这里的主要开销）。
    """
    with os.scandir(class_dir) as entries:
        documents = [Path(entry.path) for entry in entries if entry.is_file() and entry.name.endswith(".txt")]
    documents.sort(key=_numeric_key)
    return documents[:limit] if limit > 0 else documents


def split_sizes(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """按比例算出 train/val/test 的篇数；``count >= 3`` 时保证三个集合都非空。"""
    sizes = [int(count * ratios[0]), int(count * ratios[1])]
    sizes.append(count - sum(sizes))
    if count >= len(sizes):
        for index in (1, 2):
            if sizes[index] == 0:
                sizes[0] -= 1
                sizes[index] = 1
    return sizes[0], sizes[1], sizes[2]


def partition(
    documents: list[Path],
    ratios: tuple[float, float, float],
    seed: int,
    class_name: str,
) -> dict[str, list[Path]]:
    """对一个分类的文档洗牌后按 ``ratios`` 切开。

    种子由 ``seed`` 与分类名一起派生：分类之间互不影响，结果只取决于这两者，
    因此重复运行（甚至只处理某个分类）都能得到同一份划分。
    """
    order = list(documents)
    random.Random(f"{seed}:{class_name}").shuffle(order)
    n_train, n_val, _ = split_sizes(len(order), ratios)
    return {
        SPLITS[0]: order[:n_train],
        SPLITS[1]: order[n_train : n_train + n_val],
        SPLITS[2]: order[n_train + n_val :],
    }


class PackWriter:
    """把一个 split 的所有文档顺序写进 ``{split}.bin``，同时记录偏移与标签。"""

    def __init__(self, pack_dir: Path, split: str) -> None:
        self.split = split
        self.bin_path = pack_dir / f"{split}.bin"
        self.offsets_path = pack_dir / f"{split}.offsets.npy"
        self.labels_path = pack_dir / f"{split}.labels.npy"
        self._handle = self.bin_path.open("wb")
        self._offsets = [0]
        self._labels: list[int] = []
        self.total_bytes = 0

    def add(self, raw: bytes, label: int) -> None:
        self._handle.write(raw)
        self.total_bytes += len(raw)
        self._offsets.append(self.total_bytes)
        self._labels.append(label)

    def close(self) -> None:
        """关文件并把偏移 / 标签落成 ``.npy``（可直接 mmap 读取）。"""
        self._handle.close()
        np.save(self.offsets_path, np.asarray(self._offsets, dtype=np.int64))
        np.save(self.labels_path, np.asarray(self._labels, dtype=np.int64))
        LOGGER.info("[pack] %s: %d 篇 / %.2f GiB", self.split, len(self._labels), self.total_bytes / 1024**3)


def _write_split(
    documents: list[Path],
    writer: PackWriter,
    label: int,
    pool: ThreadPoolExecutor,
    chunk_size: int = 4096,
) -> None:
    """把 ``documents`` 按顺序追加进 ``writer``。

    ``pool.map`` 会先把这一块的读请求全部提交，再按顺序返回结果，于是"读第 i+1 篇"与
    "写第 i 篇"重叠起来；分块是为了把在途的文档内容限制在几 MB。
    """
    for start in range(0, len(documents), chunk_size):
        chunk = documents[start : start + chunk_size]
        for raw in pool.map(Path.read_bytes, chunk):
            writer.add(raw, label)


@dataclass
class PrepareResult:
    """预处理结果概要。"""

    class_names: list[str]
    totals: dict[str, int]
    per_class: dict[str, dict[str, int]]
    seconds: float


def prepare(
    source: Path,
    output: Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    limit_per_class: int = 0,
    overwrite: bool = False,
    workers: int = 16,
) -> PrepareResult:
    """执行划分与打包，返回各集合的统计信息。

    Args:
        source: 原始根目录，每个分类一个子目录。
        output: 输出根目录。
        seed: 洗牌种子（与分类名一起派生每个分类的 RNG）。
        ratios: train/val/test 的比例，必须为正且和为 1。
        limit_per_class: 每个分类只取前 N 篇，0 表示全部（调试用）。
        overwrite: 目标已存在时是否覆盖。
        workers: 并发读文件的线程数。
    """
    classes = discover_classes(source)
    class_names = [path.name for path in classes]
    LOGGER.info("源目录 %s：%d 个分类 %s", source, len(classes), class_names)
    LOGGER.info("比例 train/val/test = %.2f/%.2f/%.2f，seed=%d，workers=%d", *ratios, seed, workers)

    try:
        output.resolve().relative_to(source.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit(f"--output {output} 不能放在 --source {source} 里面，否则分类会被重复扫描")

    if (output / INFO_NAME).is_file() and not overwrite:
        raise SystemExit(f"{output} 已经处理过（存在 {INFO_NAME}），如需重跑请加 --overwrite")

    output.mkdir(parents=True, exist_ok=True)

    pack_dir = output / PACK_DIR
    pack_dir.mkdir(parents=True, exist_ok=True)
    _clear_packs(pack_dir)
    writers = {split: PackWriter(pack_dir, split) for split in SPLITS}

    per_class: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    totals: dict[str, int] = {split: 0 for split in SPLITS}
    started = time.perf_counter()
    failed = False
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for class_index, class_dir in enumerate(classes):
                documents = collect_documents(class_dir, limit_per_class)
                if not documents:
                    LOGGER.warning("分类 %s 没有任何 txt，跳过", class_dir.name)
                    continue
                parts = partition(documents, ratios, seed, class_dir.name)
                for split in SPLITS:
                    _write_split(parts[split], writers[split], class_index, pool)
                    per_class[split][class_dir.name] = len(parts[split])
                    totals[split] += len(parts[split])
                elapsed = time.perf_counter() - started
                LOGGER.info(
                    "[%2d/%d] %-4s %8d 篇 -> train %6d / val %5d / test %5d（累计 %.0fs）",
                    class_index + 1,
                    len(classes),
                    class_dir.name,
                    len(documents),
                    len(parts[SPLITS[0]]),
                    len(parts[SPLITS[1]]),
                    len(parts[SPLITS[2]]),
                    elapsed,
                )
    except BaseException:
        failed = True
        raise
    finally:
        for writer in writers.values():
            writer.close()
        if failed:
            _clear_packs(pack_dir)

    seconds = time.perf_counter() - started
    info: dict[str, Any] = {
        "format": "thucnews-byte-pack",
        "version": 1,
        "source": str(source),
        "seed": seed,
        "ratios": list(ratios),
        "limit_per_class": limit_per_class,
        "class_names": class_names,
        "splits": {split: {"total": totals[split], "per_class": per_class[split]} for split in SPLITS},
    }
    (output / INFO_NAME).write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for split in SPLITS:
        missing = [name for name in class_names if per_class[split].get(name, 0) == 0]
        if missing:
            LOGGER.warning("%s 集合缺少分类: %s", split, missing)
    LOGGER.info(
        "完成：train %d / val %d / test %d，共 %d 篇，耗时 %.1fs，输出 %s",
        totals[SPLITS[0]],
        totals[SPLITS[1]],
        totals[SPLITS[2]],
        sum(totals.values()),
        seconds,
        output,
    )
    return PrepareResult(class_names=class_names, totals=totals, per_class=per_class, seconds=seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-prepare-thucnews",
        description="把原始 THUCNews 按分类 8:1:1 切成 train/val/test，并生成顺序字节 pack。",
    )
    parser.add_argument("--source", type=Path, default=Path("data/THUCNews"), help="原始根目录（每个分类一个子目录）")
    parser.add_argument("--output", type=Path, default=Path("data/processed/THUCNews"), help="输出根目录")
    parser.add_argument("--seed", type=int, default=42, help="每个分类洗牌的随机种子")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test 比例（默认 0.8 0.1 0.1）",
    )
    parser.add_argument(
        "--limit-per-class", type=int, default=0, metavar="N", help="每个分类只取前 N 篇（调试用，0 表示全部）"
    )
    parser.add_argument("--workers", type=int, default=16, help="并发读文件的线程数（默认 16）")
    parser.add_argument("--overwrite", action="store_true", help="目标已存在时覆盖")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ratios = tuple(float(value) for value in args.ratios)
    if len(ratios) != 3 or any(value <= 0.0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error(f"--ratios 必须为正且和为 1，收到 {ratios}")
    train_ratio, val_ratio, test_ratio = ratios

    prepare(
        args.source,
        args.output,
        seed=args.seed,
        ratios=(train_ratio, val_ratio, test_ratio),
        limit_per_class=args.limit_per_class,
        overwrite=args.overwrite,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
