"""检查点评测入口：``uv run ml-test`` / ``python -m ml.test``。

用一个已有的 checkpoint 在**验证集或测试集**上跑一遍指标，不做训练、不写 TensorBoard。
训练入口已经不再自动碰测试集（见 ``ml.train``），测试集要靠这个命令手动跑：

* 挑模型 / 调超参看验证集（``--split val``，也就是训练时 ``ModelCheckpoint`` 盯的
  ``val/f1_macro``）；
* 测试集（``--split test``，默认）只在要汇报结果时跑，跑完不改模型。

指标名里写明平均方式：``f1_macro`` / ``f1_micro`` / ``top1_acc_micro``。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pytorch_lightning as pl
import torch

from .compilation import configure_compilation
from .config import Config, load_config
from .train import DEFAULT_CONFIG_PATH, build_datamodule, build_module_from_checkpoint, find_checkpoint

SPLITS = ("val", "test")

# 报告里每个指标的含义：写清 macro / micro，避免两种平均被混为一谈。
METRIC_NOTES = {
    "loss": "交叉熵（按样本加权平均）",
    "f1_macro": "macro F1：先按类别算 F1 再算术平均，每个类别同权",
    "f1_micro": "micro F1：先累计的 TP/FP/FN 再算 F1，单标签多分类下等于准确率",
    "top1_acc_micro": "micro top1 准确率：argmax 命中的样本占比",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-test",
        description="在验证集或测试集上评测一个 checkpoint（F1 分 macro 与 micro，top1 是 micro）。",
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
    parser.add_argument("--split", choices=SPLITS, default="test", help="评测哪个集合（默认 test）")
    parser.add_argument("--batch-size", type=int, default=None, help="覆盖 [data] batch_size（只影响速度）")
    # 用 int 而不是 float：Lightning 把 1.0 当作"100% 的 batch"，1 当作"1 个 batch"，容易踩坑。
    parser.add_argument("--limit-batches", type=int, default=None, help="只跑前 N 个 batch（调试用；省略表示全部）")
    return parser


def build_eval_trainer(cfg: Config) -> pl.Trainer:
    """评测用 Trainer：不写日志、不存 checkpoint、不跑 sanity check。"""
    return pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        inference_mode=True,
        num_sanity_val_steps=0,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
    )


def evaluate(cfg: Config, checkpoint: Path, split: str) -> dict[str, float]:
    """重建模型并跑一遍指定集合，返回 Lightning 记录的指标（键形如 ``test/f1_macro``）。"""
    datamodule = build_datamodule(cfg)
    module = build_module_from_checkpoint(cfg, checkpoint)
    trainer = build_eval_trainer(cfg)
    results = (
        trainer.validate(module, datamodule=datamodule)
        if split == "val"
        else trainer.test(module, datamodule=datamodule)
    )
    if not results:
        return {}
    # Lightning 每个 dataloader 返回一个 dict，这里只有一个 dataloader。
    return {str(key): float(value) for key, value in results[0].items()}


def format_report(split: str, checkpoint: Path, metrics: dict[str, float]) -> str:
    """把指标排版成一段可以直接贴进实验记录的报告。"""
    lines = [f"=== {split} 集评测 ===", f"checkpoint: {checkpoint}", ""]
    for name, note in METRIC_NOTES.items():
        value = metrics.get(f"{split}/{name}")
        if value is not None:
            lines.append(f"{split}/{name:<22}{value:>9.4f}  {note}")
    lines += [
        "",
        "选择模型看 val/f1_macro（训练时 ModelCheckpoint 的 monitor）；测试集只用来汇报，不参与调参。",
    ]
    return "\n".join(lines)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """把命令行覆盖项写回配置；配置是 frozen dataclass，用 ``replace`` 生成新实例。"""
    data = cfg.data
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise SystemExit(f"--batch-size 必须为正整数，收到 {args.batch_size}")
        data = replace(data, batch_size=args.batch_size)
    trainer = cfg.trainer
    if args.limit_batches is not None:
        if args.limit_batches <= 0:
            raise SystemExit(f"--limit-batches 必须为正整数，收到 {args.limit_batches}")
        trainer = replace(trainer, limit_val_batches=args.limit_batches, limit_test_batches=args.limit_batches)
    return replace(cfg, data=data, trainer=trainer)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = _apply_overrides(load_config(args.config), args)
    configure_compilation(cfg.compile)
    # 进程级全局设置，要在建模型之前设好；CUDA 上才会真正切到 TF32。
    torch.set_float32_matmul_precision(cfg.trainer.float32_matmul_precision)

    checkpoint = find_checkpoint(cfg, args.checkpoint)
    metrics = evaluate(cfg, checkpoint, args.split)
    print(format_report(args.split, checkpoint, metrics))


if __name__ == "__main__":
    main()
