"""训练入口：``uv run ml-train`` / ``python -m ml.train``。"""

import argparse
import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EMAWeightAveraging, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from fishetbl.base.BaseModule import BaseTensorModule

from .byte_film import ByteFiLMDenseNet
from .compilation import configure_compilation
from .config import Config, load_config
from .data import ByteClassificationDataModule
from .model import ByteFirstTokenClassifier
from .module import ByteClassifierModule

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.toml"
LOGGER = logging.getLogger(__name__)


def build_datamodule(cfg: Config) -> ByteClassificationDataModule:
    """按配置构建数据集模块（类别数会与 ``[model]`` 交叉校验）。"""
    return ByteClassificationDataModule(cfg.data, num_classes=cfg.model.num_classes, seed=cfg.trainer.seed)


def build_module(cfg: Config) -> ByteClassifierModule:
    """按 ``[model] arch`` 构建主模型并用 Lightning 模块包装。"""
    if cfg.model.arch == "byte_film":
        model: BaseTensorModule = ByteFiLMDenseNet(
            vocab_size=cfg.model.vocab_size,
            num_classes=cfg.model.num_classes,
            arch=cfg.arch,
        )
    else:
        model = ByteFirstTokenClassifier(
            vocab_size=cfg.model.vocab_size,
            num_classes=cfg.model.num_classes,
            bias=cfg.model.bias,
        )
    return ByteClassifierModule(model, cfg)


def build_logger(cfg: Config) -> TensorBoardLogger:
    """TensorBoard 日志：指标与超参都写进 ``save_dir/name/version_*``。"""
    logger = TensorBoardLogger(
        save_dir=cfg.logger.save_dir,
        name=cfg.logger.name,
        log_graph=cfg.logger.log_graph,
    )
    logger.log_hyperparams(_flatten("", cfg))
    return logger


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    """把配置 dataclass 展平成 TensorBoard 能接受的扁平 ``key -> 基础类型`` 字典。"""
    if is_dataclass(value) and not isinstance(value, type):
        flat: dict[str, Any] = {}
        for field in fields(value):
            key = f"{prefix}.{field.name}" if prefix else field.name
            flat |= _flatten(key, getattr(value, field.name))
        return flat
    return {prefix: value}


def build_trainer(cfg: Config) -> pl.Trainer:
    """按配置构建 Trainer，并挂上 TensorBoard 日志与按 ``val/f1`` 保存的 checkpoint。"""
    callbacks: list[Callback] = [
        ModelCheckpoint(monitor="val/f1", mode="max", save_top_k=1, filename="best"),
    ]
    if cfg.ema.enabled:
        # Lightning 自带的 EMA：验证 / 测试会用平均后的权重。
        callbacks.append(EMAWeightAveraging(decay=cfg.ema.decay, use_buffers=cfg.ema.use_buffers))
    return pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        logger=build_logger(cfg),
        callbacks=callbacks,
    )


def train(argv: list[str] | None = None) -> None:
    """解析参数、读取配置，跑完训练后在同一份权重上做测试。"""
    parser = argparse.ArgumentParser(description="THUCNews 字节级最小可行分类模型")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML 配置文件路径")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pl.seed_everything(cfg.trainer.seed, workers=True)
    configure_compilation(cfg.compile)
    # 进程级全局设置，要在建模型/跑图之前设好；CUDA 上才会真正切到 TF32。
    torch.set_float32_matmul_precision(cfg.trainer.float32_matmul_precision)
    LOGGER.info("配置: %s", cfg)

    datamodule = build_datamodule(cfg)
    module = build_module(cfg)
    trainer = build_trainer(cfg)

    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)


if __name__ == "__main__":
    train()
