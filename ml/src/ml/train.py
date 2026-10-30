"""训练入口：``uv run ml-train`` / ``python -m ml.train``。

这里也放着一条训练链路的所有组装函数（DataModule / 主模型 / Logger / Trainer），
``ml-test`` 与 ``ml-infer`` 复用其中的 :func:`build_datamodule` / :func:`build_module`
与 checkpoint 载入逻辑，保证三个命令搭出来的模型完全一致。

训练只做训练 + 验证，**不会**自动跑测试集：测试集留给 ``ml-test`` 手动评测，
否则它会被当成第二个验证集反复调参，报出来的指标就是乐观偏差。
"""

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


def find_checkpoint(cfg: Config, checkpoint: Path | None = None) -> Path:
    """定位要评测 / 推理的 checkpoint。

    训练会在每个 version 目录里存两份（见 :func:`build_trainer`）：``best.ckpt``（按
    ``val/f1_macro`` 挑的）和 ``last.ckpt``（最后一个 epoch）。这里默认取**最新一次训练**
    里的 ``best.ckpt``；想看最后一个 epoch 的效果就显式传 ``.../last.ckpt``。

    Args:
        cfg: 配置，用 ``[logger] save_dir`` / ``name`` 推断默认位置。
        checkpoint: 显式指定的路径；``None`` 时取 ``save_dir/name`` 下最新的
            ``version_*/checkpoints/*.ckpt`` 所在目录里的 ``best.ckpt``
            （Lightning 每次训练新建一个 version 目录）。

    Raises:
        SystemExit: 显式路径不存在，或默认目录下没有任何 checkpoint。
    """
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint 不存在: {checkpoint}")
        return checkpoint
    root = Path(cfg.logger.save_dir) / cfg.logger.name
    candidates = sorted(
        (path for version in root.glob("version_*") for path in (version / "checkpoints").glob("*.ckpt")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"{root} 下没有任何 checkpoint：先跑 `uv run ml-train --config <配置>`，或用 --checkpoint 显式指定路径"
        )
    newest = candidates[0]
    best = newest.parent / "best.ckpt"
    chosen = best if best.is_file() else newest
    LOGGER.info("未指定 --checkpoint，使用 %s", chosen)
    return chosen


def build_module_from_checkpoint(cfg: Config, checkpoint: Path) -> ByteClassifierModule:
    """按配置重建主模型，再载入 checkpoint 里的权重。

    权重直接存在 ``state_dict`` 里（``[ema] enabled`` 时存的就是 EMA 平均后的权重），
    所以不需要 Lightning 从超参重建模型 —— ``lightning_logs`` 里的 hparams 也不参与。
    代价是**必须用训练时那份配置**：架构超参（宽度、轮数、类别数……）对不上时
    ``load_state_dict(strict=True)`` 会直接报错；``max_length`` 这类不影响参数形状的
    字段对不上则不会报错，但会改变输入，指标不可比。

    Raises:
        FileNotFoundError: checkpoint 不存在。
        RuntimeError: 权重与配置搭出来的模型不匹配（架构超参不一致）。
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    # 本地自己训练出来的文件，里面有回调状态等非张量对象，不能用 weights_only=True 读。
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module = build_module(cfg)
    module.load_state_dict(state["state_dict"], strict=True)
    LOGGER.info(
        "已载入 checkpoint %s（epoch %s，global_step %s）",
        checkpoint,
        state.get("epoch"),
        state.get("global_step"),
    )
    return module


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
    """按配置构建 Trainer，并挂上 TensorBoard 日志与 checkpoint 回调。

    checkpoint 存两份（都在 ``save_dir/name/version_*/checkpoints/`` 下）：

    * ``best.ckpt``：``val/f1_macro`` 最好的那个 epoch，评测 / 推理默认用它；
    * ``last.ckpt``：最后一个 epoch（``save_last=True``），用于看训练结束时的状态，
      或在被打断后接着跑。
    """
    callbacks: list[Callback] = [
        ModelCheckpoint(
            monitor="val/f1_macro",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="best",
        ),
    ]
    if cfg.ema.enabled:
        # Lightning 自带的 EMA：验证会用到平均后的权重，checkpoint 里存的也是 EMA 权重。
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
    """解析参数、读取配置并训练；跑完不碰测试集（用 ``ml-test`` 单独评测）。"""
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
    LOGGER.info(
        "训练结束：best.ckpt（val/f1_macro 最优）与 last.ckpt（最后一个 epoch）在 "
        "%s/%s/version_*/checkpoints/；测试集请用 `uv run ml-test` 单独评测",
        cfg.logger.save_dir,
        cfg.logger.name,
    )


if __name__ == "__main__":
    train()
