"""ml 项目的配置结构：全部来自 ``ml/config/config.toml``。"""

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

# 与 Lightning Trainer 接受的 precision 取值保持一致。
type PrecisionType = Literal[
    "transformer-engine",
    "transformer-engine-float16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32-true",
    "64-true",
    "64",
    "32",
    "16",
    "bf16",
]


@dataclass(frozen=True)
class DataConfig:
    """数据集与 DataLoader 配置。

    数据来自 ``uv run ml-prepare-thucnews`` 生成的顺序字节 pack：根目录下需要有
    ``dataset_info.json`` 和 ``packs/{split}.bin`` + ``.offsets.npy`` + ``.labels.npy``。
    """

    root: str = "data/processed/THUCNews"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    max_length: int = 512
    # 固定补齐到 max_length：batch 形状恒定，torch.compile 无需动态形状、只需编译一次。
    # 关闭则按 batch 内最长序列补齐（变长），torch.compile 会自动改用动态形状。
    pad_to_max_length: bool = True
    batch_size: int = 64
    num_workers: int = 0


type ModelArchType = Literal["first_token", "byte_film"]


@dataclass(frozen=True)
class ModelConfig:
    """主模型配置。"""

    arch: ModelArchType = "first_token"
    vocab_size: int = 256
    num_classes: int = 14
    bias: bool = True
    use_padding_mask: bool = True


@dataclass(frozen=True)
class ArchConfig:
    """``byte_film`` 架构的超参。"""

    d_k: int = 64
    d_v: int = 64
    # mlp_k / mlp_v 的中间隐藏层宽度数组：``(64, 128)`` 表示 d -> 64 -> 128 -> d；
    # 空数组表示不加隐藏层（单层 d -> d，默认）。
    mlp_hidden: Sequence[int] = ()
    # 是否调制 V：True 时每轮用 FiLM 按当前特征调制静态 V 再逐位置 v_mix，
    # False 时每轮直接复用未调制的 V（省掉 film / v_mix 两组参数）。
    modulate_v: bool = True
    n_q: int = 16
    d_block: int = 768
    blocks_per_round: int = 2
    rounds: int = 3
    attention_dropout: float = 0.0
    film_identity_init: bool = True


@dataclass(frozen=True)
class OptimConfig:
    """优化器与损失函数配置。"""

    lr: float = 1e-3
    weight_decay: float = 0.0
    # 标签平滑强度：one-hot 目标变成 (1 - eps) / eps 在真实类与其余类之间的软目标，
    # 直接透传给 nn.CrossEntropyLoss(label_smoothing=...)。0.0 即关闭（纯 one-hot）。
    # 合理量级 0.05 ~ 0.1；超过 0 后训练 loss 下界抬高，绝对值不能和历史曲线直接对比。
    label_smoothing: float = 0.0


type SchedulerType = Literal["none", "cosine"]


@dataclass(frozen=True)
class SchedulerConfig:
    """学习率调度配置；按 epoch 更新（配合 ``[trainer] max_epochs``）。

    ``cosine`` 用的是 :class:`torch.optim.lr_scheduler.CosineAnnealingLR`，从当前 lr
    余弦降到 ``min_lr``；可选的线性 warmup 由 :class:`~torch.optim.lr_scheduler.SequentialLR`
    拼在余弦前面。余弦周期按 epoch 数算：``max_epochs - warmup_epochs``。
    """

    name: SchedulerType = "none"
    # 线性 warmup 的 epoch 数（0 表示不 warmup）。
    warmup_epochs: int = 0
    # warmup 起始学习率 = lr * warmup_start_factor，线性升到 lr 后交给余弦。
    warmup_start_factor: float = 0.1
    # 余弦终点学习率（绝对值的 eta_min）。
    min_lr: float = 0.0


@dataclass(frozen=True)
class LoggerConfig:
    """TensorBoard 日志配置。"""

    save_dir: str = "lightning_logs"
    name: str = "byte-first-token"
    log_graph: bool = False


@dataclass(frozen=True)
class TrainerConfig:
    """Lightning Trainer 配置。"""

    seed: int = 42
    max_epochs: int = 3
    accelerator: str = "auto"
    devices: int = 1
    precision: PrecisionType = "32-true"
    log_every_n_steps: int = 50
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0


@dataclass(frozen=True)
class CompileConfig:
    """torch.compile 及其 dynamo 开关配置。

    Note:
        动态形状不在这里配置，由 ``[data] pad_to_max_length`` 自动决定：变长补齐必须开
        动态形状（否则每个新 ``(B, L)`` 都会重编译，``fullgraph=True`` 下还会撞
        ``recompile_limit`` 直接失败）；固定补齐则保持静态形状最快。
    """

    enabled: bool = True
    fullgraph: bool = True
    compiled_autograd: bool = True
    capture_scalar_outputs: bool = True


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    arch: ArchConfig
    optim: OptimConfig
    scheduler: SchedulerConfig
    logger: LoggerConfig
    trainer: TrainerConfig
    compile: CompileConfig


def load_config(path: str | Path) -> Config:
    """读取 TOML 配置，缺失的小节 / 字段回落到 dataclass 默认值。

    Args:
        path: TOML 配置文件路径。

    Returns:
        解析后的 :class:`Config`。

    Raises:
        ValueError: 某个小节里出现未知字段（多半是拼写错误）。
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Config(
        data=_build(DataConfig, raw, "data"),
        model=_build(ModelConfig, raw, "model"),
        arch=_build(ArchConfig, raw, "arch"),
        optim=_build(OptimConfig, raw, "optim"),
        scheduler=_build(SchedulerConfig, raw, "scheduler"),
        logger=_build(LoggerConfig, raw, "logger"),
        trainer=_build(TrainerConfig, raw, "trainer"),
        compile=_build(CompileConfig, raw, "compile"),
    )


def _build(cls: type[Any], raw: dict[str, Any], section: str) -> Any:
    """用 ``raw[section]`` 构造 ``cls``，并拒绝未知字段。"""
    values = raw.get(section, {})
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"配置 [{section}] 中存在未知字段: {sorted(unknown)}")
    return cls(**values)
