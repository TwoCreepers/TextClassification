"""ml 项目的配置结构：全部来自 ``ml/config/config.toml``。"""

import tomllib
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
    """数据集与 DataLoader 配置。"""

    dataset_name: str = "seamew/THUCNews"
    # datasets 5.x 已不再支持 dataset script，THUCNews 需要读取 Hub 自动转换的 parquet 分支。
    revision: str | None = "refs/convert/parquet"
    text_column: str = "text"
    label_column: str = "label"
    train_split: str = "train"
    val_split: str = "validation"
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
    num_classes: int = 10
    bias: bool = True
    use_padding_mask: bool = True


@dataclass(frozen=True)
class ArchConfig:
    """``byte_film`` 架构的超参。"""

    d_k: int = 64
    d_v: int = 64
    n_q: int = 16
    d_block: int = 768
    blocks_per_round: int = 2
    rounds: int = 3
    attention_dropout: float = 0.0
    film_identity_init: bool = True


@dataclass(frozen=True)
class OptimConfig:
    """优化器配置。"""

    lr: float = 1e-3
    weight_decay: float = 0.0


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
