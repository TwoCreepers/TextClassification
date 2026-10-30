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

# torch.set_float32_matmul_precision 接受的取值。
type Float32MatmulPrecisionType = Literal["highest", "high", "medium"]


@dataclass(frozen=True)
class DataConfig:
    """数据集与 DataLoader 配置。

    数据来自 ``just prep``（纯 Python）或 ``just prep-dedup``（.NET，先去重再划分）生成的顺序字节
    pack：根目录下需要有
    ``dataset_info.json`` 和 ``packs/{split}.bin`` + ``.offsets.npy`` + ``.labels.npy``。
    """

    root: str = "data/processed/THUCNews"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    # 每篇最多保留的 UTF-8 字节数：``THUCNewsDataset`` 读取时先截断到这里；``-1`` 表示不截断。
    max_length: int = 512
    # 是否把 batch 补齐到 ``max_length``（只影响补齐长度，不影响上面的截断）；
    # ``max_length = -1``（不截断）时序列长度没有上界，只能是变长补齐，这一项不生效。
    # 固定补齐时 batch 形状恒定，torch.compile 无需动态形状、只需编译一次；
    # 关闭则只补到 batch 内最长序列（变长），torch.compile 会自动改用动态形状。
    pad_to_max_length: bool = True
    batch_size: int = 64
    num_workers: int = 0
    # DataLoader 的 pinned memory：batch 先落到页锁定内存，H2D 拷贝更快；只在有加速器时有收益。
    pin_memory: bool = False


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
    # 宽度压缩：每 ``compress_every_rounds`` 轮把主干宽度压到当前的
    # ``1 / compress_denominator``；``compress_every_rounds = 0`` 表示不压缩。
    # 目标宽度下限是 ``n_q * d_v``（初始宽度），避免轮数多时塌缩得过窄；
    # 最后一轮之后不压缩（压完直接进分类头，没有收益）。
    compress_every_rounds: int = 0
    # 分母：int 表示所有压缩点都用同一个值；数组则按压缩点出现的顺序依次取用，
    # 长度必须 >= 压缩点个数（否则构造模型时报错），多出来的元素忽略。
    # 取值必须 >= 1；为 1 时该压缩点不压，占位成恒等映射。
    compress_denominator: int | Sequence[int] = 2
    n_q: int = 16
    d_block: int = 768
    blocks_per_round: int = 2
    rounds: int = 3
    # 主干特征的四处 dropout，互相独立（取值 [0, 1)，0 表示关闭）：
    #   q_gen_dropout       - 送入 q_gen 之前；
    #   dense_block_dropout - 送入每个 dense block 之前；
    #   film_dropout        - 作为条件特征送入 FiLM 之前（modulate_v = False 时不生效）；
    #   head_dropout        - 送入分类头之前（head_norm 之前）。
    q_gen_dropout: float = 0.0
    dense_block_dropout: float = 0.0
    film_dropout: float = 0.0
    head_dropout: float = 0.0
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
class EMAConfig:
    """EMA 权重平均配置（Lightning 自带的 ``EMAWeightAveraging`` callback）。

    开启后验证 / 测试用的是平均后的权重，checkpoint 里存的也是 EMA 权重。
    """

    enabled: bool = False
    # 每步的衰减系数：``avg = decay * avg + (1 - decay) * new``。
    decay: float = 0.999
    # 是否连 buffer（如 BN 的 running stats）一起平均；False 只平均参数。
    use_buffers: bool = False


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
    # torch.set_float32_matmul_precision 的取值（进程级全局设置，不是 Trainer 参数）：
    # high 让 fp32 矩阵乘走 TF32（Ampere+ 明显更快），highest 保持严格 fp32，
    # medium 更激进（bf16 双乘）。仅在 CUDA 上有效果。
    float32_matmul_precision: Float32MatmulPrecisionType = "high"
    log_every_n_steps: int = 50
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0


@dataclass(frozen=True)
class CompileConfig:
    """torch.compile 及其 dynamo 开关配置。

    Note:
        动态形状不在这里配置，由数据是否定长自动决定（``pad_to_max_length`` 且 ``max_length``
        会截断时为定长）：变长补齐必须开动态形状（否则每个新 ``(B, L)`` 都会重编译，
        ``fullgraph=True`` 下还会撞 ``recompile_limit`` 直接失败）；固定补齐则保持静态形状最快。
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
    ema: EMAConfig
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
        ema=_build(EMAConfig, raw, "ema"),
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
