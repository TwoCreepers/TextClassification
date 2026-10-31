"""ByteFiLM-DenseNet：单 token 特征递归 + FiLM 调制 V + 纯 dense 连接。

结构（全部通道优先，latent 特征恒为 ``(B, C, 1)``）：

1. 静态记忆：byte 嵌入 -> ``mlp`` ->（可选）残差卷积 -> **仅 K** 加 RoPE，V 不过；
2. 初始聚合：可学习 latent query 对 ``(K, V)`` 做注意力，得到 ``(B, n_q*d_v, 1)``；
3. T 轮：若干 dense block 纯 cat 增宽 -> 生成 ``Q`` +（可选）FiLM 调制 V + 逐位置 ``v_mix``
   -> 注意力输出再 cat 回特征，宽度只增不减；每 ``compress_every_rounds`` 轮把主干宽度压到
   当前的 ``1 / compress_denominator``；
4. 分类头：``Dropout`` + ``RMSNorm`` + ``1x1 Conv``。

Dropout：主干特征在送入 ``q_gen``、每个 ``dense_block``、``film``（作为条件）以及分类头
（``head_norm`` 之前）之前各有一次 dropout，四处由 ``[arch] q_gen_dropout`` /
``dense_block_dropout`` / ``film_dropout`` / ``head_dropout`` 分别配置，``0`` 表示关闭。

宽度：``d0 = n_q * d_v``，每轮增宽 ``blocks_per_round * d_block + n_q * d_v``，
压缩点上除以 ``compress_denominator``（下限 ``d0``，最后一轮之后不压），压缩后的实际宽度
作为 ``d_final``。

k / v 分支：``[arch] kv_conv_enabled = true`` 时在 ``mlp_k`` / ``mlp_v`` 之后各接一个
残差卷积 ``x + GELU(Norm(Conv(x)))``（核长 / 膨胀率由 ``kv_conv_kernel_size`` /
``kv_conv_dilation`` 配置，k、v 共用；k 侧在 RoPE 之前）。卷积内部按 ``(B, C, 1, L)`` 用
2d 卷积实现，为的是躲开 inductor 对 1d 卷积反向的静态形状固化（见 :class:`_Conv2dAs1d`）。
归一化位置由 ``[arch] pre_norm`` 统一控制：``true`` 时 ``mlp()`` 与这个卷积都用
``norm_pos="before map"``（pre-norm）。

Note:
    通道映射统一用 ``MlpNormAct1d(mapper_type="linear")``（``nn.Linear`` 而非 1x1
    ``Conv1d``）。两者数学等价，但 inductor 在动态形状下会把 1d 卷积反向的符号长度固化成
    具体值，导致变长 padding 时每个新长度都重编译（``fullgraph=True`` 下还会撞
    ``recompile_limit`` 直接失败）；matmul 没有这个限制。
"""

from collections.abc import Sequence

from einops import rearrange
from torch import Tensor, cat, empty, nn
from torch.nn import init

from fishetbl.base.BaseModule import BaseTensorModule
from fishetbl.module import (
    ChannelRmsNorm,
    FiLM,
    MlpNormAct1d,
    NormAct,
    NormPosType,
    RotaryPositionEmbedding,
    TokenAttention,
    TokenEmbedding,
)

from .config import ArchConfig


def mlp(width_in: int, width_out: int, hidden: Sequence[int] = (), norm_pos: NormPosType = "after map") -> NormAct:
    """逐位置的 ``Norm -> MLP(GELU)``（不做残差 / 拼接）。

    Args:
        width_in: 输入通道数。
        width_out: 输出通道数。
        hidden: 中间隐藏层宽度数组；``(64, 128)`` 表示 ``width_in -> 64 -> 128 ->
            width_out``，空数组表示直接 ``width_in -> width_out`` 单层。
        norm_pos: 归一化相对映射的位置，由 ``[arch] pre_norm`` 经 :func:`resolve_norm_pos` 决定。
    """
    widths = (width_in, *hidden, width_out)
    return MlpNormAct1d(
        widths=widths,
        norm_pos=norm_pos,
        act="gelu",
        post_process=NormAct.IdentityPostModule,
        mapper_type="linear",
    )


def resolve_norm_pos(arch: ArchConfig) -> NormPosType:
    """把 ``[arch] pre_norm`` 映射成 :class:`NormAct` 的 ``norm_pos``。

    ``mlp()`` 与 :func:`kv_residual_conv` 共用这一个取值，两者的归一化位置因此始终一致。
    """
    return "before map" if arch.pre_norm else "after map"


class _Conv2dAs1d(nn.Module):
    """用 2d 卷积实现序列（长度维）卷积：``(B, C, L)`` -> ``(B, C, 1, L)`` -> 卷积 -> squeeze。

    与同核长的 ``Conv1d`` 数学等价（权重 ``(C, C, ks)`` 对应 ``(C, C, 1, ks)``，实测逐元素
    误差 0），差别只在 ``torch.compile``：``Conv1d`` 的反向会把长度固化成静态值，20 个不同
    长度下第 9 个就撞 ``recompile_limit``（``fullgraph=True`` 直接失败）；这个 2d 实现
    20/20 通过、只编 3 张图。开销略高（``B=32, C=16, L=512`` 上约 +7% eager、+12% inductor，
    小通道数时主要来自 kernel 启动，真实模型会摊薄）。

    Args:
        channels: 输入输出通道数（残差要求相等）。
        kernel_size: 序列维上的核长；取奇数时 ``padding="same"`` 不必额外补零拷贝。
        dilation: 序列维上的膨胀率。
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding="same",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x.unsqueeze(-2)).squeeze(-2)


def kv_residual_conv(channels: int, arch: ArchConfig) -> NormAct | None:
    """``mlp_k`` / ``mlp_v`` 之后的残差卷积层：``x + GELU(Norm(Conv(x)))``。

    归一化与激活和 ``mlp()`` 统一（RMSNorm + GELU），归一化位置也跟 ``mlp()`` 一起由
    ``[arch] pre_norm`` 决定；``padding="same"`` 保证长度不变，残差才加得回去。
    卷积本身由 :class:`_Conv2dAs1d` 实现（内部走 2d 卷积）。

    Args:
        channels: 输入输出通道数（必须相等才能残差）；``d_k`` 或 ``d_v``。
        arch: 架构超参，用 ``kv_conv_enabled`` / ``kv_conv_kernel_size`` / ``kv_conv_dilation``。

    Returns:
        ``kv_conv_enabled`` 为 ``False`` 时返回 ``None``（不建参数，调用方直接跳过）。

    Raises:
        ValueError: 核长或膨胀率小于 1。

    Note:
        卷积刻意用 2d 实现而不是 ``Conv1d``：inductor 对 ``Conv1d`` 的反向会把序列长度
        固化成静态值（20 个不同长度里第 9 个就撞 ``recompile_limit``，``fullgraph=True``
        直接失败），``Conv2d`` 则全程符号形状（同样 20 个长度只编 3 张图、20/20 通过）。
        所以这一层在 ``pad_to_max_length = false`` + ``compile.enabled = true`` 下也能用，
        不必换 backend、也不必定长补齐；代价只是卷积本身慢一点，见 :class:`_Conv2dAs1d`。
    """
    if not arch.kv_conv_enabled:
        return None
    kernel_size = arch.kv_conv_kernel_size
    dilation = arch.kv_conv_dilation
    if kernel_size < 1:
        raise ValueError(f"[arch] kv_conv_kernel_size 必须 >= 1，收到 {kernel_size}")
    if dilation < 1:
        raise ValueError(f"[arch] kv_conv_dilation 必须 >= 1，收到 {dilation}")
    return NormAct(
        mapper=_Conv2dAs1d(channels, kernel_size, dilation),
        norm=ChannelRmsNorm(channels),
        norm_pos=resolve_norm_pos(arch),
        act=nn.GELU(),
        act_pos="after map",
        # 残差放在激活之后：x + GELU(conv(norm(x)))，而不是 GELU(conv(norm(x)) + x)。
        post_process=NormAct.ResidualPostModule,
        post_process_pos="finally",
    )


def dropout(p: float, name: str) -> nn.Dropout:
    """构造 ``p`` 概率的 :class:`torch.nn.Dropout`；``p = 0`` 时等价于直通。

    Args:
        p: 丢弃概率，取值 ``[0, 1)``。
        name: 配置字段名，只用于报错信息（``[arch] <name>``）。

    Raises:
        ValueError: ``p`` 不在 ``[0, 1)`` 内（``1`` 会把输入全部丢掉，直接拒绝）。
    """
    if not 0.0 <= p < 1.0:
        raise ValueError(f"[arch] {name} 必须在 [0, 1) 之间，收到 {p}")
    return nn.Dropout(p)


def dense_block(width_in: int, width_out: int) -> NormAct:
    """DenseBlock：``RMSNorm -> MLP(GELU)``，再与输入拼接（宽度只增不减）。"""
    return MlpNormAct1d(
        widths=(width_in, width_out),
        norm="rms",
        norm_pos="before map",
        act="gelu",
        post_process=NormAct.ConcatenatePostModule,
        post_process_pos="finally",
        mapper_type="linear",
    )


def compress_block(width_in: int, width_out: int) -> NormAct:
    """宽度压缩：``RMSNorm -> ChannelLinear``（无激活），把主干通道数压到 ``width_out``。"""
    return MlpNormAct1d(
        widths=(width_in, width_out),
        norm="rms",
        norm_pos="before map",
        act=None,
        post_process=NormAct.IdentityPostModule,
        mapper_type="linear",
    )


def compression_denominators(arch: ArchConfig) -> list[int]:
    """把 ``compress_denominator`` 展开成每个压缩点一个分母（按压缩先后顺序）。

    ``int`` 表示所有压缩点共用；数组按顺序取用。压缩点个数由 ``rounds`` 与
    ``compress_every_rounds`` 决定（最后一次之后不压），数组不够长时报错。
    分母必须 ``>= 1``；为 ``1`` 时该压缩点不改变宽度，模型里用恒等映射占位。
    """
    every = arch.compress_every_rounds
    if every <= 0:
        return []
    count = (arch.rounds - 1) // every
    spec = arch.compress_denominator
    values = [spec] * count if isinstance(spec, int) else list(spec)
    if len(values) < count:
        raise ValueError(
            f"[arch] compress_denominator 需要 {count} 个值"
            f"（rounds={arch.rounds}、compress_every_rounds={every}），"
            f"实际只给了 {len(values)} 个"
        )
    values = values[:count]
    if any(value < 1 for value in values):
        raise ValueError(f"[arch] compress_denominator 每个值都必须 >= 1（1 表示该点用恒等映射），收到 {values}")
    return values


class _Iteration(nn.Module):
    """一轮迭代：dense 增宽 -> 生成 Q（可选调制 V）-> 注意力回填。

    ``arch.modulate_v`` 为 ``False`` 时不建 ``film`` / ``v_mix``，每轮直接复用静态 V。
    主干特征在送入每个 dense block、``q_gen`` 与 ``film``（作为条件）之前各做一次 dropout，
    三者分别由 ``arch.dense_block_dropout`` / ``arch.q_gen_dropout`` / ``arch.film_dropout`` 控制。
    """

    def __init__(self, width_in: int, arch: ArchConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            dense_block(width_in + i * arch.d_block, arch.d_block) for i in range(arch.blocks_per_round)
        )
        # 与 blocks 一一对齐：每个 dense block 前各一个 dropout（同率、独立实例）。
        self.block_drops = nn.ModuleList(
            dropout(arch.dense_block_dropout, "dense_block_dropout") for _ in range(arch.blocks_per_round)
        )
        width_out = width_in + arch.blocks_per_round * arch.d_block
        self.q_gen = mlp(width_out, arch.n_q * arch.d_k, norm_pos=resolve_norm_pos(arch))
        self.q_gen_drop = dropout(arch.q_gen_dropout, "q_gen_dropout")
        # FiLM 调制 + 逐位置 v_mix：只在 modulate_v 打开时存在。
        self.film: FiLM | None = None
        self.v_mix: NormAct | None = None
        self.film_drop: nn.Dropout | None = None
        if arch.modulate_v:
            self.film = FiLM(width_out, arch.d_v, identity_init=arch.film_identity_init)
            self.v_mix = mlp(arch.d_v, arch.d_v, norm_pos=resolve_norm_pos(arch))
            self.film_drop = dropout(arch.film_dropout, "film_dropout")
        self.attn = TokenAttention(arch.n_q, arch.d_k, arch.d_v, dropout=arch.attention_dropout)

    def forward(self, x: Tensor, k: Tensor, v_base: Tensor, padding_mask: Tensor | None) -> Tensor:
        for block, block_drop in zip(self.blocks, self.block_drops, strict=True):
            x = block(block_drop(x))
        query = self.q_gen(self.q_gen_drop(x))  # (B, n_q*d_k, 1)
        value = v_base  # 不调制时直接用静态 V
        film, v_mix, film_drop = self.film, self.v_mix, self.film_drop
        if film is not None and v_mix is not None and film_drop is not None:
            value = v_mix(film(v_base, film_drop(x)))  # (B, d_v, L)
        attended = self.attn(query, k, value, padding_mask)  # (B, n_q*d_v, 1)
        return cat((x, attended), dim=1)


class ByteFiLMDenseNet(BaseTensorModule):
    """ByteFiLM-DenseNet 主体。

    Args:
        vocab_size: byte 词表大小（token id 即字节值，0 号兼作 padding）。
        num_classes: 类别数。
        arch: 架构超参，见 :class:`ml.config.ArchConfig`。

    Note:
        ``K`` 过 RoPE、``Q`` 不过：Q 来自可学习 latent，本身没有位置概念，
        位置信息只通过 K 的相位注入。
    """

    def __init__(self, vocab_size: int, num_classes: int, arch: ArchConfig) -> None:
        super().__init__()
        self.arch = arch

        self.embed_k = TokenEmbedding(vocab_size, arch.d_k)
        self.embed_v = TokenEmbedding(vocab_size, arch.d_v)
        self.mlp_k = mlp(arch.d_k, arch.d_k, hidden=arch.mlp_hidden, norm_pos=resolve_norm_pos(arch))
        self.mlp_v = mlp(arch.d_v, arch.d_v, hidden=arch.mlp_hidden, norm_pos=resolve_norm_pos(arch))
        # 残差卷积紧跟各自的 mlp（k 侧在 RoPE 之前），关掉时为 None 且不建参数。
        self.k_conv = kv_residual_conv(arch.d_k, arch)
        self.v_conv = kv_residual_conv(arch.d_v, arch)
        self.rope = RotaryPositionEmbedding(head_dim=arch.d_k)

        self.latent_q = nn.Parameter(empty(1, arch.n_q * arch.d_k, 1))
        init.normal_(self.latent_q, std=0.02)

        self.attn = TokenAttention(arch.n_q, arch.d_k, arch.d_v, dropout=arch.attention_dropout)

        width = arch.n_q * arch.d_v
        min_width = width  # 压缩下限：初始宽度，避免轮数多时塌缩得过窄
        denominators = compression_denominators(arch)
        iterations: list[nn.Module] = []
        # 与 iterations 一一对齐：压缩点放压缩层，否则放 Identity 占位。
        compressions: list[nn.Module] = []
        growth = arch.blocks_per_round * arch.d_block + arch.n_q * arch.d_v
        every = arch.compress_every_rounds
        for round_idx in range(1, arch.rounds + 1):
            iterations.append(_Iteration(width, arch))
            width += growth
            compressions.append(nn.Identity())
            # 每 compress_every_rounds 轮压缩一次，最后一次之后不压（压完直接进 head）。
            # 分母为 1（或被下限夹平）时 target == width，槽位保持 Identity 占位。
            if every > 0 and round_idx % every == 0 and round_idx < arch.rounds:
                denominator = denominators[round_idx // every - 1]
                target = max(min_width, width // denominator)
                if target < width:
                    compressions[-1] = compress_block(width, target)
                    width = target
        self.iterations = nn.ModuleList(iterations)
        self.compressions = nn.ModuleList(compressions)
        self.d_final = width

        # 分类头的 dropout 在 head_norm 之前（主干特征进入输出的最后一站）。
        self.head_drop = dropout(arch.head_dropout, "head_dropout")
        self.head_norm = ChannelRmsNorm(self.d_final)
        self.head = nn.Conv1d(self.d_final, num_classes, kernel_size=1)

    def forward(self, input_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """Args:
            input_ids: ``(B, L)`` 的 byte token。
            padding_mask: 可选 ``(B, L)`` 布尔掩码，``True`` 表示真实 token；
                ``None`` 表示不做遮挡。

        Returns:
            ``(B, num_classes)`` 的 logits。
        """
        k = self.mlp_k(self.embed_k(input_ids))  # (B, d_k, L)
        if self.k_conv is not None:
            k = self.k_conv(k)  # 残差卷积：形状不变，仍在 RoPE 之前
        k = rearrange(k, "b c l -> b l c")
        k = rearrange(self.rope(k), "b l c -> b c l")  # 仅 K 带位置相位
        v_base = self.mlp_v(self.embed_v(input_ids))  # (B, d_v, L)
        if self.v_conv is not None:
            v_base = self.v_conv(v_base)

        query = self.latent_q.expand(input_ids.shape[0], -1, -1)  # (B, n_q*d_k, 1)
        x = self.attn(query, k, v_base, padding_mask)
        for iteration, compress in zip(self.iterations, self.compressions, strict=True):
            x = compress(iteration(x, k, v_base, padding_mask))
        return self.head(self.head_norm(self.head_drop(x))).squeeze(-1)  # (B, num_classes)
