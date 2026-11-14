"""ByteDenseNet：单 token 特征递归 + 纯 dense 连接。

结构（全部通道优先，latent 特征恒为 ``(B, C, 1)``）：

1. 静态记忆：byte 嵌入 ->（可选）逐通道残差卷积 -> ``mlp`` -> **仅 K** 加 RoPE，V 不过；
2. 初始聚合：可学习 latent query 对 ``(K, V)`` 做注意力，得到 ``(B, n_q*d_v, 1)``；
3. T 轮：若干 dense block 纯 cat 增宽 -> 生成 ``Q`` -> 注意力输出再 cat 回特征，
   宽度只增不减；每 ``compress_every_rounds`` 轮把主干宽度压到当前的 ``1 / compress_denominator``；
4. 分类头：``Dropout`` + ``RMSNorm`` + ``1x1 Conv``。

Dropout：主干特征在送入 ``q_gen``、每个 ``dense_block`` 以及分类头（``head_norm`` 之前）
之前各有一次 dropout，三处由 ``[arch] q_gen_dropout`` / ``dense_block_dropout`` /
``head_dropout`` 分别配置，``0`` 表示关闭。

宽度：``d0 = n_q * d_v``，每轮增宽 ``blocks_per_round * d_block + n_q * d_v``，
压缩点上除以 ``compress_denominator``（下限 ``d0``，最后一轮之后不压），压缩后的实际宽度
作为 ``d_final``。

k / v 分支：``[arch] kv_conv_k_enabled`` / ``kv_conv_v_enabled`` 分别控制两侧在 byte 嵌入
之后、``mlp_k`` / ``mlp_v`` 之前接的逐通道卷积 ``GELU(Norm(Conv(x)))``
（``pre_norm = true`` 时是 ``GELU(Conv(Norm(x)))``；``kv_conv_k_norm`` /
``kv_conv_v_norm`` 关掉归一化后是 ``GELU(Conv(x))``；核长 / 膨胀率在 ``kv_conv_k_*`` /
``kv_conv_v_*`` 里各配一套，两侧互不影响）：卷积按通道独立（``groups = C``），只在长度
方向混合邻域、不混通道，参数量约为通道混合版的 ``1 / C``；k 侧因此仍在 RoPE 之前，卷积看到的
是不带位置相位的嵌入。卷积内部按 ``(B, C, 1, L)`` 用 2d 卷积实现，为的是躲开 inductor 对 1d
卷积反向的静态形状固化（见 :class:`_Conv2dAs1d`）。归一化位置由 ``[arch] pre_norm``
统一控制：``true`` 时先归一化再映射（``NormMlpGelu`` / ``NormConvAct``），
``false`` 时先映射再归一化（``MlpNormGelu`` / ``ConvNormAct``）；卷积那侧只有在
``kv_conv_*_norm = true`` 时才有归一化。``mlp_k`` / ``mlp_v`` / ``q_gen``
与这层卷积都不带残差。

Note:
    通道映射统一用 :class:`MlpNormGelu` / :class:`NormMlpGelu`（内部 ``ChannelLinear``，
    即 ``nn.Linear`` 而非 1x1 ``Conv1d``）。两者数学等价，但 inductor 在动态形状下会把 1d
    卷积反向的符号长度固化成具体值，导致变长 padding 时每个新长度都重编译
    （``fullgraph=True`` 下还会撞 ``recompile_limit`` 直接失败）；matmul 没有这个限制。
"""

from einops import rearrange
from torch import Tensor, cat, empty, nn
from torch.nn import init

from fishetbl.base.BaseModule import BaseTensorModule
from fishetbl.module import (
    ChannelRmsNorm,
    ConvNormAct,
    MlpNormGelu,
    MlpNormResGelu,
    NormConvAct,
    NormMlp,
    NormMlpGelu,
    NormMlpGeluCat,
    NormMlpResGelu,
    RotaryPositionEmbedding,
    TokenAttention,
    TokenEmbedding,
)

from .config import ArchConfig

type MlpGeluVariant = MlpNormGelu | NormMlpGelu | MlpNormResGelu | NormMlpResGelu


def mlp(
    width_in: int,
    width_out: int,
    *hidden: int,
    pre_norm: bool = False,
    residual: bool = False,
) -> MlpGeluVariant:
    """逐位置的 GELU MLP，归一化位置与残差由两个开关选类决定。

    Args:
        width_in: 输入通道数。
        width_out: 输出通道数。
        *hidden: 隐藏层通道数，按顺序插入；不传表示 ``width_in -> width_out`` 单层。
        pre_norm: ``True`` 先归一化再映射（``Norm...`` 开头），``False`` 先映射再归一化
            （``Mlp...`` 开头）。由 ``[arch] pre_norm`` 统一控制。
        residual: ``True`` 在激活前先加残差（``gelu(x + ...)``，要求 ``width_in == width_out``，
            对应 :class:`MlpNormResGelu` / :class:`NormMlpResGelu`）；``False`` 是纯激活
            （``gelu(norm(mlp(x)))``）。模型里 k / v / q_gen 目前都用 ``False``。

    Returns:
        :class:`MlpNormGelu` / :class:`NormMlpGelu` /
        :class:`MlpNormResGelu` / :class:`NormMlpResGelu` 之一。
    """
    norm = ChannelRmsNorm(width_in if pre_norm else width_out)
    if pre_norm and residual:
        cls = NormMlpResGelu
    elif pre_norm:
        cls = NormMlpGelu
    elif residual:
        cls = MlpNormResGelu
    else:
        cls = MlpNormGelu
    return cls(width_in, width_out, *hidden, norm=norm)


class _Conv2dAs1d(nn.Module):
    """用逐通道的 2d 卷积实现序列（长度维）卷积：``(B, C, L)`` -> ``(B, C, 1, L)`` -> 卷积 -> squeeze。

    逐通道（``groups=C``）：每个通道各自拿一条 ``(1, ks)`` 的核在长度方向滑动，通道之间不
    混合，权重 ``(C, 1, ks)`` 对应 ``(C, 1, 1, ks)``；与 ``groups=C`` 的同核长 ``Conv1d``
    数学等价（同一组权重下只是多一个长度为 1 的维度，计算本身没变）。差别只在
    ``torch.compile``：``Conv1d`` 的反向会把长度固化成静态值，20 个不同长度下第 9 个就撞
    ``recompile_limit``（``fullgraph=True`` 直接失败）；这个 2d 实现 20/20 通过、只编 3 张图。
    代价是比等价的 1d 实现略慢（多一个长度为 1 的维度；小通道数下主要是 kernel 启动开销，
    真实模型会摊薄）。

    Args:
        channels: 通道数，卷积的输入 / 输出通道都等于它（``groups=channels``：逐通道，不混通道；
            残差要求形状不变）。
        kernel_size: 序列维上的核长；取奇数时 ``padding="same"`` 不必额外补零拷贝。
        dilation: 序列维上的膨胀率。
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            groups=channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            padding="same",
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x.unsqueeze(-2)).squeeze(-2)


def kv_conv(
    channels: int,
    *,
    pre_norm: bool,
    norm: bool,
    enabled: bool,
    kernel_size: int,
    dilation: int,
    name: str,
) -> ConvNormAct | NormConvAct | None:
    """嵌入与 `mlp_k` / `mlp_v` 之间的逐通道卷积：`GELU(Norm(Conv(x)))`。

    接在 byte 嵌入之后、`mlp_k` / `mlp_v` 之前，不带残差。归一化可用
    `[arch] kv_conv_k_norm` / `kv_conv_v_norm` 关掉：关掉时这一层是 `GELU(Conv(x))`
    （norm 换成恒等映射，`pre_norm` 对这一层不再有影响）。卷积按通道独立
    （`groups = channels`），只在长度方向混合邻域、不混通道；激活与 `mlp()` 统一（GELU）。
    归一化位置跟 `mlp()` 一起由 `[arch] pre_norm` 决定（分别用 :class:`ConvNormAct` /
    :class:`NormConvAct`）；`padding="same"` 保证长度不变。卷积本身由
    :class:`_Conv2dAs1d` 实现（内部走 2d 卷积）。

    k / v 两侧的开关、是否归一化、核长、膨胀率都是分开传的（配置里是
    `kv_conv_k_*` / `kv_conv_v_*`）。

    Args:
        channels: 通道数，同时也是卷积的输入 / 输出通道（逐通道卷积）；`d_k` 或 `d_v`。
        pre_norm: 归一化位置；`True` 用 :class:`NormConvAct`（先归一化再卷积），`False` 用
            :class:`ConvNormAct`（先卷积再归一化）。`norm = false` 时这一项不生效。
        norm: 是否做归一化；`False` 时把归一化层换成恒等映射，即 `GELU(Conv(x))`。
        enabled: 这一侧是否启用；`False` 时返回 `None`（不建参数，调用方直接跳过）。
        kernel_size: 长度维上的核长，必须 `>= 1`。
        dilation: 长度维上的膨胀率，必须 `>= 1`。
        name: 配置前缀（`kv_conv_k` / `kv_conv_v`），只用于报错信息。

    Returns:
        `enabled` 为 `False` 时返回 `None`，否则返回卷积层。

    Raises:
        ValueError: 核长或膨胀率小于 1。

    Note:
        卷积刻意用 2d 实现而不是 `Conv1d`：inductor 对 `Conv1d` 的反向会把序列长度
        固化成静态值（20 个不同长度里第 9 个就撞 `recompile_limit`，`fullgraph=True`
        直接失败），`Conv2d` 则全程符号形状（同样 20 个长度只编 3 张图、20/20 通过）。
        所以这一层在 `pad_to_max_length = false` + `compile.enabled = true` 下也能用，
        不必换 backend、也不必定长补齐；代价只是卷积本身比 1d 实现慢一点，见 :class:`_Conv2dAs1d`。
    """
    if not enabled:
        return None
    if kernel_size < 1:
        raise ValueError(f"[arch] {name}_kernel_size 必须 >= 1，收到 {kernel_size}")
    if dilation < 1:
        raise ValueError(f"[arch] {name}_dilation 必须 >= 1，收到 {dilation}")
    cls = NormConvAct if pre_norm else ConvNormAct
    norm_layer = ChannelRmsNorm(channels) if norm else nn.Identity()
    return cls(_Conv2dAs1d(channels, kernel_size, dilation), norm_layer, nn.GELU())


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


def dense_block(width_in: int, width_out: int) -> NormMlpGeluCat:
    """DenseBlock：``RMSNorm -> ChannelLinear -> GELU``，再与输入拼接（宽度只增不减）。"""
    return NormMlpGeluCat(width_in, width_out, norm=ChannelRmsNorm(width_in))


def compress_block(width_in: int, width_out: int) -> NormMlp:
    """宽度压缩：``RMSNorm -> ChannelLinear``（无激活），把主干通道数压到 ``width_out``。"""
    return NormMlp(width_in, width_out, norm=ChannelRmsNorm(width_in))


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
    """一轮迭代：dense 增宽 -> 生成 Q -> 注意力回填。

    主干特征在送入每个 dense block、``q_gen`` 之前各做一次 dropout，二者分别由
    ``arch.dense_block_dropout`` / ``arch.q_gen_dropout`` 控制。V 每轮直接复用静态
    ``v_base``，不再做任何调制。
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
        self.q_gen = mlp(width_out, arch.n_q * arch.d_k, pre_norm=arch.pre_norm)
        self.q_gen_drop = dropout(arch.q_gen_dropout, "q_gen_dropout")
        self.attn = TokenAttention(arch.n_q, arch.d_k, arch.d_v, dropout=arch.attention_dropout)

    def forward(self, x: Tensor, k: Tensor, v_base: Tensor, padding_mask: Tensor | None) -> Tensor:
        for block, block_drop in zip(self.blocks, self.block_drops, strict=True):
            x = block(block_drop(x))
        query = self.q_gen(self.q_gen_drop(x))  # (B, n_q*d_k, 1)
        attended = self.attn(query, k, v_base, padding_mask)  # (B, n_q*d_v, 1)
        return cat((x, attended), dim=1)


class ByteDenseNet(BaseTensorModule):
    """ByteDenseNet 主体。

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
        # mlp 的隐藏层与残差卷积都是 k、v 各自一套配置，互不影响。
        self.mlp_k = mlp(arch.d_k, arch.d_k, *arch.mlp_k_hidden, pre_norm=arch.pre_norm)
        self.mlp_v = mlp(arch.d_v, arch.d_v, *arch.mlp_v_hidden, pre_norm=arch.pre_norm)
        # 残差卷积在嵌入之后、各自的 mlp 之前；某一侧关掉时那一侧为 None 且不建参数。
        self.k_conv = kv_conv(
            arch.d_k,
            pre_norm=arch.pre_norm,
            norm=arch.kv_conv_k_norm,
            enabled=arch.kv_conv_k_enabled,
            kernel_size=arch.kv_conv_k_kernel_size,
            dilation=arch.kv_conv_k_dilation,
            name="kv_conv_k",
        )
        self.v_conv = kv_conv(
            arch.d_v,
            pre_norm=arch.pre_norm,
            norm=arch.kv_conv_v_norm,
            enabled=arch.kv_conv_v_enabled,
            kernel_size=arch.kv_conv_v_kernel_size,
            dilation=arch.kv_conv_v_dilation,
            name="kv_conv_v",
        )
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
        k = self.embed_k(input_ids)  # (B, d_k, L)
        if self.k_conv is not None:
            k = self.k_conv(k)  # 逐通道残差卷积：形状不变，仍在 mlp_k 与 RoPE 之前
        k = self.mlp_k(k)
        k = rearrange(k, "b c l -> b l c")
        k = rearrange(self.rope(k), "b l c -> b c l")  # 仅 K 带位置相位
        v_base = self.embed_v(input_ids)  # (B, d_v, L)
        if self.v_conv is not None:
            v_base = self.v_conv(v_base)  # 同上，v 侧不过 RoPE
        v_base = self.mlp_v(v_base)

        query = self.latent_q.expand(input_ids.shape[0], -1, -1)  # (B, n_q*d_k, 1)
        x = self.attn(query, k, v_base, padding_mask)
        for iteration, compress in zip(self.iterations, self.compressions, strict=True):
            x = compress(iteration(x, k, v_base, padding_mask))
        return self.head(self.head_norm(self.head_drop(x))).squeeze(-1)  # (B, num_classes)
