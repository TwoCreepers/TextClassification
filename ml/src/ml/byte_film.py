"""ByteFiLM-DenseNet：单 token 特征递归 + FiLM 调制 V + 纯 dense 连接。

结构（全部通道优先，latent 特征恒为 ``(B, C, 1)``）：

1. 静态记忆：byte 嵌入 -> ``mlp`` -> **仅 K** 加 RoPE，V 不过；
2. 初始聚合：可学习 latent query 对 ``(K, V)`` 做注意力，得到 ``(B, n_q*d_v, 1)``；
3. T 轮：若干 dense block 纯 cat 增宽 -> 生成 ``Q`` +（可选）FiLM 调制 V + 逐位置 ``v_mix``
   -> 注意力输出再 cat 回特征，宽度只增不减；每 ``compress_every_rounds`` 轮把主干宽度压到
   当前的 ``1 / compress_denominator``；
4. 分类头：``RMSNorm`` + ``1x1 Conv``。

宽度：``d0 = n_q * d_v``，每轮增宽 ``blocks_per_round * d_block + n_q * d_v``，
压缩点上除以 ``compress_denominator``（下限 ``d0``，最后一轮之后不压），压缩后的实际宽度
作为 ``d_final``。

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
    RotaryPositionEmbedding,
    TokenAttention,
    TokenEmbedding,
)

from .config import ArchConfig


def mlp(width_in: int, width_out: int, hidden: Sequence[int] = ()) -> NormAct:
    """逐位置的 ``RMSNorm -> MLP(GELU)``（不做残差 / 拼接）。

    Args:
        width_in: 输入通道数。
        width_out: 输出通道数。
        hidden: 中间隐藏层宽度数组；``(64, 128)`` 表示 ``width_in -> 64 -> 128 ->
            width_out``，空数组表示直接 ``width_in -> width_out`` 单层。
    """
    widths = (width_in, *hidden, width_out)
    return MlpNormAct1d(
        widths=widths,
        act="gelu",
        post_process=NormAct.IdentityPostModule,
        mapper_type="linear",
    )


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
    """

    def __init__(self, width_in: int, arch: ArchConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            dense_block(width_in + i * arch.d_block, arch.d_block) for i in range(arch.blocks_per_round)
        )
        width_out = width_in + arch.blocks_per_round * arch.d_block
        self.q_gen = mlp(width_out, arch.n_q * arch.d_k)
        # FiLM 调制 + 逐位置 v_mix：只在 modulate_v 打开时存在。
        self.film: FiLM | None = None
        self.v_mix: NormAct | None = None
        if arch.modulate_v:
            self.film = FiLM(width_out, arch.d_v, identity_init=arch.film_identity_init)
            self.v_mix = mlp(arch.d_v, arch.d_v)
        self.attn = TokenAttention(arch.n_q, arch.d_k, arch.d_v, dropout=arch.attention_dropout)

    def forward(self, x: Tensor, k: Tensor, v_base: Tensor, padding_mask: Tensor | None) -> Tensor:
        for block in self.blocks:
            x = block(x)
        query = self.q_gen(x)  # (B, n_q*d_k, 1)
        value = v_base  # 不调制时直接用静态 V
        film, v_mix = self.film, self.v_mix
        if film is not None and v_mix is not None:
            value = v_mix(film(v_base, x))  # (B, d_v, L)
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
        self.mlp_k = mlp(arch.d_k, arch.d_k, hidden=arch.mlp_hidden)
        self.mlp_v = mlp(arch.d_v, arch.d_v, hidden=arch.mlp_hidden)
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
        k = rearrange(k, "b c l -> b l c")
        k = rearrange(self.rope(k), "b l c -> b c l")  # 仅 K 带位置相位
        v_base = self.mlp_v(self.embed_v(input_ids))  # (B, d_v, L)

        query = self.latent_q.expand(input_ids.shape[0], -1, -1)  # (B, n_q*d_k, 1)
        x = self.attn(query, k, v_base, padding_mask)
        for iteration, compress in zip(self.iterations, self.compressions, strict=True):
            x = compress(iteration(x, k, v_base, padding_mask))
        return self.head(self.head_norm(x)).squeeze(-1)  # (B, num_classes)
