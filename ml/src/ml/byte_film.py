"""ByteFiLM-DenseNet：单 token 特征递归 + FiLM 调制 V + 纯 dense 连接。

结构（全部通道优先，latent 特征恒为 ``(B, C, 1)``）：

1. 静态记忆：byte 嵌入 -> ``mlp`` -> **仅 K** 加 RoPE，V 不过；
2. 初始聚合：可学习 latent query 对 ``(K, V)`` 做注意力，得到 ``(B, n_q*d_v, 1)``；
3. T 轮：若干 dense block 纯 cat 增宽 -> 生成 ``Q`` + FiLM 调制 V + 逐位置 ``v_mix``
   -> 注意力输出再 cat 回特征，宽度只增不减；
4. 分类头：``RMSNorm`` + ``1x1 Conv``。

宽度：``d0 = n_q * d_v``，``d_final = d0 + rounds * (blocks_per_round * d_block + n_q * d_v)``。

Note:
    通道映射统一用 ``MlpNormAct1d(mapper_type="linear")``（``nn.Linear`` 而非 1x1
    ``Conv1d``）。两者数学等价，但 inductor 在动态形状下会把 1d 卷积反向的符号长度固化成
    具体值，导致变长 padding 时每个新长度都重编译（``fullgraph=True`` 下还会撞
    ``recompile_limit`` 直接失败）；matmul 没有这个限制。
"""

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


def mlp(width_in: int, width_out: int) -> NormAct:
    """逐位置的 ``RMSNorm -> MLP(GELU)``（不做残差 / 拼接）。"""
    return MlpNormAct1d(
        widths=(width_in, width_out),
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


class _Iteration(nn.Module):
    """一轮迭代：dense 增宽 -> 生成 Q + FiLM 调制 V -> 注意力回填。"""

    def __init__(self, width_in: int, arch: ArchConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            dense_block(width_in + i * arch.d_block, arch.d_block) for i in range(arch.blocks_per_round)
        )
        width_out = width_in + arch.blocks_per_round * arch.d_block
        self.q_gen = mlp(width_out, arch.n_q * arch.d_k)
        self.film = FiLM(width_out, arch.d_v, identity_init=arch.film_identity_init)
        self.v_mix = mlp(arch.d_v, arch.d_v)
        self.attn = TokenAttention(arch.n_q, arch.d_k, arch.d_v, dropout=arch.attention_dropout)

    def forward(self, x: Tensor, k: Tensor, v_base: Tensor, padding_mask: Tensor | None) -> Tensor:
        for block in self.blocks:
            x = block(x)
        query = self.q_gen(x)  # (B, n_q*d_k, 1)
        value = self.v_mix(self.film(v_base, x))  # (B, d_v, L)
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
        self.mlp_k = mlp(arch.d_k, arch.d_k)
        self.mlp_v = mlp(arch.d_v, arch.d_v)
        self.rope = RotaryPositionEmbedding(head_dim=arch.d_k)

        self.latent_q = nn.Parameter(empty(1, arch.n_q * arch.d_k, 1))
        init.normal_(self.latent_q, std=0.02)

        self.attn = TokenAttention(arch.n_q, arch.d_k, arch.d_v, dropout=arch.attention_dropout)

        width = arch.n_q * arch.d_v
        iterations = nn.ModuleList()
        for _ in range(arch.rounds):
            iterations.append(_Iteration(width, arch))
            width += arch.blocks_per_round * arch.d_block + arch.n_q * arch.d_v
        self.iterations = iterations
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
        for iteration in self.iterations:
            x = iteration(x, k, v_base, padding_mask)
        return self.head(self.head_norm(x)).squeeze(-1)  # (B, num_classes)
