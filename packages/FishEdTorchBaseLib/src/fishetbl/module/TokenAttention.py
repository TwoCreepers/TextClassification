from typing import override

from torch import Tensor, device, dtype
from torch.nn import functional as F

from ..base.BaseModule import BaseModule
from ..utils._assert import _assert


class TokenAttention(BaseModule[[Tensor, Tensor, Tensor, Tensor | None], Tensor]):
    """无投影的单头注意力：``n_q`` 個 query 共享同一份 K / V。

    形状约定（全部通道优先）：

    * ``q``：``(B, n_q * q_dim, 1)``——单 token 特征，通道维按 query 切开；
    * ``k``：``(B, k_dim, T)``；
    * ``v``：``(B, v_dim, T)``；
    * 输出：``(B, n_q * v_dim, 1)``。

    内部用 :func:`torch.nn.functional.scaled_dot_product_attention`，
    缩放因子为 ``1 / sqrt(k_dim)``。模块本身没有可学习参数（Q/K/V 的投影由
    调用方负责），因此可以放心在多个位置复用。

    Args:
        num_queries: query 数量 ``n_q``；``q`` 的通道数必须是它的整数倍。
        k_dim: K（以及每个 query）的维度。
        v_dim: V 的维度，允许与 ``k_dim`` 不等。
        dropout: 注意力 dropout，仅训练时生效。
    """

    def __init__(
        self,
        num_queries: int,
        k_dim: int,
        v_dim: int,
        *,
        dropout: float = 0.0,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        _assert(num_queries > 0, f"num_queries 必须为正整数，收到 {num_queries}。")
        _assert(k_dim > 0, f"k_dim 必须为正整数，收到 {k_dim}。")
        _assert(v_dim > 0, f"v_dim 必须为正整数，收到 {v_dim}。")
        _assert(0.0 <= dropout < 1.0, f"dropout 必须在 [0, 1) 内，收到 {dropout}。")
        self.num_queries = num_queries
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.dropout = dropout

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """对 ``K`` / ``V`` 做一次无 mask 的 softmax 注意力。

        Args:
            q: ``(B, n_q * k_dim, 1)`` 的 query 特征。
            k: ``(B, k_dim, T)`` 的静态记忆。
            v: ``(B, v_dim, T)`` 的（可能被逐位置调制过的）value。
            padding_mask: 可选 ``(B, T)`` 的布尔掩码，``True`` 表示该位置可见。
                某个样本全被遮住时会退化成不做遮挡（等价于不传 mask），避免 NaN。

        Returns:
            ``(B, n_q * v_dim, 1)`` 的注意力输出。

        Raises:
            AssertionError: 形状不符（通道数、序列长度、mask 形状）。
        """
        batch = q.shape[0]
        seq_len = k.shape[-1]
        _assert(
            lambda: q.shape == (batch, self.num_queries * self.k_dim, 1),
            "q 形状应为 (B, n_q*k_dim, 1)，收到",
            q.shape,
        )
        _assert(
            lambda: k.shape == (batch, self.k_dim, seq_len),
            "k 形状应为 (B, k_dim, T)，收到",
            k.shape,
        )
        _assert(
            lambda: v.shape == (batch, self.v_dim, seq_len),
            "v 形状应为 (B, v_dim, T)，收到",
            v.shape,
        )

        # (B, n_q * k_dim, 1) -> (B, n_q, 1, k_dim)，序列维只有 1
        query = q.reshape(batch, self.num_queries, 1, self.k_dim)
        key = k.transpose(1, 2).unsqueeze(1)  # (B, 1, T, k_dim)
        value = v.transpose(1, 2).unsqueeze(1)  # (B, 1, T, v_dim)

        attn_mask = None
        if padding_mask is not None:
            _assert(
                lambda: padding_mask.shape == (batch, seq_len),
                "padding_mask 形状应为 (B, T)，收到",
                padding_mask.shape,
            )
            # 保证每行至少一个可见位置；否则 softmax 全 -inf 会出 NaN
            visible = padding_mask | ~padding_mask.any(dim=-1, keepdim=True)
            attn_mask = visible[:, None, None, :]  # (B, 1, 1, T)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return out.reshape(batch, self.num_queries * self.v_dim, 1)
