from typing import override

from torch import Tensor, device, dtype, nn

from ..base.BaseModule import BaseTensorModule
from ..utils._assert import _assert


class TokenEmbedding(BaseTensorModule):
    """把整数 token 序列映射成通道优先的向量序列。

    ``(B, L)`` 的 token id -> ``(B, C, L)``（``channel_first=True``，默认）或
    ``(B, L, C)``。后者用于需要把特征维放在末维的模块（例如 RoPE）。

    Args:
        vocab_size: 词表大小。
        dim: 每个 token 的向量维度。
        channel_first: 为 ``True`` 时输出通道优先的 ``(B, C, L)``，
            否则输出 ``(B, L, C)``。
        device: 参数所在设备。
        dtype: 参数数据类型。
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        channel_first: bool = True,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        _assert(vocab_size > 0, f"vocab_size 必须为正整数，收到 {vocab_size}。")
        _assert(dim > 0, f"dim 必须为正整数，收到 {dim}。")
        self.vocab_size = vocab_size
        self.dim = dim
        self.channel_first = channel_first
        self.embedding = nn.Embedding(vocab_size, dim, device=device, dtype=dtype)

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Args:
            tokens: ``(B, L)`` 的整数 token id。

        Returns:
            ``(B, dim, L)`` 或 ``(B, L, dim)`` 的嵌入。
        """
        _assert(lambda: tokens.ndim == 2, "tokens 形状应为 (B, L)，收到", tokens.shape)
        embedded = self.embedding(tokens)
        return embedded.transpose(1, 2) if self.channel_first else embedded
