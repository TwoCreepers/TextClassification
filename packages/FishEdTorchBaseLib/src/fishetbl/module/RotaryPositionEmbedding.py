from typing import override

from torch import Tensor, arange, cat, device, dtype, float32, int64, outer

from ..base.BaseModule import BaseTensorModule
from ..utils._assert import _assert


class RotaryPositionEmbedding(BaseTensorModule):
    """经典 RoPE（RoFormer 原版形式）：用位置相关的旋转作用在序列末维上。

    记最后一维 ``head_dim = d``，按前后两半配对旋转：

    .. math::

        \\mathrm{rotate\\_half}(x) = \\mathrm{cat}(-x_{d/2:}, x_{:d/2})

        x' = x \\cdot \\cos(\\theta) + \\mathrm{rotate\\_half}(x) \\cdot \\sin(\\theta)

    其中 ``theta[t, i] = t * base ** (-2i / d)``，``t`` 为位置。这是 RoFormer
    论文与官方实现里的版本（Llama 等模型沿用同一形式）；相邻维配对
    （GPT-NeoX 形式）是本实现的维度置换，不属于这里。

    模块不带可学习参数，``cos`` / ``sin`` 预计算到 ``max_seq_len`` 并作为
    非持久化 buffer 跟随设备迁移；位置超出缓存长度时按需计算，**不会** 在
    forward 里修改 buffer，因此可以安全地接入 ``torch.compile``。

    Args:
        head_dim: 旋转维度，即输入的最后一维；必须是偶数。
        base: 旋转频率基数，默认 ``10000.0``。
        max_seq_len: 预计算缓存的最大序列长度。位置超出时按需实时计算。
        device: 缓存所在设备。
        dtype: 缓存数据类型。建议保持 ``float32``（默认），低精度缓存会损失
            角度精度；计算结果的 dtype 始终跟随输入。

    Example:
        >>> import torch
        >>> rope = RotaryPositionEmbedding(head_dim=8)
        >>> x = torch.randn(2, 3, 5, 8)  # (B, H, T, D)
        >>> rope(x).shape
        torch.Size([2, 3, 5, 8])
        >>> bool(torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1)))
        True

    Note:
        * 经典 RoPE 是纯旋转，因此保持每一对维度的模长（单元测试可据此校验）。
        * 只用相对位置：``<RoPE(x, m), RoPE(y, n)>`` 只依赖 ``m - n``。
        * 增量推理时可传 ``offset``（KV cache 长度）或 ``position_ids``。
        * 参数校验统一走 :func:`fishetbl.utils._assert._assert`，可用
          ``_assert_mode("disable")`` 关闭，或由 ``_set_mode`` 全局调整；
          ONNX 导出期间断言会自动跳过。
    """

    # 显式声明 buffer 类型，避免类型检查器把 register_buffer 的字段推断成 Module。
    inv_freq: Tensor
    _cos_cached: Tensor
    _sin_cached: Tensor

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_seq_len: int = 4096,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        _assert(head_dim > 0 and head_dim % 2 == 0, f"head_dim 必须是正偶数，收到 {head_dim}。")
        _assert(max_seq_len > 0, f"max_seq_len 必须为正数，收到 {max_seq_len}。")

        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        cache_dtype = dtype if dtype is not None else float32
        inv_freq = base ** (-arange(0, head_dim, 2, device=device, dtype=cache_dtype) / head_dim)
        freqs = outer(arange(max_seq_len, device=device, dtype=cache_dtype), inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("_sin_cached", freqs.sin(), persistent=False)

    @override
    def forward(self, x: Tensor, position_ids: Tensor | None = None, offset: int = 0) -> Tensor:
        """对 ``x`` 沿最后一维做位置旋转。

        Args:
            x: 输入张量，形状 ``(..., T, head_dim)``；最后一维必须等于
                ``head_dim``，倒数第二维视为序列维。
            position_ids: 可选的位置序列，形状 ``(T,)``，用于打包序列 /
                自定义位置。给出时直接按需计算，不使用预计算缓存。
            offset: 未给 ``position_ids`` 时，位置从 ``offset`` 开始计；
                增量推理时可传 KV cache 已有的长度。

        Returns:
            形状与 ``x`` 相同的旋转结果，dtype 跟随 ``x``。

        Raises:
            AssertionError: 最后一维与 ``head_dim`` 不一致，或 ``position_ids``
                形状不是 ``(T,)``。校验通过 :func:`fishetbl.utils._assert._assert`
                完成，可用 ``_assert_mode("disable")`` 关闭（ONNX 导出时也会跳过）。
        """
        _assert(
            x.shape[-1] == self.head_dim,
            f"输入最后一维应为 head_dim={self.head_dim}，收到 {x.shape[-1]}。",
        )
        if position_ids is not None:
            _assert(
                position_ids.ndim == 1,
                f"position_ids 形状应为 (T,)，收到 {tuple(position_ids.shape)}。",
            )

        cos, sin = self._cos_sin(x.shape[-2], offset, position_ids)
        cos = cat((cos, cos), dim=-1).to(x.dtype)
        sin = cat((sin, sin), dim=-1).to(x.dtype)
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """前后两半交换并给前半取负：``cat((-x2, x1))``。"""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return cat((-x2, x1), dim=-1)

    def _cos_sin(self, seq_len: int, offset: int, position_ids: Tensor | None) -> tuple[Tensor, Tensor]:
        """取 ``(T, head_dim / 2)`` 的 cos / sin；缓存不够时实时计算。"""
        if position_ids is not None:
            positions = position_ids.to(device=self._cos_cached.device, dtype=int64)
            freqs = outer(positions.to(self.inv_freq.dtype), self.inv_freq)
            return freqs.cos(), freqs.sin()

        end = offset + seq_len
        if end <= self.max_seq_len:
            return self._cos_cached[offset:end], self._sin_cached[offset:end]

        positions = arange(offset, end, device=self._cos_cached.device, dtype=self.inv_freq.dtype)
        freqs = outer(positions, self.inv_freq)
        return freqs.cos(), freqs.sin()
