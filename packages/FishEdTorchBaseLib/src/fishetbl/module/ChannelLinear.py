from typing import override

from torch import Tensor, device, dtype, nn

from ..base.BaseModule import BaseTensorModule
from ..utils._assert import _assert


class ChannelLinear(BaseTensorModule):
    """按通道做全连接：``(B, C_in, ...) -> (B, C_out, ...)``。

    数学上等价于 ``kernel_size=1`` 的 :class:`torch.nn.Conv1d`（或 Conv2d），但内部走
    :class:`torch.nn.Linear` 的矩阵乘。单独提供它的原因是 ``torch.compile`` 的**动态形状**：
    inductor 的 1d 卷积反向 lowering 需要静态 stride，会把符号化的长度**固化**成具体值，
    于是变长 padding 下每个新长度都要重编译（``fullgraph=True`` 时还会撞
    ``recompile_limit`` 直接失败）；matmul 没有这个限制。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        bias: 是否使用偏置。
        device: 参数所在设备。
        dtype: 参数数据类型。

    Example:
        >>> import torch
        >>> m = ChannelLinear(4, 8)
        >>> m(torch.randn(2, 4, 5)).shape
        torch.Size([2, 8, 5])
        >>> conv = torch.nn.Conv1d(4, 8, kernel_size=1)
        >>> with torch.no_grad():
        ...     _ = conv.weight.copy_(m.linear.weight[:, :, None])
        ...     _ = conv.bias.copy_(m.linear.bias)
        >>> x = torch.randn(2, 4, 5)
        >>> bool(torch.allclose(conv(x), m(x)))
        True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias, device=device, dtype=dtype)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: 通道优先的特征，通道维为第 1 维（如 ``(B, C, L)``）；
                其它维度（含长度 ``L``）可以是符号化的。

        Returns:
            与 ``x`` 形状相同、仅第 1 维变成 ``out_channels`` 的特征。输出是**连续**的
            （与 ``Conv1d`` 的输出布局一致）：``movedim`` 只是视图，直接把它交给下游
            （如 ``scaled_dot_product_attention``）时，inductor 在动态形状下可能选到
            layout 不兼容的 kernel。

        Raises:
            AssertionError: 输入少于 2 维。
        """
        _assert(lambda: x.ndim >= 2, "ChannelLinear 输入形状应为 (B, C, ...)，收到", x.shape)
        return self.linear(x.movedim(1, -1)).movedim(-1, 1).contiguous()
