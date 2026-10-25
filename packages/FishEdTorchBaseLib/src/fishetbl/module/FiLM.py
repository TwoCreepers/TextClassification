from typing import cast, override

from torch import Tensor, device, dtype, nn, no_grad

from ..base.BaseModule import BaseModule
from ..utils._assert import _assert


class FiLM(BaseModule[[Tensor, Tensor], Tensor]):
    """FiLM：用条件向量逐通道调制信号。

    条件 ``cond`` 只取单个位置的特征 ``(B, cond_dim, 1)``，投影出
    ``(gamma, beta)`` 后广播到信号的其余维度：

    .. math::

        \\mathrm{FiLM}(x \\mid c) = \\gamma(c) \\odot x + \\beta(c)

    Args:
        cond_dim: 条件特征的通道数。
        feature_dim: 被调制信号的通道数。
        identity_init: 是否把投影初始化为恒等调制（``gamma = 1``、``beta = 0``），
            让 FiLM 在训练开始时等价于直通，避免乘性调制初期发散。
        device: 参数所在设备。
        dtype: 参数数据类型。

    Example:
        >>> import torch
        >>> film = FiLM(cond_dim=4, feature_dim=8)
        >>> x = torch.randn(2, 8, 5)
        >>> cond = torch.randn(2, 4, 1)
        >>> bool(torch.allclose(film(x, cond), x))   # identity_init 生效
        True
    """

    def __init__(
        self,
        cond_dim: int,
        feature_dim: int,
        *,
        identity_init: bool = True,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        _assert(cond_dim > 0, f"cond_dim 必须为正整数，收到 {cond_dim}。")
        _assert(feature_dim > 0, f"feature_dim 必须为正整数，收到 {feature_dim}。")
        self.cond_dim = cond_dim
        self.feature_dim = feature_dim
        self.proj = nn.Conv1d(cond_dim, 2 * feature_dim, kernel_size=1, device=device, dtype=dtype)
        if identity_init:
            bias = cast(Tensor, self.proj.bias)
            with no_grad():
                self.proj.weight.zero_()
                bias.zero_()
                bias[: self.feature_dim].fill_(1.0)  # gamma = 1, beta = 0

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Args:
            x: ``(B, feature_dim, ...)`` 的被调制信号，尾维任意。
            cond: ``(B, cond_dim, 1)`` 的单位置条件特征。

        Returns:
            与 ``x`` 同形状的调制结果。

        Raises:
            AssertionError: ``x`` 的通道数或 ``cond`` 的形状不符。
        """
        _assert(
            lambda: x.ndim >= 2 and x.shape[1] == self.feature_dim,
            "x 形状应为 (B, feature_dim, ...)，收到",
            x.shape,
        )
        _assert(
            lambda: cond.shape == (x.shape[0], self.cond_dim, 1),
            "cond 形状应为 (B, cond_dim, 1)，收到",
            cond.shape,
        )

        gamma, beta = self.proj(cond).chunk(2, dim=1)  # (B, feature_dim, 1) x2
        return x * gamma + beta
