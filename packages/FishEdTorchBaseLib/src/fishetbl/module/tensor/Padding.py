from collections.abc import Sequence
from typing import override

from torch.nn.functional import pad as torch_pad

from ...base.BaseModule import BaseTensorModule, Tensor
from ...typing.alias import PadMode


class Padding(BaseTensorModule):
    """张量填充模块，封装 :func:`torch.nn.functional.pad`。

    按 PyTorch 约定，``pad`` 从 **最后一维** 开始，成对给出
    每维 ``(前, 后)`` 的填充量；长度为 ``2 * k`` 的序列表示对最后
    ``k`` 维进行填充。

    Example:
        >>> import torch
        >>> m = Padding((1, 1), mode="constant", value=0.0)
        >>> m(torch.tensor([1.0, 2.0, 3.0]))
        tensor([0., 1., 2., 3., 0.])

    Note:
        各 ``mode`` 对维度和取值有额外约束（例如 ``reflect`` /
        ``replicate`` / ``circular`` 只支持部分填充方式，且
        ``value`` 仅在 ``constant`` 模式下生效），具体行为与异常
        以底层 :func:`torch.nn.functional.pad` 为准。
    """

    pad: tuple[int, int] | Sequence[int]
    pad_mode: PadMode
    value: float | None

    @override
    def __init__(
        self,
        pad: tuple[int, int] | Sequence[int],
        pad_mode: PadMode = "constant",
        value: float | None = None,
    ) -> None:
        """保存填充配置。

        Args:
            pad: 填充量。``(left, right)`` 表示仅对最后一维填充；
                更长的序列按 ``(last_dim_left, last_dim_right,
                second_last_dim_left, ...)`` 顺序依次作用于倒数各维。
            mode: 填充方式，取值 ``'constant'``、``'reflect'``、
                ``'replicate'``、``'circular'`` 之一。默认
                ``'constant'``。
            value: ``constant`` 模式下的填充值；``None`` 时按 PyTorch
                默认行为使用 ``0``。其他模式下该参数被忽略。
        """
        super().__init__()
        self.pad = pad
        self.pad_mode = pad_mode
        self.value = value

    @override
    def forward(self, x: Tensor) -> Tensor:
        """对输入张量执行填充。

        Args:
            x: 待填充的张量。

        Returns:
            填充后的新张量；形状在 ``pad`` 指定的各维上相应扩大。
        """
        return torch_pad(x, pad=self.pad, mode=self.pad_mode, value=self.value)
