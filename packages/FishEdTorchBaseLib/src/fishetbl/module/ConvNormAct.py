from typing import override

from torch import Tensor

from ..base.BaseModule import BaseTensorModule
from ..typing.alias import TensorMap


class _ConvActBase(BaseTensorModule):
    """各 Conv 变体的公共构造：持有 ``conv`` / ``norm`` / ``act`` 三个模块。

    ``conv`` 是任意通道映射（如 :class:`torch.nn.Conv1d` / :class:`torch.nn.Conv2d` /
    ``byte_dense`` 里的 ``_Conv2dAs1d``），``norm`` / ``act`` 是现成模块。归一化与激活的
    相对位置由各子类的 ``forward`` 直接写死，不再有 ``norm_pos`` / ``act_pos`` /
    ``post_process`` 这类参数：

    * :class:`ConvNormActRes`：``Conv -> Norm -> Act -> 残差``，即 ``x + act(norm(conv(x)))``；
    * :class:`NormConvActRes`：``Norm -> Conv -> Act -> 残差``，即 ``x + act(conv(norm(x)))``；
    * :class:`ConvNormResAct`：``Conv -> Norm -> 残差 -> Act``，即 ``act(x + norm(conv(x)))``；
    * :class:`NormConvResAct`：``Norm -> Conv -> 残差 -> Act``，即 ``act(x + conv(norm(x)))``。

    Args:
        conv: 卷积模块，输入输出都应是通道优先的 ``(B, C, ...)``。
        norm: 归一化模块，通道数需与它在流水线中的位置一致。
        act: 激活模块。
    """

    def __init__(self, conv: TensorMap, norm: TensorMap, act: TensorMap) -> None:
        super().__init__()
        self.conv = conv
        self.norm = norm
        self.act = act


class ConvNormActRes(_ConvActBase):
    """``Conv -> Norm -> Act -> 残差``：``x + act(norm(conv(x)))``。

    残差加在激活之后（不是 ``act(x + ...)``）。要求 ``conv`` 不改变形状（同通道、同长度），
    否则残差加不回去；逐通道卷积 + ``padding="same"`` 天然满足。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.norm(self.conv(x)))


class NormConvActRes(_ConvActBase):
    """``Norm -> Conv -> Act -> 残差``：``x + act(conv(norm(x)))``。

    与 :class:`ConvNormActRes` 的区别只有归一化位置：这里先归一化再卷积，归一化模块的
    通道数取输入通道数；后者先卷积再归一化，通道数取输出通道数。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.act(self.conv(self.norm(x)))


class ConvNormResAct(_ConvActBase):
    """``Conv -> Norm -> 残差 -> Act``：``act(x + norm(conv(x)))``。

    残差加在激活之前（与 :class:`ConvNormActRes` 相反）。同样要求 ``conv`` 不改变形状。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.norm(self.conv(x)))


class NormConvResAct(_ConvActBase):
    """``Norm -> Conv -> 残差 -> Act``：``act(x + conv(norm(x)))``。

    残差加在激活之前（与 :class:`NormConvActRes` 相反）。同样要求 ``conv`` 不改变形状。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.conv(self.norm(x)))
