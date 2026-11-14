from typing import override

from torch import Tensor, cat, device, dtype, nn

from ..base.BaseModule import BaseTensorModule
from ..typing.alias import TensorMap
from ..utils._assert import _assert
from .ChannelLinear import ChannelLinear


class _MlpGeluBase(BaseTensorModule):
    """各 MLP 变体的公共构造：``ChannelLinear`` 堆叠 + 固定的 GELU + 一个归一化。

    ``in_c -> out_c`` 之间可以插入任意多个隐藏层（``*hidden``）：相邻两个全连接层之间固定
    插一个 ``nn.GELU``，最终激活也固定为 GELU，因此激活不再是参数、也不再有 ``inner_act``
    之类的复用/深拷贝逻辑。

    本类只负责建层与持有模块，具体顺序由各子类的 ``forward`` 直接写死（不再有位置参数）：

    * :class:`MlpNormGelu`：``Linear -> [GELU -> Linear]* -> Norm -> GELU``；
    * :class:`NormMlpGelu`：``Norm -> Linear -> [GELU -> Linear]* -> GELU``；
    * :class:`MlpNormResGelu`：``Linear -> [GELU -> Linear]* -> Norm -> 残差 -> GELU``；
    * :class:`NormMlpResGelu`：``Norm -> Linear -> [GELU -> Linear]* -> 残差 -> GELU``；
    * :class:`NormMlpGeluCat`：``Norm -> Linear -> [GELU -> Linear]* -> GELU -> cat``；
    * :class:`NormMlp`：``Norm -> Linear 堆叠``（没有最终激活）。

    Args:
        in_c: 输入通道数。
        out_c: 输出通道数。
        *hidden: 隐藏层通道数，按顺序依次插入；不传表示 ``in_c -> out_c`` 单层。
        norm: 归一化模块，调用方按所需通道数构造（如 :class:`ChannelRmsNorm`）。
        bias: 全连接层是否使用偏置。
        device: 模块参数所在设备。
        dtype: 模块参数数据类型。

    Note:
        通道优先：``(B, C, ...)`` 进、``(B, C', ...)`` 出，除通道维外其余维度原样保留；
        内部走 :class:`ChannelLinear`（``nn.Linear`` + ``movedim``），对动态长度友好。
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        *hidden: int,
        norm: TensorMap,
        bias: bool = True,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        widths = (in_c, *hidden, out_c)
        _assert(lambda: all(width > 0 for width in widths), f"通道数必须都是正整数，收到 {widths}。")
        layers: list[nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(ChannelLinear(widths[index], widths[index + 1], bias=bias, device=device, dtype=dtype))
            if index < len(widths) - 2:
                layers.append(nn.GELU())
        self.layers = nn.Sequential(*layers)
        self.norm = norm
        self.gelu = nn.GELU()


class MlpNormGelu(_MlpGeluBase):
    """``Linear -> [GELU -> Linear]* -> Norm -> GELU``（post-norm）。

    Example:
        >>> import torch
        >>> m = MlpNormGelu(64, 64, 128, norm=torch.nn.Identity())
        >>> m(torch.randn(2, 64, 32)).shape
        torch.Size([2, 64, 32])
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.gelu(self.norm(self.layers(x)))


class NormMlpGelu(_MlpGeluBase):
    """``Norm -> Linear -> [GELU -> Linear]* -> GELU``（pre-norm）。

    与 :class:`MlpNormGelu` 的区别只有归一化位置：这里先归一化再映射，归一化模块的
    通道数取 ``in_c``；后者先映射再归一化，通道数取 ``out_c``。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.gelu(self.layers(self.norm(x)))


class MlpNormResGelu(_MlpGeluBase):
    """``Linear -> [GELU -> Linear]* -> Norm -> 残差 -> GELU``：``gelu(x + norm(mlp(x)))``。

    残差加在激活之前；要求 ``in_c == out_c``，否则残差加不回去。归一化模块的通道数取
    ``out_c``。

    Example:
        >>> import torch
        >>> m = MlpNormResGelu(64, 64, 128, norm=torch.nn.Identity())
        >>> m(torch.randn(2, 64, 32)).shape
        torch.Size([2, 64, 32])
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.gelu(x + self.norm(self.layers(x)))


class NormMlpResGelu(_MlpGeluBase):
    """``Norm -> Linear -> [GELU -> Linear]* -> 残差 -> GELU``：``gelu(x + mlp(norm(x)))``。

    与 :class:`MlpNormResGelu` 的区别只有归一化位置：这里先归一化再映射（通道数取
    ``in_c``），后者先映射再归一化（通道数取 ``out_c``）。同样要求 ``in_c == out_c``。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.gelu(x + self.layers(self.norm(x)))


class NormMlpGeluCat(_MlpGeluBase):
    """``Norm -> Linear -> [GELU -> Linear]* -> GELU -> cat``（dense block）。

    最终的 GELU 结果沿通道维与输入拼接（新特征在前、输入在后），因此输出通道数是
    ``out_c + in_c``，宽度只增不减。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return cat((self.gelu(self.layers(self.norm(x))), x), dim=1)


class NormMlp(_MlpGeluBase):
    """``Norm -> Linear 堆叠``（没有最终激活），用于把通道数压回较窄的宽度。

    隐藏层之间仍固定插 GELU；byte_dense 的压缩层不加隐藏层，因此就是
    ``Norm -> ChannelLinear``。
    """

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(self.norm(x))
