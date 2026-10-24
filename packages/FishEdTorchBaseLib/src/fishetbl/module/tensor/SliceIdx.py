from typing import override

from ...base.BaseModule import BaseTensorModule, Tensor
from ...typing.alias import SliceIndex


class SliceIdx(BaseTensorModule):
    """张量索引 / 切片模块。

    把 Python 的索引表达式固化为模块，便于在模块化流水线中表达
    如“取所有 batch 的首通道”“取 ``[:, 0]``”等
    固定切片操作，并保持其可组合、可复用的形式。

    Example:
        >>> import torch
        >>> m = SliceIdx((slice(None), 0))   # 等价于 x[:, 0]
        >>> x = torch.randn(4, 3)
        >>> m(x).shape
        torch.Size([4])

    Note:
        * 索引以普通属性 ``idx`` 保存，不参与 ``state_dict``。
        * 若索引结果为标量张量（0 维），返回的仍是 ``Tensor``；
          若 ``idx`` 中含整数索引导致降维，输出维度相应减少。
    """

    @override
    def __init__(self, idx: SliceIndex) -> None:
        super().__init__()
        self.idx = idx

    @override
    def forward(self, x: Tensor) -> Tensor:
        """按保存的索引对输入张量取值。

        Args:
            x: 待索引的张量。

        Returns:
            ``x[*self.idx]`` 的结果，通常为原始张量的视图。
        """
        return x[*self.idx]