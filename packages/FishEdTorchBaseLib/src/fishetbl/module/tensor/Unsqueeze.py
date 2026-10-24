from typing import override

from ...base.BaseModule import BaseTensorModule, Tensor


class Unsqueeze(BaseTensorModule):
    """在指定维度插入长度为 1 的维度，封装 :meth:`Tensor.unsqueeze`。

    等价于 ``x.unsqueeze(dim)``，用于补维以对齐形状
    （如给无 batch 维的张量补 batch 维、给通道维补维等）。

    Example:
        >>> import torch
        >>> m = Unsqueeze(0)
        >>> m(torch.randn(3)).shape
        torch.Size([1, 3])

    Note:
        ``dim`` 支持负数，按 :meth:`Tensor.unsqueeze` 的语义解释
        （相对插入后的维度计），允许范围为 ``[-(ndim+1), ndim]``。
    """

    @override
    def __init__(self, dim: int) -> None:
        """保存插入维度的位置。

        Args:
            dim: 新维度插入的位置，可为负数；插入后该维长度为 1。
        """
        super().__init__()
        self.dim = dim

    @override
    def forward(self, x: Tensor) -> Tensor:
        """在 ``dim`` 处插入长度为 1 的新维度。

        Args:
            x: 输入张量。

        Returns:
            插维后的张量，``ndim`` 比输入多 1。
        """
        super().__init__()
        return x.unsqueeze(self.dim)