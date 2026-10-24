from typing import override

from ..base.BaseModule import BaseModule


class Identity[T](BaseModule[[T], T]):
    """恒等模块：原样返回输入。

    常用于需要“占位”的模块组合场景（例如残差分支、可选变换、
    配置驱动的流水线中表示“不做任何处理”），从而避免在调用处
    写 ``if module is not None`` 分支。

    Type Parameters:
        T: 输入与输出类型。

    Example:
        >>> Identity()(42)
        42
        >>> import torch
        >>> x = torch.randn(3)
        >>> torch.equal(Identity()(x), x)
        True

    Note:
        该模块不持有任何参数或 buffer，也不复制输入，因此对返回的
        张量做原地修改会影响调用方持有的原张量。
    """

    @override
    def forward(self, value: T) -> T:
        """原样返回输入。

        Args:
            value: 任意输入值。

        Returns:
            与 ``value`` 完全相同的对象（同一引用）。
        """
        return value