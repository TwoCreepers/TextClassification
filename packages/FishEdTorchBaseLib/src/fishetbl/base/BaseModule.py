from typing import TypeAlias

from torch import Tensor

from ..mixin.BufferMixin import BufferMixin
from .TypeModule import TypeModule


class BaseModule[**TArgs, TResult](BufferMixin, TypeModule[TArgs, TResult]):
    """融合 :class:`BufferMixin` 与 :class:`TypeModule` 的模块基类。

    本类同时提供：

    * :meth:`BufferMixin.get_self_buffer` 等 buffer 辅助方法；
    * :class:`TypeModule` 的 ``__call__`` 类型收窄。

    本类 **不** 强制初始化：子类仍应在其 ``__init__`` 中调用
    ``super().__init__()``。如果需要强制初始化，请与
    :class:`ForceInitializeModule` 组合，或直接使用 :class:`FModule`。

    Type Parameters:
        TArgs: ``forward`` 的参数规格。
        TResult: ``forward`` 的返回类型。

    Example:
        >>> from torch import Tensor
        >>> class Double(BaseModule[[Tensor], Tensor]):
        ...     def __init__(self) -> None:
        ...         super().__init__()          # 需要手动调用
        ...         self.register_buffer("k", torch.tensor(2.0))
        ...     def forward(self, x: Tensor) -> Tensor:
        ...         return x * self.get_self_buffer("k")
    """

    ...  # 行为完全由 BufferMixin 与 TypeModule 的 MRO 组合而来


BaseTensorModule: TypeAlias = BaseModule[[Tensor], Tensor]
BaseTensorBranchesModule: TypeAlias = BaseModule[[Tensor], tuple[Tensor, ...]]
