from typing import TypeAlias

from torch import Tensor

from .BaseModule import BaseModule
from .ForceInitializeModule import ForceInitializeModule


class FModule[**TArgs, TResult](BaseModule[TArgs, TResult], ForceInitializeModule):
    """融合 :class:`BaseModule` 与 :class:`ForceInitializeModule` 的模块基类。

    本类整合了三项能力：

    * **buffer 辅助方法** —— 来自 :class:`BufferMixin`
      （例如 :meth:`BufferMixin.get_self_buffer`）；
    * **类型收窄** —— 来自 :class:`TypeModule`
      （``module(x)`` 的静态类型与 ``forward(x)`` 一致）；
    * **强制初始化** —— 来自 :class:`ForceInitializeModule`
      （``nn.Module.__init__`` 已在 ``__new__`` 阶段完成）。

    MRO 为
    ``FModule → BaseModule → BufferMixin → TypeModule
    → ForceInitializeModule → nn.Module → ABC → object``。

    子类应当：

    * 实现 :meth:`forward`；
    * 在 ``__init__`` 中 **不要** 调用 ``super().__init__()``；
    * 可选地使用 :meth:`BufferMixin.get_self_buffer` 等辅助方法。

    Type Parameters:
        TArgs: ``forward`` 的参数规格。
        TResult: ``forward`` 的返回类型。

    Example:
        >>> from torch import Tensor
        >>> class Double(FModule[[Tensor], Tensor]):
        ...     def __init__(self) -> None:
        ...         self.register_buffer("k", torch.tensor(2.0))
        ...     def forward(self, x: Tensor) -> Tensor:
        ...         return x * self.get_self_buffer("k")
        >>> y: Tensor = Double()(torch.randn(3))
    """

    ...  # 行为完全由 BaseModule 与 ForceInitializeModule 的 MRO 组合而来