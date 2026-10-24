from typing import Any, Self, override

import torch.nn as nn


class ForceInitializeModule(nn.Module):
    """强制初始化的 PyTorch 模块基类。

    标准 ``nn.Module`` 依赖子类在其 ``__init__`` 中调用
    ``super().__init__()`` 来创建 ``_parameters`` / ``_buffers`` /
    ``_modules`` 等注册表；一旦子类忘记调用，就会在注册参数时抛出
    ``AttributeError``。

    本类把 ``nn.Module.__init__`` 提前到 ``__new__`` 阶段执行，从而：

    * 子类即便省略 ``super().__init__()``，参数/子模块/buffer 的
      注册机制也一定可用；
    * 子类的 ``__init__`` 可以只写自己的初始化逻辑，无需关心
      ``nn.Module`` 的内部状态；
    * 避免子类因显式调用 ``super().__init__()`` 而造成注册表的
      重复初始化。

    子类通常应当：

    * 继承本类（或其子类，如 :class:`FModule`）；
    * 在自己的 ``__init__`` 中 **不要** 再调用 ``super().__init__()``。
    """

    @override
    def __new__(cls, *args: Any, **kwds: Any) -> Self:
        """创建实例并立即完成 ``nn.Module.__init__``。

        通过在本阶段调用 ``nn.Module.__init__(obj)``，
        让实例在返回给 ``__init__`` 之前就具备完整的注册表。

        Args:
            *args: 传递给 ``object.__new__`` 的位置参数（通常为空）。
            **kwds: 传递给 ``object.__new__`` 的关键字参数（通常为空）。

        Returns:
            已完成 ``nn.Module`` 初始化的实例。
        """
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        return obj

    @override
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """空实现，初始化已在 :meth:`__new__` 中强制完成。

        该实现用于阻止子类通过 ``super().__init__()`` 重复初始化
        ``nn.Module`` 的内部注册表。子类可以自由覆盖本方法，
        但仍然 **不应** 调用 ``super().__init__()``。

        Args:
            *args: 占位参数，忽略。
            **kwargs: 占位参数，忽略。
        """
        ...  # 初始化已在 __new__ 中强制进行，防止子类重复初始化
