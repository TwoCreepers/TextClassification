from abc import ABC, abstractmethod
from typing import override

import torch.nn as nn


class TypeModule[**TArgs, TResult](nn.Module, ABC):
    """带参数化类型标注的 ``nn.Module`` 抽象基类。

    标准 ``nn.Module.__call__`` 的返回类型是 ``Any``，会丢失
    ``forward`` 的返回类型信息。本类通过覆写 ``__call__``，把
    ``forward`` 的参数签名与返回类型参数化，使
    ``module(x)`` 在静态类型检查（mypy / pyright / Pylance 等）下
    拥有与 ``forward(x)`` 相同的类型。

    Type Parameters:
        TArgs: 参数规格（``ParamSpec``），描述 ``forward`` 接受的位置
            参数与关键字参数。
        TResult: ``forward`` 的返回类型。

    Note:
        本类只影响静态类型层面；运行时行为与 ``nn.Module`` 完全
        一致（hook、forward pre/post hook 等仍由父类负责）。

        本类不改变 ``nn.Module`` 的初始化语义：子类仍需正常调用
        ``super().__init__()``。若希望获得 buffer 辅助方法，可与
        :class:`BufferMixin` 组合成 :class:`BaseModule`；若希望
        强制初始化，可与 :class:`ForceInitializeModule` 组合成
        :class:`FModule`。
    """

    @override
    def __call__(self, *args: TArgs.args, **kwds: TArgs.kwargs) -> TResult:
        """执行前向计算。

        该覆写仅收窄 ``__call__`` 的类型签名
        （从 ``(*args, **kwargs) -> Any`` 收窄为
        ``(*TArgs.args, **TArgs.kwargs) -> TResult``），运行时直接
        委托给 :meth:`nn.Module.__call__`，因此 hook 等机制不受影响。

        Args:
            *args: 与 :meth:`forward` 相同的位置参数。
            **kwds: 与 :meth:`forward` 相同的关键字参数。

        Returns:
            与 :meth:`forward` 相同的返回值。
        """
        return super().__call__(*args, **kwds)

    @abstractmethod
    def forward(self, *args: TArgs.args, **kwds: TArgs.kwargs) -> TResult:
        """定义模块的前向计算逻辑，由子类实现。

        Args:
            *args: 前向计算的输入。
            **kwds: 前向计算的输入。

        Returns:
            前向计算的结果。
        """
        ...