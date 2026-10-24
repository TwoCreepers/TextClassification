from collections.abc import Callable
from typing import override

from ..base.BaseModule import BaseModule


class LambdaModule[**TArgs, TResult](BaseModule[TArgs, TResult]):
    """将任意可调用对象包装为 ``TModule`` 的适配模块。

    用于把普通函数 / lambda / 闭包直接接入模块流水线，
    无需为一次性逻辑单独定义子类。典型用途：

    * 在 ``nn.Sequential`` 之类的组合结构中插入无状态的函数变换；
    * 快速原型与实验；
    * 由配置或工厂函数动态生成变换步骤。

    Type Parameters:
        TArgs: 被包装函数的参数规格（``ParamSpec``）。
        TResult: 被包装函数的返回类型。

    Example:
        >>> import torch
        >>> m = LambdaModule(lambda x: x.clamp(min=0.0))
        >>> y = m(torch.randn(3))  # 等价于 torch.relu

    Note:
        * 被包装的函数通过普通属性 ``func`` 持有，因此 **不会** 出现在
          ``state_dict`` / ``named_modules`` 中，也不参与设备迁移
          （``.to()``）。若函数本身持有可训练参数或张量状态，应改用
          ``nn.Module`` 子类。
        * 由于函数不可序列化时整个模块也不可 ``pickle``，需要保存
          模型时应改用具名模块。
        * ``__init__`` 不调用 ``super().__init__()``，这是
          :class:`BaseModule` 的设计约定（初始化已在 ``__new__`` 中
          完成）。
    """

    @override
    def __init__(self, func: Callable[TArgs, TResult]) -> None:
        """保存被包装的可调用对象。

        Args:
            func: 任意可调用对象，其签名需与 ``TArgs -> TResult``
                一致。调用时其返回值将直接作为模块输出。
        """
        super().__init__()
        self.func = func

    @override
    def forward(self, *args: TArgs.args, **kwargs: TArgs.kwargs) -> TResult:
        """以前向参数调用被包装的函数。

        Args:
            *args: 转发给 ``self.func`` 的位置参数。
            **kwargs: 转发给 ``self.func`` 的关键字参数。

        Returns:
            ``self.func(*args, **kwargs)`` 的返回值。
        """
        return self.func(*args, **kwargs)
