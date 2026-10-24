import torch.nn as nn
from torch import Tensor

from ..utils._Sentinel import SENTINEL


class BufferMixin(nn.Module):
    """提供 buffer 访问辅助方法的 mixin。

    本类只添加方法，不改变 ``nn.Module`` 的初始化语义；
    需要强制初始化的场景请与 :class:`ForceInitializeModule` 组合，
    或直接使用 :class:`FModule`。

    子类仍应在其 ``__init__`` 中调用 ``super().__init__()``。
    """

    def get_self_buffer(self, name: str) -> Tensor | None:
        """获取当前模块自身的 buffer，且不向子模块递归查找。

        与 :meth:`nn.Module.get_buffer` 不同，本方法只检查
        ``self._buffers``，不会沿名字中的 ``.`` 路径下钻到子模块，
        因此不会因为不同模块存在同名 buffer 而出现歧义。

        Args:
            name: buffer 名称。

        Returns:
            对应的 buffer 张量。若该 buffer 是通过
            ``register_buffer(name, None)`` 注册的，则返回 ``None``。

        Raises:
            ValueError: 当前模块未注册名为 ``name`` 的 buffer。
        """
        buf = self._buffers.get(name, SENTINEL)
        if buf is SENTINEL:
            raise ValueError(f"buffer {name} is not found.")
        return buf
