from collections.abc import Generator
from typing import Self, overload


class NumberedContainerMixin[T]:
    """把若干元素以 "0"、"1"、… 为 key 存为实例属性，
    并提供 len / iter / 整数索引 / 切片 的定长容器协议。

    之所以用字符串数字当 key，是为了配合 BaseModule
    这类会在 __setattr__ 里登记子模块的基类。

    约定：子类的构造函数接受 *items，并按相同顺序重建自身，
    切片才能返回同类容器。
    """

    _item_len: int

    def __init__(self, *items: T) -> None:
        super().__init__()
        self._init_items(*items)

    def _init_items(self, *items: T) -> None:
        self._item_len = len(items)
        for i in range(self._item_len):
            setattr(self, str(i), items[i])

    def __len__(self) -> int:
        return self._item_len

    def __iter__(self) -> Generator[T]:
        for i in range(self._item_len):
            yield getattr(self, str(i))

    @overload
    def __getitem__(self, key: int) -> T: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def __getitem__(self, key: int | slice) -> T | Self:
        if isinstance(key, int):
            idx = key + self._item_len if key < 0 else key
            if idx < 0 or idx >= self._item_len:
                raise IndexError(f"{type(self).__name__} index out of range: {key} (length {self._item_len})")
            return getattr(self, str(idx))

        start, stop, step = key.indices(self._item_len)
        return type(self)(*(getattr(self, str(i)) for i in range(start, stop, step)))
