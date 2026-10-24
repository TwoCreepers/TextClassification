from typing import Sequence, override

from ..base.BaseModule import BaseTensorModule, Tensor
from ..typing.alias import SliceIndex
from ..utils._assert import _assert
from ..module.Identity import Identity
from ..module.tensor.Unsqueeze import Unsqueeze
from ..module.tensor.SliceIdx import SliceIdx


def _build_idx(ndim: int, dims: Sequence[int]) -> SliceIndex:
    """将需要插入 ``None`` 的维度位置转换为索引元组。

    返回的索引元组可直接用于 ``x[idx]``，语义上等价于按 ``dims``
    的顺序依次调用 ``x.unsqueeze(d)``：后一个 ``dims`` 元素以
    已插入之后的形状为基准解释。本函数正确处理重复 dim、
    负数 dim、乱序 dim 三种情况。

    Args:
        ndim: 输入张量 ``x`` 的维度数。
        dims: 新维度的插入位置（在依次插入后的新形状下的索引）。
            支持负数索引；负数按 ``d + (当前维度数 + 1)`` 归一化，
            与 ``torch.Tensor.unsqueeze`` 的语义一致。

    Returns:
        长度等于 ``ndim + len(dims)`` 的索引元组，每个元素为
        ``None``（在该处插入新维度）或 ``slice(None)``（保留
        原始维度）。

    Raises:
        AssertionError: 若某次插入时维度越界（合法范围
            ``[-(当前维度数 + 1), 当前维度数]``）。

    Example:
        >>> _build_idx(2, (0, 2))
        (None, slice(None), None, slice(None))
    """
    # 精确模拟连续 unsqueeze（正确处理重复 dim、负 dim、乱序 dim）
    slots: list[None | slice] = [slice(None)] * ndim
    n = ndim
    for d in dims:
        _assert(
            -(n + 1) <= d <= n,
            f"维度 {d} 越界（当前维度数 {n}，插入后维度数 {n + 1}，"
            f"合法范围 [-{n + 1}, {n}]）",
        )
        if d < 0:
            d += n + 1
        slots.insert(d, None)
        n += 1
    return tuple(slots)


def expand_dims_to_start(x: Tensor, num: int) -> Tensor:
    """在张量最前面插入 ``num`` 个新维度。

    等价于 ``x[(None,) * num + (...,)]``，也等价于连续 ``num`` 次
    调用 ``x.unsqueeze(0)``。

    Args:
        x: 输入张量。
        num: 需要在开头插入的新维度数；须为非负整数。

    Returns:
        形状为 ``(1,) * num + x.shape`` 的张量。

    Raises:
        AssertionError: 若 ``num < 0``。

    Example:
        >>> x = torch.arange(6).reshape(2, 3)
        >>> expand_dims_to_start(x, 1).shape
        torch.Size([1, 2, 3])
    """
    _assert(num >= 0, f"num 必须为非负整数, 但得到 {num}")
    slice_args = [None] * num + [...]
    return x[*slice_args]


class Expand_Dims_To_Start(BaseTensorModule):
    """在张量最前面插入 ``num`` 个新维度的模块。

    构造时预先计算切片参数，``forward`` 时执行切片。

    Args:
        num: 需要在开头插入的新维度数。
    """

    @override
    def __init__(self, num: int):
        """保存切片参数。

        Args:
            num: 需要在开头插入的新维度数。

        Raises:
            AssertionError: 若 ``num < 0``。
        """
        super().__init__()
        _assert(num >= 0, f"num 必须为非负整数, 但得到 {num}")
        self.slice_args = [None] * num + [...]

    @override
    def forward(self, x: Tensor) -> Tensor:
        """执行前向计算。

        Args:
            x: 输入张量。

        Returns:
            形状为 ``(1,) * num + x.shape`` 的张量。
        """
        return x[*self.slice_args]


def expand_dims_to_end(x: Tensor, num: int) -> Tensor:
    """在张量末尾插入 ``num`` 个新维度。

    等价于 ``x[(...,) + (None,) * num]``，也等价于连续 ``num`` 次
    调用 ``x.unsqueeze(-1)``。

    Args:
        x: 输入张量。
        num: 需要在末尾插入的新维度数；须为非负整数。

    Returns:
        形状为 ``x.shape + (1,) * num`` 的张量。

    Raises:
        AssertionError: 若 ``num < 0``。

    Example:
        >>> x = torch.arange(6).reshape(2, 3)
        >>> expand_dims_to_end(x, 1).shape
        torch.Size([2, 3, 1])
    """
    _assert(num >= 0, f"num 必须为非负整数, 但得到 {num}")
    slice_args = [...] + [None] * num
    return x[*slice_args]


class Expand_Dims_To_End(BaseTensorModule):
    """在张量末尾插入 ``num`` 个新维度的模块。

    构造时预先计算切片参数，``forward`` 时执行切片。

    Args:
        num: 需要在末尾插入的新维度数。
    """

    @override
    def __init__(self, num: int):
        """保存切片参数。

        Args:
            num: 需要在末尾插入的新维度数。

        Raises:
            AssertionError: 若 ``num < 0``。
        """
        super().__init__()
        _assert(num >= 0, f"num 必须为非负整数, 但得到 {num}")
        self.slice_args = [...] + [None] * num

    @override
    def forward(self, x: Tensor) -> Tensor:
        """执行前向计算。

        Args:
            x: 输入张量。

        Returns:
            形状为 ``x.shape + (1,) * num`` 的张量。
        """
        return x[*self.slice_args]


def expand_dims(x: Tensor, dims: int | Sequence[int], ndim: int | None = None) -> Tensor:
    """在张量的指定位置插入新维度。

    与 :func:`torch.unsqueeze` 相比，本函数允许一次指定多个插入
    位置（``dims`` 为序列时）。插入过程按 ``dims`` 的顺序依次
    应用，后一个元素以已插入之后的形状为基准，与连续调用
    ``unsqueeze`` 完全一致。

    Args:
        x: 输入张量。
        dims: 新维度的插入位置。为 ``int`` 时退化为
            ``x.unsqueeze(dims)``；为 ``Sequence[int]`` 时按顺序
            连续插入。支持负数索引。
        ndim: 插入前 ``x`` 的维度数。为 ``None`` 时取 ``x.ndim``；
            显式传入时若与 ``x.ndim`` 不一致会报错（包括
            ``len(dims) == 1`` 的分支）。

    Returns:
        在 ``dims`` 指定位置各插入一个长度 1 的新维度后的张量。

    Raises:
        AssertionError: 若显式 ``ndim`` 与 ``x.ndim`` 不一致，或
            某次插入时维度越界。

    Example:
        >>> x = torch.arange(6).reshape(2, 3)
        >>> expand_dims(x, 0).shape
        torch.Size([1, 2, 3])
        >>> expand_dims(x, (0, 2)).shape
        torch.Size([1, 2, 1, 3])

    Note:
        由于 ``dims`` 按顺序应用，``(0, 2)`` 与 ``(2, 0)`` 的结果
        形状不同：前者得到 ``(1, A, 1, B)``，后者得到
        ``(1, A, B, 1)``。
    """
    dims = (dims,) if isinstance(dims, int) else dims
    dims_len = len(dims)
    if dims_len == 0:
        return x
    elif dims_len == 1:
        if ndim is not None:
            _assert(
                ndim == x.ndim,
                f"显式 ndim ({ndim}) 与输入张量维度 ({x.ndim}) 不一致",
            )
        return x.unsqueeze(dims[0])

    if ndim is None:
        ndim = x.ndim
    else:
        _assert(
            ndim == x.ndim,
            f"显式 ndim ({ndim}) 与输入张量维度 ({x.ndim}) 不一致",
        )
    return x[_build_idx(ndim, dims)]


class _Expand_Dims_Not_Known_Ndim(BaseTensorModule):
    """在未知张量总维度时插入新维度的模块。

    当构造 :class:`Expand_Dims` 时未指定 ``ndim`` 时返回此模块。
    实际 ``ndim`` 在 ``forward`` 时从输入张量的 ``x.ndim`` 获得，
    因此可在张量维度确定之前预先构建模块。

    Args:
        dims: 新维度的插入位置（构造时已 ``tuple`` 化）。
    """

    @override
    def __init__(self, dims: Sequence[int]) -> None:
        """保存插入位置。

        Args:
            dims: 新维度的插入位置；构造时会被 ``tuple`` 化。
        """
        super().__init__()
        self.dims = tuple(dims)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """执行前向计算。

        Args:
            x: 输入张量。

        Returns:
            在 ``dims`` 指定位置各插入一个长度 1 的新维度后的张量。

        Raises:
            AssertionError: 若某次插入时维度越界。
        """
        return x[*_build_idx(x.ndim, self.dims)]


def Expand_Dims(dims: int | Sequence[int], ndim: int | None = None) -> BaseTensorModule:
    """构造插入新维度的模块。

    Args:
        dims: 新维度的插入位置。为 ``int`` 时退化为
            :class:`Unsqueeze`；为 ``Sequence[int]`` 时按顺序
            连续插入。支持负数索引。
        ndim: 插入前 ``x`` 的维度数。若显式传入，构造期即做合法性
            校验（单维度时校验范围 ``[-ndim-1, ndim]``；多维度时
            按连续插入后的索引校验）。为 ``None`` 时返回延迟推断
            ``x.ndim`` 的模块（或 ``Unsqueeze`` / ``Identity``）。

    Returns:
        可调用的 :class:`BaseTensorModule`：

        * ``len(dims) == 0`` → :class:`Identity`；
        * ``len(dims) == 1`` → :class:`Unsqueeze`；
        * ``ndim is None`` → :class:`_Expand_Dims_Not_Known_Ndim`；
        * 否则 → 预计算索引的 :class:`SliceIdx`。

    Raises:
        AssertionError: 当 ``ndim`` 显式给出时，若某次插入的维度越界。

    Example:
        >>> x = torch.arange(6).reshape(2, 3)
        >>> Expand_Dims(0)(x).shape              # Unsqueeze
        torch.Size([1, 2, 3])
        >>> Expand_Dims((0, 2))(x).shape         # 延迟推断 ndim
        torch.Size([1, 2, 1, 3])
        >>> Expand_Dims((0, 2), ndim=2)(x).shape # 预计算索引
        torch.Size([1, 2, 1, 3])
    """
    dims = (dims,) if isinstance(dims, int) else dims
    dims_len = len(dims)
    if dims_len == 0:
        return Identity()
    elif dims_len == 1:
        if ndim is not None:
            d = dims[0]
            _assert(
                -(ndim + 1) <= d <= ndim,
                f"维度 {d} 越界（插入后维度数 {ndim + 1}，"
                f"合法范围 [-{ndim + 1}, {ndim}]）",
            )
        return Unsqueeze(dims[0])

    if ndim is None:
        return _Expand_Dims_Not_Known_Ndim(dims)
    else:
        return SliceIdx(_build_idx(ndim, dims))