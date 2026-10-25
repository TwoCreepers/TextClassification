from collections.abc import Sequence
from typing import Literal, override

from torch.nn.functional import pad as torch_pad

from ..base.BaseModule import BaseTensorModule, Tensor
from ..module.Identity import Identity
from ..module.tensor.Padding import Padding
from ..typing.alias import PadAlign, PadMode
from ..utils._assert import _assert


def _build_pad_sequence(ndim: int, dims: Sequence[int], pad: Sequence[int]):
    """
    将 (dims, pad) 对齐的描述转换为 :func:`torch.nn.functional.pad` 所需的 pad 序列。

    Args:
        ndim: 张量的总维度数（例如 4D 张量 → ndim=4）
        dims: 需要填充的维度列表，如 (1, 2)
        pad:  与 dims 一一对齐的填充量，每两个数为 (before, after)。
              例如 pad=(1,1, 2,3) 表示
                  dim=dims[0] 前面填 1、后面填 1
                  dim=dims[1] 前面填 2、后面填 3

    Returns:
        F.pad 需要的 pad 序列（从最后一维开始，每两个一组，末尾冗余的 (0,0) 会被裁剪）

    例：
        ndim=4, dims=(1,2), pad=(1,1,2,3)
        → [0, 0, 2, 3, 1, 1]
        （对应 F.pad 的最后一维 pair 在前：dim3=(0,0), dim2=(2,3), dim1=(1,1)）

    Raises:
        AssertionError: 若 ``dims`` 中含有重复维度，或某个维度越界
            （合法范围 ``[-ndim, ndim - 1]``）。
    """

    # 归一化负数维度 + 建立 维度 → (before, after) 的映射
    pad_map = {}
    for i, d in enumerate(dims):
        _assert(
            lambda: -ndim <= d < ndim,
            f"维度 {d} 越界（张量维度数 {ndim}，合法范围 [-{ndim}, {ndim - 1}]）",
        )
        dn = d % ndim
        _assert(lambda: dn not in pad_map, f"重复的维度: {d}")
        pad_map[dn] = (pad[2 * i], pad[2 * i + 1])

    # 从最后一维往前逐维展开；F.pad 的语义就是从末维开始应用
    seq = []
    for d in range(ndim - 1, -1, -1):
        before, after = pad_map.get(d, (0, 0))
        seq.extend([before, after])

    # 裁剪末尾多余的 (0, 0)，因为 F.pad 会自动补齐缺失维度为 0
    while len(seq) >= 2 and seq[-1] == 0 and seq[-2] == 0:
        seq = seq[:-2]

    return seq


def _split_pad(total: int, align: PadAlign) -> tuple[int, int]:
    """把总填充量 ``total`` 按 ``align`` 拆成 (before, after)。

    Args:
        total: 需要填充的总长度（整数；``total > 0`` 表示填充，
            ``total < 0`` 表示裁剪）。
        align: 对齐方式，见 :func:`pad_to`。

    Returns:
        ``(before, after)``，满足 ``before + after == total``。
        当 ``total > 0`` 时正值表示新增的元素；当 ``total < 0``
        时负值表示被裁掉的元素。

        对 ``center_end`` / ``center_start``：绝对值大的一半视为
        "稍长"的一方。具体：

        * ``center_end``：``|after| >= |before|``（末尾稍长/裁剪略多）；
        * ``center_start``：``|before| >= |after|``（头部稍长/裁剪略多）。

        该约定对正负 ``total`` 一致成立。

    Raises:
        AssertionError: 若 ``align`` 取值非法。
    """
    if total == 0:
        return 0, 0
    if align == "end":
        return 0, total
    if align == "start":
        return total, 0
    # center_end / center_start：按绝对值拆，再乘回符号
    # 用绝对值避免 Python 负数向下取整导致 "稍长" 的方向被反转
    sign = 1 if total > 0 else -1
    mag = abs(total)
    before_mag = mag // 2
    after_mag = mag - before_mag  # 绝对值大的一半
    if align == "center_end":
        return sign * before_mag, sign * after_mag
    if align == "center_start":
        return sign * after_mag, sign * before_mag
    raise AssertionError(
        f"未知的 align: {align!r}，应为 'end' / 'start' / 'center_end' / 'center_start'",
    )


def _compute_pad_pairs(
    cur_shape: Sequence[int],
    target_shape: Sequence[int | None],
    align: PadAlign,
) -> list[tuple[int, int]]:
    _assert(
        lambda: len(cur_shape) == len(target_shape),
        f"shape 长度 ({len(target_shape)}) 必须与张量维度数 ({len(cur_shape)}) 一致",
    )
    pairs: list[tuple[int, int]] = []
    for i, (c, t) in enumerate(zip(cur_shape, target_shape)):
        if t is None or t == -1:
            pairs.append((0, 0))
            continue
        # 绑定到新变量：``t`` 是循环变量，类型窄化（已排除 None）不会带进 lambda。
        target = t
        _assert(
            lambda: target >= 0,
            f"第 {i} 维目标大小必须 >= 0（-1/None 表示不变），但得到 {target}",
        )
        pairs.append(_split_pad(target - c, align))
    return pairs


def pad(
    x: Tensor,
    pad: tuple[int, int] | Sequence[int],
    dims: int | Sequence[int] = -1,
    pad_mode: PadMode = "constant",
    value: float | None = None,
    ndim: int | None = None,
) -> Tensor:
    """在张量的指定维度上执行填充。

    与 :func:`torch.nn.functional.pad` 相比，本函数通过 ``dims`` 显式
    指定待填充的维度（不要求从末维开始连续），并统一接收
    ``(before, after)`` 成对的填充量。

    Args:
        x: 待填充的张量（至少 1 维）。
        pad: 与 ``dims`` 一一对应的填充量，每两个数为
            ``(before, after)``；若 ``dims`` 有 ``k`` 个元素，则
            ``pad`` 长度必须为 ``2 * k``。
        dims: 需要填充的维度；可为单个整数或整数序列，支持负数
            索引（按 ``x.ndim`` 归一化）。默认 ``-1``，表示仅填充
            最后一维。若为空序列则直接返回 ``x``。
        pad_mode: 填充模式，支持 ``"constant"``、``"reflect"``、
            ``"replicate"``、``"circular"``。默认 ``"constant"``。
        value: ``pad_mode="constant"`` 时使用的填充值；其他模式下被忽略。
        ndim: 张量的总维度数。仅当 ``x.ndim`` 在调用时已确定可
            留空；显式传入可避免重复推断 ``x.ndim``，但必须与
            ``x.ndim`` 一致。

    Returns:
        填充后的张量。其形状在 ``dims`` 指定的维度 ``d`` 上为
        ``x.size(d) + before + after``，其余维度保持不变。

    Raises:
        AssertionError: 若 ``len(pad) != 2 * len(dims)``、``x.ndim == 0``、
            显式 ``ndim`` 与 ``x.ndim`` 不一致，或 ``dims`` 越界/重复。

    Example:
        >>> x = torch.arange(6).reshape(2, 3)
        >>> pad(x, (1, 1), dims=0)  # 在第 0 维两侧各填 1
        tensor([[0, 0, 0],
                [0, 1, 2],
                [3, 4, 5],
                [0, 0, 0]])

    Note:
        若 ``dims`` 中维度在 ``F.pad`` 意义下不连续，内部会以
        ``(0, 0)`` 占位补齐，对最终结果无影响。
    """
    dims = (dims,) if isinstance(dims, int) else dims
    _assert(
        lambda: len(pad) == 2 * len(dims),
        f"pad 长度必须为 dims 长度的两倍, 但得到 len(pad)={len(pad)}, len(dims)={len(dims)}",
    )
    if not dims:
        return x
    if ndim is None:
        ndim = x.ndim
    else:
        _assert(
            lambda: ndim == x.ndim,
            f"显式 ndim ({ndim}) 与输入张量维度 ({x.ndim}) 不一致",
        )
    _assert(lambda: x.ndim > 0, "输入张量必须至少 1 维")
    pad_seq = _build_pad_sequence(ndim, dims, pad)
    return torch_pad(x, pad=pad_seq, mode=pad_mode, value=value)


class _Pad_Not_Known_Ndim(BaseTensorModule):
    """在未知张量总维度时执行填充的模块。

    当构造 :class:`Pad` 时未指定 ``ndim`` 时返回此模块。实际
    ``ndim`` 在 ``forward`` 时从输入张量的 ``x.ndim`` 获得，因此
    可在张量维度确定之前预先构建模块。

    Attributes:
        pad: 与 ``dims`` 一一对应的填充量，每两个数为
            ``(before, after)``。
        dims: 需要填充的维度（构造时已 ``tuple`` 化，防止外部修改）。
        pad_mode: 填充模式。
        value: ``mode="constant"`` 时的填充值；其他模式下被忽略。
    """

    pad: tuple[int, int] | Sequence[int]
    dims: tuple[int, ...]
    pad_mode: PadMode
    value: float | None

    @override
    def __init__(
        self,
        pad: tuple[int, int] | Sequence[int],
        dims: Sequence[int],
        pad_mode: PadMode = "constant",
        value: float | None = None,
    ) -> None:
        """保存填充参数。

        Args:
            pad: 与 ``dims`` 一一对应的填充量。
            dims: 需要填充的维度；构造时会被 ``tuple`` 化。
            pad_mode: 填充模式。
            value: ``mode="constant"`` 时的填充值。
        """
        super().__init__()
        self.pad = pad
        self.dims = tuple(dims)
        self.pad_mode = pad_mode
        self.value = value

    @override
    def forward(self, x: Tensor) -> Tensor:
        """对 ``x`` 执行填充。

        Args:
            x: 待填充张量。

        Returns:
            填充后的张量。

        Raises:
            AssertionError: 若 ``self.dims`` 越界或重复。
        """
        pad_seq = _build_pad_sequence(x.ndim, self.dims, self.pad)
        return torch_pad(x, pad=pad_seq, mode=self.pad_mode, value=self.value)


def Pad(
    pad: tuple[int, int] | Sequence[int],
    dims: int | Sequence[int] = -1,
    pad_mode: PadMode = "constant",
    value: float | None = None,
    ndim: int | None = None,
) -> BaseTensorModule:
    """构造填充张量的模块。

    Args:
        pad: 与 ``dims`` 一一对应的填充量，每两个数为
            ``(before, after)``；若 ``dims`` 有 ``k`` 个元素，则
            ``pad`` 长度必须为 ``2 * k``。
        dims: 需要填充的维度；可为单个整数或整数序列，支持负数
            索引。默认 ``-1``，表示仅填充最后一维。若为空序列
            则返回 :class:`Identity`（无操作）。
        pad_mode: 填充模式，支持 ``"constant"``、``"reflect"``、
            ``"replicate"``、``"circular"``。默认 ``"constant"``。
        value: ``mode="constant"`` 时的填充值；其他模式下被忽略。
        ndim: 张量的总维度数。为 ``None`` 时返回
            :class:`_Pad_Not_Known_Ndim`（延迟到 ``forward`` 推断）；
            否则在构造时预先计算 ``F.pad`` 序列，返回更高效的
            :class:`Padding` 模块。

    Returns:
        可调用的 :class:`BaseTensorModule`：

        * ``len(dims) == 0`` → :class:`Identity`；
        * ``ndim is None`` → :class:`_Pad_Not_Known_Ndim`；
        * 否则 → 预计算序列的 :class:`Padding`。

    Raises:
        AssertionError: 若 ``len(pad) != 2 * len(dims)``、``ndim <= 0``，
            或 ``dims`` 越界/重复（后者仅在 ``ndim`` 显式给出时于构造期检查；
            否则延迟到 ``forward``）。

    Example:
        >>> m = Pad((1, 1), dims=0)            # 延迟推断 ndim
        >>> m = Pad((1, 1), dims=0, ndim=2)    # 预计算 F.pad 序列
    """
    dims = (dims,) if isinstance(dims, int) else dims
    _assert(
        lambda: len(pad) == 2 * len(dims),
        f"pad 长度必须为 dims 长度的两倍, 但得到 len(pad)={len(pad)}, len(dims)={len(dims)}",
    )
    if not dims:
        return Identity()
    if ndim is None:
        return _Pad_Not_Known_Ndim(pad, dims, pad_mode, value)
    else:
        _assert(lambda: ndim > 0, "ndim 必须为正整数")
        pad_seq = _build_pad_sequence(ndim, dims, pad)
        return Padding(pad_seq, pad_mode, value)


def pad_to(
    x: Tensor,
    shape: Sequence[int | None],
    pad_mode: PadMode = "constant",
    align: PadAlign = "center_end",
    value: float | None = None,
) -> Tensor:
    """将张量各维填充到目标形状。

    与 :func:`pad` 需要显式给出 (before, after) 不同，本函数直接
    给出**目标 shape**，并按 ``align`` 分配两侧的填充量。

    Args:
        x: 待填充的张量。
        shape: 目标形状，长度须等于 ``x.ndim``。某元素为 ``-1``
            或 ``None`` 时表示该维保持不变。
        pad_mode: 填充模式，与 :func:`torch.nn.functional.pad`
            的 ``mode`` 完全一致，支持 ``"constant"``、
            ``"reflect"``、``"replicate"``、``"circular"``。
            默认 ``"constant"``。
        align: 对齐方式，四选一：

            * ``"end"``: padding 全部放在维度末尾（原数据头部对齐）；
            * ``"start"``: padding 全部放在维度头部（原数据末尾对齐）；
            * ``"center_end"``: 两端均分，不能均分时末尾稍长；
            * ``"center_start"``: 两端均分，不能均分时头部稍长。

            默认 ``"center_end"``（与多数视觉任务常用约定一致）。
        value: ``pad_mode="constant"`` 时使用的填充值；其他模式下被忽略。

    Returns:
        形状等于 ``shape``（对 ``-1``/``None`` 维沿用原大小）的
        新张量；若所有维度都无需填充，则直接返回 ``x``。

    Raises:
        AssertionError: 若 ``len(shape) != x.ndim``、某维目标大小
            为负（且不是 ``-1`` / ``None``），或 ``align`` 取值非法。

    Example:
        >>> x = torch.randn(1, 3, 200, 200)
        >>> y = pad_to(x, (-1, -1, 224, 224))                # 末尾略长
        >>> y = pad_to(x, (-1, -1, 224, 224), align="end")   # 全在末尾
        >>> y = pad_to(x, (-1, -1, 224, 224), align="start") # 全在头部
        >>> # 反射填充下需要谨慎：F.pad 要求 pad < 输入边长
        >>> y = pad_to(x, (-1, -1, 210, 210), pad_mode="reflect")

    Note:
        * 裁剪依赖 :func:`torch.nn.functional.pad` 的负值外推，请注意 torch 版本的具体实现。
        * 当某维需要裁剪（``total < 0``）时，``align`` 的语义与正填充
          一致地分配到两端（例如 ``"center_end"`` 表示头部裁剪得略少、
          末尾裁剪得略多），但裁剪是否被支持取决于 torch 版本。
    """
    pairs = _compute_pad_pairs(x.shape, shape, align)
    if all(b == 0 and a == 0 for b, a in pairs):
        return x

    dims = tuple(range(x.ndim))
    flat_pad: list[int] = []
    for b, a in pairs:
        flat_pad.extend([b, a])

    pad_seq = _build_pad_sequence(x.ndim, dims, flat_pad)
    return torch_pad(x, pad=pad_seq, mode=pad_mode, value=value)


class Pad_To(BaseTensorModule):
    """按目标形状填充张量的模块。

    与 :func:`pad_to` 语义完全一致，区别仅在于本类返回
    :class:`BaseTensorModule`，可嵌入 :class:`Sequential` 等结构；
    填充序列在每次 ``forward`` 时按输入 ``x.shape`` 动态生成。

    Attributes:
        shape: 目标形状；``-1`` / ``None`` 表示该维不变。
        pad_mode: 填充模式。
        align: 对齐方式，见 :func:`pad_to`。
        value: ``pad_mode="constant"`` 时的填充值。
    """

    shape: tuple[int | None, ...]
    pad_mode: PadMode
    align: PadAlign
    value: float | None

    @override
    def __init__(
        self,
        shape: Sequence[int | None],
        pad_mode: PadMode = "constant",
        align: PadAlign = "center_end",
        value: float | None = None,
    ) -> None:
        """保存目标形状与填充参数。

        Args:
            shape: 目标形状；``-1`` / ``None`` 表示该维不变。
            pad_mode: 填充模式，支持 ``"constant"``、``"reflect"``、
                ``"replicate"``、``"circular"``。
            align: 对齐方式，支持 ``"end"`` / ``"start"`` /
                ``"center_end"`` / ``"center_start"``。
            value: ``pad_mode="constant"`` 时使用的填充值。
        """
        super().__init__()
        self.shape = tuple(shape)
        self.pad_mode = pad_mode
        self.align = align
        self.value = value

    @override
    def forward(self, x: Tensor) -> Tensor:
        """对 ``x`` 执行前向填充。

        Args:
            x: 输入张量。其 ``ndim`` 必须等于 ``len(self.shape)``。

        Returns:
            形状等于 ``self.shape``（对 ``-1``/``None`` 维沿用原大小）
            的新张量。

        Raises:
            AssertionError: 若 ``len(self.shape) != x.ndim``，或某维
                目标大小为负（且不是 ``-1`` / ``None``）。
        """
        return pad_to(x, self.shape, self.pad_mode, self.align, self.value)


def pad_align(
    x: Tensor,
    frame_lens: int | Sequence[int],
    dims: int | Sequence[int] = -1,
    pad_mode: Literal["constant", "reflect", "replicate", "circular"] = "constant",
    align: PadAlign = "center_end",
    value: float | None = None,
) -> Tensor:
    """把张量指定维度填充到能被对应 ``frame_len`` 整除。

    对每个 ``(dim, frame_len)`` 对，计算该维当前大小 ``size``
    所需的最小填充量 ``total = (-size) % frame_len``，并按
    ``align`` 分配到两端。与 :func:`pad_to` 需要显式目标 shape
    不同，本函数的目标大小由 ``frame_len`` 动态决定。

    Args:
        x: 待填充的张量。
        frame_lens: 与 ``dims`` 一一对应的帧长。为 ``int`` 时等价于
            长度为 1 的序列；每个元素须为正整数。
        dims: 需要对齐的维度，可为单个整数或整数序列，支持负数
            索引；长度必须与 ``frame_lens`` 相同，且不允许重复。
            默认 ``-1``，即只对齐最后一维。
        pad_mode: 填充模式，与 ``torch.nn.functional.pad`` 的
            ``mode`` 完全一致，支持 ``"constant"``、``"reflect"``、
            ``"replicate"``、``"circular"``。默认 ``"constant"``。
        align: 对齐方式，四选一：

            * ``"end"``: padding 全部放在维度末尾（原数据头部对齐）；
            * ``"start"``: padding 全部放在维度头部（原数据末尾对齐）；
            * ``"center_end"``: 两端均分，不能均分时末尾稍长；
            * ``"center_start"``: 两端均分，不能均分时头部稍长。

            默认 ``"center_end"``。
        value: ``pad_mode="constant"`` 时使用的填充值；其他模式下被忽略。

    Returns:
        形状中 ``dims`` 指定的维度 ``d`` 变为
        ``ceil(x.size(d) / frame_len) * frame_len`` 的新张量，其中
        ``frame_len`` 是与 ``d`` 对应的帧长；若所有维均已整除，
        则直接返回 ``x``。

    Raises:
        AssertionError: 若 ``len(frame_lens) != len(dims)``、某个
            ``frame_len <= 0``、``dims`` 中出现重复维度或越界，或
            ``align`` 取值非法。

    Example:
        >>> x = torch.randn(2, 3, 100, 101)
        >>> # 100 → 112（pad 12），101 → 112（pad 11）
        >>> pad_align(x, (16, 16), dims=(2, 3)).shape
        torch.Size([2, 3, 112, 112])
        >>> # 全在末尾：原数据贴头部
        >>> pad_align(x, 16, dims=2, align="end").shape
        torch.Size([2, 3, 112, 101])
        >>> # 反射填充
        >>> pad_align(x, 8, dims=2, pad_mode="reflect").shape
        torch.Size([2, 3, 104, 101])
    """
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    frame_lens = (frame_lens,) if isinstance(frame_lens, int) else tuple(frame_lens)
    _assert(
        lambda: len(dims) == len(frame_lens),
        f"frame_lens 长度 ({len(frame_lens)}) 必须与 dims 长度 ({len(dims)}) 一致",
    )

    # 归一化负数维度并检查重复；同时把 dims 转成非负，供 _build_pad_sequence 使用
    dn_list: list[int] = []
    for d in dims:
        _assert(
            lambda: -x.ndim <= d < x.ndim,
            f"维度 {d} 越界（张量维度数 {x.ndim}，合法范围 [-{x.ndim}, {x.ndim - 1}]）",
        )
        dn = d % x.ndim
        _assert(lambda: dn not in dn_list, f"重复的维度: {d}")
        dn_list.append(dn)

    flat_pad: list[int] = []
    any_pad = False
    for dn, fl in zip(dn_list, frame_lens):
        _assert(lambda: fl > 0, f"frame_len 必须为正整数, 但得到 {fl}")
        size = x.shape[dn]
        total = (-size) % fl
        before, after = _split_pad(total, align)
        if total:
            any_pad = True
        flat_pad.extend([before, after])

    if not any_pad:
        return x

    pad_seq = _build_pad_sequence(x.ndim, dn_list, flat_pad)
    return torch_pad(x, pad=pad_seq, mode=pad_mode, value=value)


class Pad_Align(BaseTensorModule):
    """把张量指定维度填充到能被对应 ``frame_len`` 整除的模块。

    与 :func:`pad_align` 语义完全一致，区别仅在于本类返回
    :class:`BaseTensorModule`，可嵌入 :class:`Sequential` 等结构；
    填充序列在每次 ``forward`` 时按输入 ``x.shape`` 动态计算。

    Attributes:
        frame_lens: 与 ``dims`` 一一对应的帧长。
        dims: 需要对齐的维度。
        pad_mode: 填充模式。
        align: 对齐方式，见 :func:`pad_to`。
        value: ``pad_mode="constant"`` 时的填充值。
    """

    frame_lens: tuple[int, ...]
    dims: tuple[int, ...]
    pad_mode: Literal["constant", "reflect", "replicate", "circular"]
    align: PadAlign
    value: float | None

    @override
    def __init__(
        self,
        frame_lens: int | Sequence[int],
        dims: int | Sequence[int] = -1,
        pad_mode: Literal["constant", "reflect", "replicate", "circular"] = "constant",
        align: PadAlign = "center_end",
        value: float | None = None,
    ) -> None:
        """保存对齐参数。

        Args:
            frame_lens: 与 ``dims`` 一一对应的帧长；为 ``int`` 时
                等价于长度为 1 的序列。
            dims: 需要对齐的维度；为 ``int`` 时等价于长度为 1 的
                序列。
            pad_mode: 填充模式。
            align: 对齐方式。
            value: ``pad_mode="constant"`` 时的填充值。

        Raises:
            AssertionError: 若 ``len(frame_lens) != len(dims)``。
                帧长非正、维度越界/重复的检查延迟到 ``forward``。
        """
        super().__init__()
        self.frame_lens = (frame_lens,) if isinstance(frame_lens, int) else tuple(frame_lens)
        self.dims = (dims,) if isinstance(dims, int) else tuple(dims)
        _assert(
            len(self.frame_lens) == len(self.dims),
            f"frame_lens 长度 ({len(self.frame_lens)}) 必须与 dims 长度 ({len(self.dims)}) 一致",
        )
        self.pad_mode = pad_mode
        self.align = align
        self.value = value

    @override
    def forward(self, x: Tensor) -> Tensor:
        """对 ``x`` 执行前向对齐填充。

        Args:
            x: 输入张量。

        Returns:
            形状中 ``self.dims`` 指定的维度都被对应
            ``frame_len`` 整除的新张量。

        Raises:
            AssertionError: 若帧长非正，或 ``self.dims`` 越界/重复。
        """
        return pad_align(x, self.frame_lens, self.dims, self.pad_mode, self.align, self.value)
