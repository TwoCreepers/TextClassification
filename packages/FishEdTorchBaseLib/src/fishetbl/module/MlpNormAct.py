from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal

from torch import device, dtype, nn
from torch.nn.utils.parametrizations import weight_norm

from ..typing.alias import ActStrType, Map, TensorMap
from ..utils._act import build_activation
from .ChannelRmsNorm import ChannelRmsNorm
from .LambdaModule import LambdaModule
from .NormAct import ActPosType, NormAct, NormPosType, PostModule, PostModuleBuilder, PostProcessPosType

type NormStrType = Literal["bn", "in", "rms"]


def MlpNormAct1d(
    in_c: int,
    out_c: int,
    hidden_c: int | Sequence[int] | None = None,
    inner_act: TensorMap | ActStrType | None = None,
    bias: bool | None = None,
    is_norm_weight: bool = False,
    init_mapper_weight: Map[[nn.Conv1d], nn.Conv1d] | None = None,
    norm: TensorMap | NormStrType | None = "rms",
    norm_pos: NormPosType = "after map",
    act: TensorMap | ActStrType | None = "relu",
    act_pos: ActPosType = "after map",
    post_process: PostModule | PostModuleBuilder = NormAct.ResidualPostModule,
    post_process_pos: PostProcessPosType = "before act or invalid",
    device: device | None = None,
    dtype: dtype | None = None,
) -> NormAct:
    """构造在序列特征通道维上做全连接的 :class:`NormAct`，即 ``NormAct`` 版本的 1d MLP。

    输入输出形状均为 ``(N, C, L)``：每个 ``l`` 位置共享同一组权重，相当于对通道维
    做 ``Linear``（这里用 ``kernel_size=1`` 的卷积实现，与 :func:`MlpNormAct2d`
    保持同一套参数与权重处理方式）。

    网络主体（mapper）为 ``in_c -> *hidden_c -> out_c`` 的堆叠，相邻两层之间按
    ``inner_act`` 插入激活；norm / act / post_process 的位置语义与
    :func:`MlpNormAct2d` 完全一致。

    Args:
        in_c: 输入通道数。
        out_c: 输出通道数。
        hidden_c: 隐层宽度。``int`` 表示一个隐层，``Sequence[int]`` 表示多个隐层，
            ``None`` 表示没有隐层（退化为单个通道变换）。
        inner_act: 隐层之间的激活。``None`` 时沿用 ``act``（当 ``act`` 为字符串时取同名
            激活，否则不插激活）；字符串按每个隐层宽度构建独立实例，模块实例则会被
            复用（后续位置深拷贝，保证各自持有独立参数）。
        bias: 是否使用偏置。``None`` 时跟随 ``norm``：使用 ``"bn"`` / ``"in"`` 时
            默认 ``False``，其余情况默认 ``True``。
        is_norm_weight: 是否对所有全连接层施加 ``weight_norm``。
        init_mapper_weight: 对每个全连接层调用的权重初始化函数。
        norm: 归一化。字符串 ``"bn"`` / ``"in"`` / ``"rms"`` 按需构建，
            也可直接传入模块，``None`` 表示不做归一化。
        norm_pos: 归一化相对 mapper 的位置。
        act: 激活，字符串或模块，``None`` 表示不使用激活。
        act_pos: 激活相对 mapper 的位置。
        post_process: 残差 / 拼接等后处理模块或其构造函数。
        post_process_pos: 后处理相对激活的位置。
        device: 模块参数所在设备。
        dtype: 模块参数数据类型。

    Returns:
        由上述组件组装而成的 :class:`NormAct` 模块。

    Example:
        >>> import torch
        >>> m = MlpNormAct1d(64, 64, hidden_c=256)
        >>> m(torch.randn(2, 64, 32)).shape
        torch.Size([2, 64, 32])
    """
    return _mlp_norm_act(
        1,
        in_c,
        out_c,
        hidden_c,
        inner_act,
        bias,
        is_norm_weight,
        init_mapper_weight,
        norm,
        norm_pos,
        act,
        act_pos,
        post_process,
        post_process_pos,
        device,
        dtype,
    )


def MlpNormAct2d(
    in_c: int,
    out_c: int,
    hidden_c: int | Sequence[int] | None = None,
    inner_act: TensorMap | ActStrType | None = None,
    bias: bool | None = None,
    is_norm_weight: bool = False,
    init_mapper_weight: Map[[nn.Conv2d], nn.Conv2d] | None = None,
    norm: TensorMap | NormStrType | None = "rms",
    norm_pos: NormPosType = "after map",
    act: TensorMap | ActStrType | None = "relu",
    act_pos: ActPosType = "after map",
    post_process: PostModule | PostModuleBuilder = NormAct.ResidualPostModule,
    post_process_pos: PostProcessPosType = "before act or invalid",
    device: device | None = None,
    dtype: dtype | None = None,
) -> NormAct:
    """构造在特征图通道维上做全连接的 :class:`NormAct`，即 ``NormAct`` 版本的 2d MLP。

    输入输出形状均为 ``(N, C, H, W)``：每个 ``(h, w)`` 位置共享同一组权重，
    相当于对通道维做 ``Linear``（这里用 ``kernel_size=1`` 的卷积实现，
    与 :func:`ConvNormAct2d` 保持同一套参数与权重处理方式）。

    网络主体（mapper）为 ``in_c -> *hidden_c -> out_c`` 的堆叠，
    相邻两层之间按 ``inner_act`` 插入激活；norm / act / post_process
    的位置语义与 :func:`ConvNormAct2d` 完全一致。

    Args:
        in_c: 输入通道数。
        out_c: 输出通道数。
        hidden_c: 隐层宽度。``int`` 表示一个隐层，``Sequence[int]`` 表示多个隐层，
            ``None`` 表示没有隐层（退化为单个通道变换）。
        inner_act: 隐层之间的激活。``None`` 时沿用 ``act``（当 ``act`` 为字符串时取同名
            激活，否则不插激活）；字符串按每个隐层宽度构建独立实例，模块实例则会被
            复用（后续位置深拷贝，保证各自持有独立参数）。
        bias: 是否使用偏置。``None`` 时跟随 ``norm``：使用 ``"bn"`` / ``"in"`` 时
            默认 ``False``，其余情况默认 ``True``。
        is_norm_weight: 是否对所有全连接层施加 ``weight_norm``。
        init_mapper_weight: 对每个全连接层调用的权重初始化函数。
        norm: 归一化。字符串 ``"bn"`` / ``"in"`` / ``"rms"`` 按需构建，
            也可直接传入模块，``None`` 表示不做归一化。
        norm_pos: 归一化相对 mapper 的位置。
        act: 激活，字符串或模块，``None`` 表示不使用激活。
        act_pos: 激活相对 mapper 的位置。
        post_process: 残差 / 拼接等后处理模块或其构造函数。
        post_process_pos: 后处理相对激活的位置。
        device: 模块参数所在设备。
        dtype: 模块参数数据类型。

    Returns:
        由上述组件组装而成的 :class:`NormAct` 模块。

    Example:
        >>> import torch
        >>> m = MlpNormAct2d(64, 64, hidden_c=256)
        >>> m(torch.randn(2, 64, 8, 8)).shape
        torch.Size([2, 64, 8, 8])
    """
    return _mlp_norm_act(
        2,
        in_c,
        out_c,
        hidden_c,
        inner_act,
        bias,
        is_norm_weight,
        init_mapper_weight,
        norm,
        norm_pos,
        act,
        act_pos,
        post_process,
        post_process_pos,
        device,
        dtype,
    )


def _mlp_norm_act(
    nd: Literal[1, 2],
    in_c: int,
    out_c: int,
    hidden_c: int | Sequence[int] | None,
    inner_act: TensorMap | ActStrType | None,
    bias: bool | None,
    is_norm_weight: bool,
    init_mapper_weight: Map[[Any], Any] | None,
    norm: TensorMap | NormStrType | None,
    norm_pos: NormPosType,
    act: TensorMap | ActStrType | None,
    act_pos: ActPosType,
    post_process: PostModule | PostModuleBuilder,
    post_process_pos: PostProcessPosType,
    device: device | None,
    dtype: dtype | None,
) -> NormAct:
    """1d / 2d 共用的组装实现；``nd`` 只影响全连接层与归一化所用的维度版本。

    ``init_mapper_weight`` 这里放宽为 ``Any``：它只被当作“对每个全连接层调用的回调”
    使用，具体签名由 :func:`MlpNormAct1d` / :func:`MlpNormAct2d` 声明确保。
    """
    if isinstance(hidden_c, int):
        hidden_c = (hidden_c,)
    hidden_widths = list(hidden_c) if hidden_c is not None else []

    if inner_act is None and isinstance(act, str):
        inner_act = act  # pyright: ignore[reportAssignmentType]

    conv_cls: type[nn.Conv1d] | type[nn.Conv2d] = nn.Conv1d if nd == 1 else nn.Conv2d

    norm_c = in_c if norm_pos == "before map" else out_c
    if norm == "bn":
        norm = (nn.BatchNorm1d if nd == 1 else nn.BatchNorm2d)(norm_c, device=device, dtype=dtype)
        if bias is None:
            bias = False
    elif norm == "in":
        norm = (nn.InstanceNorm1d if nd == 1 else nn.InstanceNorm2d)(norm_c, affine=True, device=device, dtype=dtype)
        if bias is None:
            bias = False
    elif norm == "rms":
        norm = ChannelRmsNorm(norm_c, device=device, dtype=dtype)

    if bias is None:
        bias = True

    widths = [in_c, *hidden_widths, out_c]
    layers: list[nn.Module] = []
    for i, (c_in, c_out) in enumerate(zip(widths[:-1], widths[1:])):
        layer = conv_cls(c_in, c_out, kernel_size=1, bias=bias, device=device, dtype=dtype)
        if init_mapper_weight is not None:
            layer = init_mapper_weight(layer)
        if is_norm_weight:
            layer = weight_norm(layer)
        layers.append(layer)
        if i == len(widths) - 2:
            break
        inner = _build_inner_act(inner_act, c_out, device, dtype, is_first=i == 0)
        if inner is not None:
            layers.append(inner)

    mapper: TensorMap = nn.Sequential(*layers)
    act = (
        build_activation(act, in_c if act_pos == "before map" else out_c, device, dtype)
        if isinstance(act, str)
        else act
    )

    return NormAct(mapper, norm, norm_pos, act, act_pos, post_process, post_process_pos)


def _build_inner_act(
    spec: TensorMap | ActStrType | None,
    channels: int,
    device: device | None,
    dtype: dtype | None,
    is_first: bool = True,
) -> nn.Module | None:
    """按隐层宽度解析 ``inner_act``，保证每个位置拿到独立的模块实例。"""
    if spec is None:
        return None
    if isinstance(spec, str):
        return build_activation(spec, channels, device, dtype)
    if isinstance(spec, nn.Module):
        return spec if is_first else deepcopy(spec)
    return LambdaModule(spec)
