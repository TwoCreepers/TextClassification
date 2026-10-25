from collections.abc import Sequence
from copy import deepcopy
from typing import Literal

from torch import device, dtype, nn
from torch.nn.utils.parametrizations import weight_norm

from ..typing.alias import ActStrType, Map, TensorMap
from ..utils._act import build_activation
from ..utils._assert import _assert
from .ChannelLinear import ChannelLinear
from .ChannelRmsNorm import ChannelRmsNorm
from .LambdaModule import LambdaModule
from .NormAct import ActPosType, NormAct, NormPosType, PostModule, PostModuleBuilder, PostProcessPosType

type NormStrType = Literal["bn", "in", "rms"]
type MapperType = Literal["conv", "linear"]


def MlpNormAct1d(
    widths: Sequence[int],
    inner_act: TensorMap | ActStrType | None = None,
    bias: bool | None = None,
    is_norm_weight: bool = False,
    init_mapper_weight: Map[[nn.Module], nn.Module] | None = None,
    norm: TensorMap | NormStrType | None = "rms",
    norm_pos: NormPosType = "after map",
    act: TensorMap | ActStrType | None = "relu",
    act_pos: ActPosType = "after map",
    post_process: PostModule | PostModuleBuilder = NormAct.ResidualPostModule,
    post_process_pos: PostProcessPosType = "before act or invalid",
    mapper_type: MapperType = "conv",
    device: device | None = None,
    dtype: dtype | None = None,
) -> NormAct:
    """构造在序列特征通道维上做全连接的 :class:`NormAct`，即 ``NormAct`` 版本的 1d MLP。

    输入输出形状均为 ``(N, C, L)``：每个 ``l`` 位置共享同一组权重，相当于对通道维
    做 ``Linear``。默认用 ``kernel_size=1`` 的卷积实现（与 :func:`ConvNormAct2d`
    保持同一套参数与权重处理方式）；需要 ``torch.compile`` 动态形状（变长 padding）
    时传 ``mapper_type="linear"``，见下。

    网络主体（mapper）由宽度链 ``widths`` 直接给出：``widths[0] -> widths[1] -> ...``，
    相邻两层之间按 ``inner_act`` 插入激活，因此既支持逐层收窄 / 扩张，也支持
    ``(64, 256, 64)`` 这种瓶颈结构，无需额外区分 in / hidden / out。

    Args:
        widths: 宽度链，至少两个元素，且必须都是正整数。例如 ``(64, 256, 64)``
            表示 ``64 -> 256 -> 64`` 的两层全连接。
        inner_act: 层间激活。``None`` 时沿用 ``act``（当 ``act`` 为字符串时取同名
            激活，否则不插激活）；字符串按每个位置的通道数构建独立实例，模块实例则会被
            复用（后续位置深拷贝，保证各自持有独立参数）。
        bias: 是否使用偏置。``None`` 时跟随 ``norm``：使用 ``"bn"`` / ``"in"`` 时
            默认 ``False``，其余情况默认 ``True``。
        is_norm_weight: 是否对所有全连接层施加 ``weight_norm``。
        init_mapper_weight: 对每个映射层调用的权重初始化函数（``"conv"`` 时收到
            :class:`torch.nn.Conv1d`，``"linear"`` 时收到 :class:`ChannelLinear`）。
        norm: 归一化。字符串 ``"bn"`` / ``"in"`` / ``"rms"`` 按需构建，
            也可直接传入模块，``None`` 表示不做归一化。
        norm_pos: 归一化相对 mapper 的位置；通道数按 ``widths[0]`` / ``widths[-1]`` 取。
        act: 激活，字符串或模块，``None`` 表示不使用激活。
        act_pos: 激活相对 mapper 的位置。
        post_process: 残差 / 拼接等后处理模块或其构造函数。
        post_process_pos: 后处理相对激活的位置。
        mapper_type: 映射层的实现方式。``"conv"``（默认）用 ``kernel_size=1`` 的
            :class:`torch.nn.Conv1d`；``"linear"`` 用 :class:`ChannelLinear`（内部走
            matmul）。两者数学等价，但**变长 padding + ``torch.compile`` 动态形状时必须用
            ``"linear"``**：inductor 的 1d 卷积反向需要静态 stride，会把符号化的长度固化成
            具体值，于是每个新长度都重编译（``fullgraph=True`` 下还会撞
            ``recompile_limit`` 直接失败）。
        device: 模块参数所在设备。
        dtype: 模块参数数据类型。

    Returns:
        由上述组件组装而成的 :class:`NormAct` 模块。

    Raises:
        AssertionError: ``widths`` 少于两个元素，或存在非正宽度。校验通过
            :func:`fishetbl.utils._assert._assert` 完成，可用
            ``_assert_mode("disable")`` 关闭。

    Example:
        >>> import torch
        >>> m = MlpNormAct1d((64, 256, 64))            # 64 -> 256 -> 64，带残差
        >>> m(torch.randn(2, 64, 32)).shape
        torch.Size([2, 64, 32])
        >>> m = MlpNormAct1d((32, 64, 48), post_process=NormAct.IdentityPostModule)
        >>> m(torch.randn(2, 32, 8)).shape
        torch.Size([2, 48, 8])
    """
    _assert(lambda: len(widths) >= 2, f"widths 至少需要两个元素，收到 {len(widths)} 个。")
    _assert(lambda: all(width > 0 for width in widths), f"widths 必须都是正整数，收到 {tuple(widths)}。")

    if inner_act is None and isinstance(act, str):
        # act 是字符串时就复用同名激活；具体支持的名称由 build_activation 校验。
        resolved_inner_act: TensorMap | str | None = act
    else:
        resolved_inner_act = inner_act

    norm_c = widths[0] if norm_pos == "before map" else widths[-1]
    if norm == "bn":
        norm = nn.BatchNorm1d(norm_c, device=device, dtype=dtype)
        if bias is None:
            bias = False
    elif norm == "in":
        norm = nn.InstanceNorm1d(norm_c, affine=True, device=device, dtype=dtype)
        if bias is None:
            bias = False
    elif norm == "rms":
        norm = ChannelRmsNorm(norm_c, device=device, dtype=dtype)

    if bias is None:
        bias = True

    layers: list[nn.Module] = []
    last_pair = len(widths) - 2
    for i in range(last_pair + 1):
        c_in, c_out = widths[i], widths[i + 1]
        layer: nn.Module = (
            nn.Conv1d(c_in, c_out, kernel_size=1, bias=bias, device=device, dtype=dtype)
            if mapper_type == "conv"
            else ChannelLinear(c_in, c_out, bias=bias, device=device, dtype=dtype)
        )
        if init_mapper_weight is not None:
            layer = init_mapper_weight(layer)
        if is_norm_weight:
            layer = weight_norm(layer)
        layers.append(layer)
        if i == last_pair:
            break
        inner = _build_inner_act(resolved_inner_act, c_out, device, dtype, is_first=i == 0)
        if inner is not None:
            layers.append(inner)

    mapper: TensorMap = nn.Sequential(*layers)
    act = (
        build_activation(act, widths[0] if act_pos == "before map" else widths[-1], device, dtype)
        if isinstance(act, str)
        else act
    )

    return NormAct(mapper, norm, norm_pos, act, act_pos, post_process, post_process_pos)


def _build_inner_act(
    spec: TensorMap | str | None,
    channels: int,
    device: device | None,
    dtype: dtype | None,
    is_first: bool = True,
) -> nn.Module | None:
    """按所在位置的通道数解析 ``inner_act``，保证每个位置拿到独立的模块实例。"""
    if spec is None:
        return None
    if isinstance(spec, str):
        return build_activation(spec, channels, device, dtype)
    if isinstance(spec, nn.Module):
        return spec if is_first else deepcopy(spec)
    return LambdaModule(spec)
