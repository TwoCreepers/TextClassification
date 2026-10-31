from typing import Literal

from torch import device, dtype, nn
from torch.nn.utils.parametrizations import weight_norm

from ..typing.alias import ActStrType, Map, Size2d, TensorMap
from ..utils._act import build_activation
from .ChannelRmsNorm import ChannelRmsNorm
from .NormAct import ActPosType, NormAct, NormPosType, PostModule, PostModuleBuilder, PostProcessPosType


def ConvNormAct1d(
    in_c: int,
    out_c: int,
    ks: int,
    stride: int = 1,
    padding: Literal["valid", "same"] | int = "same",
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    dilation: int = 1,
    groups: int = 1,
    bias: bool | None = None,
    is_norm_weight: bool = False,
    init_conv_weight: Map[[nn.Conv1d], nn.Conv1d] | None = None,
    norm: TensorMap | Literal["bn", "in", "rms"] | None = "rms",
    norm_pos: NormPosType = "after map",
    act: TensorMap | ActStrType | None = "relu",
    act_pos: ActPosType = "after map",
    post_process: PostModule | PostModuleBuilder = NormAct.ResidualPostModule,
    post_process_pos: PostProcessPosType = "before act or invalid",
    device: device | None = None,
    dtype: dtype | None = None,
) -> NormAct:
    """构造在序列通道维上做一维卷积的 :class:`NormAct`（``(N, C, L)`` 进 ``(N, C', L')``）。

    与 :func:`ConvNormAct2d` 同一套参数与权重处理方式，只是把 2d 换成 1d。
    ``padding="same"`` 配 ``stride=1`` 时输出长度与输入相同，因此当 ``in_c == out_c``
    时可以把 ``post_process`` 设成 :attr:`NormAct.ResidualPostModule` 做残差。
    ``padding="same"`` 且核长为偶数（配奇膨胀率）时 PyTorch 会警告并要求零填充拷贝，
    想要满速就把 ``ks`` 取奇数。

    Args:
        in_c: 输入通道数。
        out_c: 输出通道数。
        ks: 卷积核长度。
        stride: 卷积步长。
        padding: ``"same"`` 保持长度（等价于按 ``dilation * (ks - 1)`` 对称补齐）、
            ``"valid"`` 不补、或直接给一个整数。
        padding_mode: 补零之外的其他填充方式，仅 ``padding`` 为数字时生效。
        dilation: 膨胀率（空洞卷积的间隔）；``> 1`` 时感受野变大而参数量不变。
        groups: 分组卷积的组数。
        bias: 是否使用偏置。``None`` 时跟随 ``norm``：``"bn"`` / ``"in"`` 默认
            ``False``，其余默认 ``True``。
        is_norm_weight: 是否对卷积施加 ``weight_norm``。
        init_conv_weight: 对卷积调用的权重初始化函数。
        norm: 归一化。字符串 ``"bn"`` / ``"in"`` / ``"rms"`` 按需构建，也可直接传模块，
            ``None`` 表示不做归一化。
        norm_pos: 归一化相对卷积的位置；通道数按 ``in_c`` / ``out_c`` 取。
        act: 激活，字符串或模块，``None`` 表示不使用激活。
        act_pos: 激活相对卷积的位置。
        post_process: 残差 / 拼接等后处理模块或其构造函数。
        post_process_pos: 后处理相对激活的位置。
        device: 模块参数所在设备。
        dtype: 模块参数数据类型。

    Returns:
        由上述组件组装而成的 :class:`NormAct` 模块。
    """
    norm_c = in_c if norm_pos == "before map" else out_c
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

    conv = nn.Conv1d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=ks,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        groups=groups,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    if init_conv_weight is not None:
        conv = init_conv_weight(conv)
    if is_norm_weight:
        conv = weight_norm(conv)

    act = (
        build_activation(act, in_c if act_pos == "before map" else out_c, device, dtype)
        if isinstance(act, str)
        else act
    )

    return NormAct(conv, norm, norm_pos, act, act_pos, post_process, post_process_pos)


def ConvNormAct2d(
    in_c: int,
    out_c: int,
    ks: int | Size2d,
    stride: int | Size2d = 1,
    padding: Literal["valid", "same"] | int | Size2d = "same",
    padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
    dilation: int | Size2d = 1,
    groups: int = 1,
    bias: bool | None = None,
    is_norm_weight: bool = False,
    init_conv_weight: Map[[nn.Conv2d], nn.Conv2d] | None = None,
    norm: TensorMap | Literal["bn", "in", "rms"] | None = "rms",
    norm_pos: NormPosType = "after map",
    act: TensorMap | ActStrType | None = "relu",
    act_pos: ActPosType = "after map",
    post_process: PostModule | PostModuleBuilder = NormAct.ResidualPostModule,
    post_process_pos: PostProcessPosType = "before act or invalid",
    device: device | None = None,
    dtype: dtype | None = None,
) -> NormAct:
    norm_c = in_c if norm_pos == "before map" else out_c
    if norm == "bn":
        norm = nn.BatchNorm2d(norm_c, device=device, dtype=dtype)
        if bias is None:
            bias = False
    elif norm == "in":
        norm = nn.InstanceNorm2d(norm_c, affine=True, device=device, dtype=dtype)
        if bias is None:
            bias = False
    elif norm == "rms":
        norm = ChannelRmsNorm(norm_c, device=device, dtype=dtype)

    if bias is None:
        bias = True

    conv = nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=ks,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        groups=groups,
        bias=bias,
        device=device,
        dtype=dtype,
    )
    if init_conv_weight is not None:
        conv = init_conv_weight(conv)
    if is_norm_weight:
        conv = weight_norm(conv)

    act = (
        build_activation(act, in_c if act_pos == "before map" else out_c, device, dtype)
        if isinstance(act, str)
        else act
    )

    return NormAct(conv, norm, norm_pos, act, act_pos, post_process, post_process_pos)
