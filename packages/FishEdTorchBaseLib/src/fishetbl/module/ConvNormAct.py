from typing import Literal

from torch import device, dtype, nn
from torch.nn.utils.parametrizations import weight_norm

from ..typing.alias import ActStrType, Map, Size2d, TensorMap
from ..utils._act import build_activation
from .ChannelRmsNorm import ChannelRmsNorm
from .NormAct import ActPosType, NormAct, NormPosType, PostModule, PostModuleBuilder, PostProcessPosType


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
