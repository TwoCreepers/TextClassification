from functools import partial

from torch import device, dtype

from ..typing.alias import ActStrType

import torch.nn as nn

def build_activation(
    act: ActStrType,
    channels: int,
    device: device | None = None,
    dtype: dtype | None = None,
) -> nn.Module:
    """
    根据激活函数名称构造对应的 nn.Module。
    支持 inplace 的激活统一使用 inplace=True。

    Args:
        act:      激活函数名称（ActStrType）
        channels: 通道数，仅 PReLU 使用（其可学习参数数量）
        device:   模块参数所在设备
        dtype:    模块参数数据类型

    Returns:
        对应的 nn.Module 实例
    """
    factories = {
        "relu":        partial(nn.ReLU, inplace=True),
        "relu6":       partial(nn.ReLU6, inplace=True),
        "selu":        partial(nn.SELU, inplace=True),
        "silu":        partial(nn.SiLU, inplace=True),
        "mish":        partial(nn.Mish, inplace=True),
        "hardsigmoid": partial(nn.Hardsigmoid, inplace=True),
        "hardswish":   partial(nn.Hardswish, inplace=True),
        "sigmoid":     nn.Sigmoid,
        "tanh":        nn.Tanh,
        "logsigmoid":  nn.LogSigmoid,
        "softsign":    nn.Softsign,
        "tanhshrink":  nn.Tanhshrink,
        "gelu":        nn.GELU,
        "prelu":       partial(nn.PReLU, num_parameters=channels),
    }

    if act not in factories:
        raise ValueError(f"Unsupported activation: {act!r}")

    # 无参数的激活（如 nn.ReLU）构造函数不接受 device / dtype，统一在构造后迁移。
    return factories[act]().to(device=device, dtype=dtype)