"""torch.compile 相关的 dynamo 全局开关。"""

import torch

from .config import CompileConfig


def configure_compilation(cfg: CompileConfig) -> None:
    """按配置写入 dynamo 开关，必须在任何 ``torch.compile`` 生效之前调用。

    * ``compiled_autograd``：用编译后的反向图（编译自动求导）；
    * ``capture_scalar_outputs``：追踪 ``.item()`` / 标量输出，避免图被切断。

    Args:
        cfg: 编译相关配置。
    """
    torch._dynamo.config.compiled_autograd = cfg.compiled_autograd
    torch._dynamo.config.capture_scalar_outputs = cfg.capture_scalar_outputs
