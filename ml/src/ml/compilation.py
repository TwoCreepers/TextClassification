"""torch.compile 相关的配置：dynamo 全局开关 + 编译引擎（backend）校验。"""

import torch

from .config import CompileConfig


def configure_compilation(cfg: CompileConfig) -> None:
    """按配置写入 dynamo 开关，必须在任何 ``torch.compile`` 生效之前调用。

    * ``compiled_autograd``：用编译后的反向图（编译自动求导）；
    * ``capture_scalar_outputs``：追踪 ``.item()`` / 标量输出，避免图被切断。

    顺带校验 ``backend``：名字拼错在这里就报错，不用等到第一次 forward 才炸。

    Args:
        cfg: 编译相关配置。
    """
    # 只为提前失败，返回值在建模时会再取一次（校验是纯字符串比较，成本可忽略）。
    resolve_backend(cfg.backend)
    torch._dynamo.config.compiled_autograd = cfg.compiled_autograd
    torch._dynamo.config.capture_scalar_outputs = cfg.capture_scalar_outputs


def resolve_backend(backend: str) -> str:
    """校验 ``[compile] backend``（编译引擎）是不是当前 torch 认识的 backend，返回原值。

    ``exclude_tags=None`` 是必须的：默认的 ``list_backends()`` 会把打了 ``debug`` 标签的
    ``eager`` / ``aot_eager`` 过滤掉，而这两个正是最常用的兜底引擎。

    Args:
        backend: 配置里的引擎名，如 ``"inductor"`` / ``"aot_eager"`` / ``"eager"``。

    Returns:
        校验通过的引擎名，直接传给 ``torch.compile(backend=...)``。

    Raises:
        ValueError: 名字不在当前 torch 的 backend 列表里（拼写错误），报错信息里列出常用值。
    """
    available = set(torch._dynamo.list_backends(exclude_tags=None))
    if backend in available:
        return backend
    # 常用值（含打了 debug 标签的 eager / aot_eager）+ 其余正式 backend；不列 *_TESTING_ONLY 那些。
    common = [name for name in ("inductor", "aot_eager", "eager") if name in available]
    others = sorted(set(torch._dynamo.list_backends()) - set(common))
    raise ValueError(
        f"[compile] backend = {backend!r} 不是当前 torch {torch.__version__} 支持的编译引擎；"
        f"常用值：{common}；其他可用：{others}"
    )
