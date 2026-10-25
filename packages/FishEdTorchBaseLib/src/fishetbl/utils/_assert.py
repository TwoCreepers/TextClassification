from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, Literal

import torch

# 断言模式：
#   * ``"enable"``：默认值；``torch.compile`` / ``torch.export`` 追踪期间自动跳过 ——
#     trace 期对符号化条件求值只会固化成额外的 guard（甚至 graph break），与"校验
#     参数"的目的相悖；
#   * ``"disable"``：完全关闭；
#   * ``"enable_ignore_compile_status"``：即使正在 trace 也照常断言（排查编译期问题用）。
type AssertMode = Literal["enable", "disable", "enable_ignore_compile_status"]

ASSERT_MODE: AssertMode = "enable"


def _set_mode(mode: AssertMode) -> None:
    """全局切换断言模式，取值见 :data:`AssertMode`。"""
    global ASSERT_MODE
    ASSERT_MODE = mode


@contextmanager
def _assert_mode(mode: AssertMode) -> Generator[None]:
    """在 ``with`` 块内临时切换断言模式，退出时恢复原模式。"""
    global ASSERT_MODE
    original = ASSERT_MODE
    ASSERT_MODE = mode
    try:
        yield
    finally:
        ASSERT_MODE = original


def _assert(condition: Any | Callable[[], Any], *messages: Any) -> None:
    """可控断言。

    说明信息用可变参数传入，**只在断言失败时才拼接 / 字符串化**。这一点在
    ``torch.compile`` 下很关键：``_assert(cond, f"...{tuple(x.shape)}")`` 会在 trace 期
    就把符号化尺寸格式化成具体值，于是每个长度都加一条常量化 guard，导致每个 batch
    形状变化都重编译。写成 ``_assert(cond, "形状应为 ...，收到", x.shape)`` 则不会。

    Args:
        condition: 断言条件；可以是布尔值、符号化布尔值，或返回它们的无参 callable。
            传 callable 时条件**延迟到真正需要判断时才求值**：断言被跳过时 callable
            完全不会被调用，因此在 ``torch.compile`` trace 期不会去碰符号化值，可避免
            多余的 guard 或 graph break。eager 下 callable 会被立即调用，行为与直接
            传条件完全一致。
        *messages: 失败说明的片段，失败时按空格拼接，各片段依次 ``str()``。

    Raises:
        AssertionError: 断言开启且条件为假时抛出，信息由 ``*messages`` 拼接而成。
    """
    if ASSERT_MODE == "enable_ignore_compile_status":
        enabled = True
    elif ASSERT_MODE == "enable":
        enabled = not torch.compiler.is_compiling()
    else:
        enabled = False

    # 关闭时直接返回：连 condition 都不求值。
    if not enabled:
        return

    if callable(condition):
        condition = condition()

    if not condition:
        raise AssertionError(" ".join(str(part) for part in messages))
