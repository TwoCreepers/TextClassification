from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Literal

import torch

ASSERT_MODE: Literal["enable", "disable", "enable_ignore_onnx_status"] = "enable"


def _set_mode(mode: Literal["enable", "disable", "enable_ignore_onnx_status"]) -> None:
    global ASSERT_MODE
    ASSERT_MODE = mode


@contextmanager
def _assert_mode(mode: Literal["enable", "disable", "enable_ignore_onnx_status"]) -> Generator[None]:
    global ASSERT_MODE
    original = ASSERT_MODE
    ASSERT_MODE = mode
    try:
        yield
    finally:
        ASSERT_MODE = original


def _assert(condition: Any, message: Any) -> None:
    global ASSERT_MODE
    if ASSERT_MODE == "enable_ignore_onnx_status":
        assert condition, message
    elif ASSERT_MODE == "enable" and not torch.onnx.is_in_onnx_export():
        assert condition, message
