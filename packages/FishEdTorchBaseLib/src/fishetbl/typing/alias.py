from types import EllipsisType
from typing import Callable, Literal, NamedTuple

from torch import Tensor

type IndexAtom = int | slice | None | EllipsisType | Tensor
type SliceIndex = tuple[IndexAtom, ...]

type Map[**TArgs, TResult] = Callable[TArgs, TResult]
type TransformMap[T] = Map[[T], T]
type TensorMap = TransformMap[Tensor]

type PadMode = Literal["constant", "reflect", "replicate", "circular"]
type PadAlign = Literal["end", "start", "center_end", "center_start"]

type ActStrType = Literal["relu", "relu6", "selu", "silu", "mish", "sigmoid", "tanh", "logsigmoid", "softsign", "hardsigmoid", "hardswish", "tanhshrink", "prelu", "gelu"]

class Size2d(NamedTuple):
    h: int
    w: int