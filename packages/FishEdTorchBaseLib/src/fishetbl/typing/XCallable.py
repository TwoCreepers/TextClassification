from typing import Protocol, Self


class XCallable[TArg, TResult](Protocol):
    def __call__(self: Self, x: TArg) -> TResult: ...
