from typing import Self, Protocol

class XCallable[TArg, TResult](Protocol):
    def __call__(
        self:Self,
        x: TArg
    ) -> TResult:
        ...
