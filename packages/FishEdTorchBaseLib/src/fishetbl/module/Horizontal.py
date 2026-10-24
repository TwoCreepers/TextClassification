from typing import override

from ..base.BaseModule import BaseModule
from ..mixin.NumberedContainerMixin import NumberedContainerMixin
from ..typing.alias import Map


class Horizontal[TMapperInput, TMapperOutput](
    NumberedContainerMixin[Map[[TMapperInput], TMapperOutput]],
    BaseModule[[TMapperInput], tuple[TMapperOutput, ...]],
):
    @override
    def forward(self, x: TMapperInput) -> tuple[TMapperOutput, ...]:
        return tuple(mapper(x) for mapper in self)
