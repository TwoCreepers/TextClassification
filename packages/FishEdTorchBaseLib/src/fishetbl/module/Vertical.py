from typing import override

from ..base.BaseModule import BaseModule
from ..mixin.NumberedContainerMixin import NumberedContainerMixin
from ..typing.alias import TransformMap


class Vertical[T](
    NumberedContainerMixin[TransformMap[T]],
    BaseModule[[T], T],
):
    @override
    def forward(self, x: T) -> T:
        for mapper in self:
            x = mapper(x)
        return x
