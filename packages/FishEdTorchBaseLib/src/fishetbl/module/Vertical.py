from typing import override

from ..base.BaseModule import BaseModule
from ..typing.alias import TransformMap
from ..mixin.NumberedContainerMixin import NumberedContainerMixin


class Vertical[T](
    NumberedContainerMixin[TransformMap[T]],
    BaseModule[[T], T],
):
    @override
    def forward(self, x: T) -> T:
        for mapper in self:
            x = mapper(x)
        return x