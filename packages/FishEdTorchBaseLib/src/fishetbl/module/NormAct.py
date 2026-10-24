from collections.abc import Callable
from typing import Literal, override

from torch import Tensor, cat, nn

from ..base.BaseModule import BaseModule, BaseTensorModule
from ..base.TypeModule import TypeModule
from ..typing.alias import TensorMap
from .Identity import Identity

type NormPosType = Literal["before map", "after map"]
type ActPosType = Literal["before map", "after map"]
type PostProcessPosType = Literal["finally", "before act or invalid"]

type PostModule = TypeModule[[Tensor, Tensor], Tensor]
type PostModuleBuilder = Callable[[], PostModule]


class NormAct(BaseTensorModule):
    class IdentityPostModule(BaseModule[[Tensor, Tensor], Tensor]):
        @override
        def forward(self, output: Tensor, input: Tensor) -> Tensor:
            return output

    class ResidualPostModule(BaseModule[[Tensor, Tensor], Tensor]):
        @override
        def forward(self, output: Tensor, input: Tensor) -> Tensor:
            return output + input

    class ConcatenatePostModule(BaseModule[[Tensor, Tensor], Tensor]):
        @override
        def forward(self, output: Tensor, input: Tensor) -> Tensor:
            return cat((output, input), dim=1)

    norm_pos: NormPosType
    act_pos: ActPosType
    post_process_pos: PostProcessPosType

    def __init__(
        self,
        mapper: TensorMap,
        norm: TensorMap | None = None,
        norm_pos: NormPosType = "after map",
        act: TensorMap | None = None,
        act_pos: ActPosType = "after map",
        post_process: PostModule | PostModuleBuilder = IdentityPostModule,
        post_process_pos: PostProcessPosType = "before act or invalid",
    ) -> None:
        super().__init__()
        self.mapper = mapper
        self.norm = Identity[Tensor]() if norm is None else norm
        self.norm_pos = norm_pos
        self.act = Identity[Tensor]() if act is None else act
        self.act_pos = act_pos
        self.post_process = post_process if isinstance(post_process, nn.Module) else post_process()
        self.post_process_pos = post_process_pos

    @override
    def forward(self, x: Tensor) -> Tensor:
        x0 = x
        if self.norm_pos == "before map":
            x = self.norm(x)
        if self.act_pos == "before map":
            x = self.act(x)
        x = self.mapper(x)
        if self.norm_pos == "after map":
            x = self.norm(x)
        if self.act_pos == "after map":
            if self.post_process_pos == "before act or invalid":
                x = self.post_process(x, x0)
            x = self.act(x)
        if self.post_process_pos == "finally":
            x = self.post_process(x, x0)
        return x
