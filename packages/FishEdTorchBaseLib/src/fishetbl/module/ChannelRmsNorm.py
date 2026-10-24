from typing import override

from torch import Tensor, device, dtype, nn, ones, rsqrt

from ..base.BaseModule import BaseTensorModule
from ..tensor_utils.shape import expand_dims_to_end

class ChannelRmsNorm(BaseTensorModule):
    def __init__(
            self,
            num_channels: int,
            eps: float = 1e-6,
            elementwise_affine: bool = True,
            device: device | None = None,
            dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(ones((1, num_channels), device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
    @override
    def forward(self, x: Tensor) -> Tensor:
        rms = rsqrt(x.square().mean(dim=1, keepdim=True) + self.eps)

        if self.weight is not None:
            rms = rms * expand_dims_to_end(self.weight, num=x.ndim - 2)

        return x * rms