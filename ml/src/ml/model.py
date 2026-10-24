"""最小可行性分类模型：只取序列的第一个字节 token 做分类。"""

from torch import Tensor, device, dtype, nn
from torch.nn.functional import one_hot

from fishetbl.base.BaseModule import BaseTensorModule


class ByteFirstTokenClassifier(BaseTensorModule):
    """取第一个 byte token，one-hot 后过一层 ``nn.Linear`` 得到类别 logits。

    这是用于打通训练链路的占位模型：只依赖第一个 token，因此不做任何序列建模。
    0 号 token 既表示 ``0x00`` 字节，也用作 padding。

    Args:
        vocab_size: byte 词表大小，默认 256（token id 即字节值）。
        num_classes: 分类类别数。
        bias: 线性层是否使用偏置。
        device: 参数所在设备。
        dtype: 参数数据类型。
    """

    def __init__(
        self,
        vocab_size: int = 256,
        num_classes: int = 10,
        bias: bool = True,
        device: device | None = None,
        dtype: dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.projection = nn.Linear(vocab_size, num_classes, bias=bias, device=device, dtype=dtype)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Args:
            input_ids: ``(B, L)`` 的 byte token，只使用第 0 列。

        Returns:
            ``(B, num_classes)`` 的 logits。
        """
        first_token = input_ids[:, 0]
        features = one_hot(first_token, self.vocab_size).to(self.projection.weight.dtype)
        return self.projection(features)
