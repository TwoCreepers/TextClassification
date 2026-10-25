"""Lightning 包装：把主模型的训练 / 验证 / 测试流程接到 Lightning 上。"""

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassF1Score

from .config import Config
from .data import PAD_TOKEN_ID


class ByteClassifierModule(pl.LightningModule):
    """训练包装，真正的计算都在主模型（``arch`` 决定是哪个）里。

    训练步只返回 loss、不做任何日志操作（日志由 :class:`ml.callbacks.LossLoggingCallback`
    在 step 之外完成），这样 ``torch.compile(fullgraph=True)`` 的前反向图不会被日志打断。
    F1 只在验证集与测试集上计算。

    Args:
        model: 主模型，``forward(input_ids, padding_mask=None) -> logits``。
        cfg: 完整配置，用到 ``data`` / ``model`` / ``optim`` / ``compile`` 小节。
    """

    def __init__(self, model: nn.Module, cfg: Config) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        num_classes = cfg.model.num_classes
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self._compiled_forward = (
            # 动态形状跟着补齐方式走：变长补齐必须开（否则每个新 (B, L) 都会重编译，
            # fullgraph=True 下还会撞 recompile_limit 直接失败）；固定补齐保持静态最快。
            torch.compile(
                model.forward,
                fullgraph=cfg.compile.fullgraph,
                dynamic=not cfg.data.pad_to_max_length,
            )
            if cfg.compile.enabled
            else None
        )

    def forward(self, input_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        """前向：若启用 ``torch.compile``，走编译后的主模型 forward。"""
        if self._compiled_forward is not None:
            return self._compiled_forward(input_ids, padding_mask)
        return self.model(input_ids, padding_mask)

    def _padding_mask(self, input_ids: Tensor) -> Tensor | None:
        """按配置从 token id 生成 padding mask（``True`` = 真实 token）。"""
        if not self.cfg.model.use_padding_mask:
            return None
        return input_ids != PAD_TOKEN_ID

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """只计算并返回 loss；记录交给 LossLoggingCallback。"""
        logits = self(batch["input_ids"], self._padding_mask(batch["input_ids"]))
        return self.criterion(logits, batch["labels"])

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["input_ids"], self._padding_mask(batch["input_ids"]))
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        self.val_f1.update(logits.argmax(dim=-1), labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val/f1", self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.val_f1.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["input_ids"], self._padding_mask(batch["input_ids"]))
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        self.test_f1.update(logits.argmax(dim=-1), labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        self.log("test/f1", self.test_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.test_f1.reset()

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
        )
