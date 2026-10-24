"""Lightning 包装：把主模型的训练 / 验证 / 测试流程接到 Lightning 上。"""

import pytorch_lightning as pl
import torch
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassF1Score

from .config import Config
from .model import ByteFirstTokenClassifier


class ByteFirstTokenModule(pl.LightningModule):
    """训练包装，真正的计算都在 :class:`ByteFirstTokenClassifier` 里。

    训练步只返回 loss、不做任何日志操作（日志由 :class:`ml.callbacks.LossLoggingCallback`
    在 step 之外完成），这样 ``torch.compile(fullgraph=True)`` 的前反向图不会被日志打断。
    F1 只在验证集与测试集上计算。

    Args:
        model: 主模型。
        cfg: 完整配置，用到 ``model`` / ``optim`` / ``compile`` 小节。
    """

    def __init__(self, model: ByteFirstTokenClassifier, cfg: Config) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss()
        num_classes = cfg.model.num_classes
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self._compiled_forward = (
            torch.compile(model.forward, fullgraph=cfg.compile.fullgraph) if cfg.compile.enabled else None
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """前向：若启用 ``torch.compile``，走编译后的主模型 forward。"""
        if self._compiled_forward is not None:
            return self._compiled_forward(input_ids)
        return self.model(input_ids)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """只计算并返回 loss；记录交给 LossLoggingCallback。"""
        logits = self(batch["input_ids"])
        return self.criterion(logits, batch["labels"])

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["input_ids"])
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        self.val_f1.update(logits.argmax(dim=-1), labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val/f1", self.val_f1.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.val_f1.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["input_ids"])
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
