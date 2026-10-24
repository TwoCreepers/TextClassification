"""训练 loss 的日志回调：把 loss 记录挪到 step 之外，避免打断编译图。"""

from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor


class LossLoggingCallback(pl.Callback):
    """在 epoch 结束时记录训练 loss 的均值。

    训练步只返回 loss、不做任何 ``self.log`` 调用，因此 ``torch.compile`` 编译出的
    整图前向 / 反向不会因为日志里的 ``.item()`` 之类的同步操作被撕开。
    """

    def __init__(self) -> None:
        super().__init__()
        self._losses: list[Tensor] = []

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._losses.clear()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """收下 ``training_step`` 返回的 loss（已脱离计算图）。"""
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if isinstance(loss, Tensor):
            self._losses.append(loss.detach().float())

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """把本 epoch 的平均 loss 交给 Lightning 记录。"""
        if not self._losses:
            return
        mean_loss = torch.stack(self._losses).mean()
        pl_module.log("train/loss", mean_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._losses.clear()
