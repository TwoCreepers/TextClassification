"""Lightning 包装：把主模型的训练 / 验证 / 测试流程接到 Lightning 上。"""

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn, optim
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from .compilation import resolve_backend
from .config import Config
from .data import PAD_TOKEN_ID, fixed_padding_length


def classification_metrics(num_classes: int) -> MetricCollection:
    """验证 / 测试用的指标集合：F1（macro / micro）+ top1 准确率（micro）。

    ``MetricCollection`` 把三个指标打包成一组：``update(preds, labels)`` 一次更新全部，
    ``compute()`` 直接返回 ``{指标名: 张量}``，``reset()`` 清空累计状态。

    指标名里写死平均方式，避免 ``f1`` 这种两种平均混着叫的名字：

    * ``f1_macro``：先按类别算 F1 再取算术平均，每个类别同权。多分类下小类容易被大类的
      高 F1 掩盖，macro 对小类更敏感，因此 checkpoint 也按它挑；
    * ``f1_micro``：先累计所有类别的 TP / FP / FN 再算 F1，等价于逐样本统计；单标签多分类
      下它和准确率相等；
    * ``top1_acc_micro``：top1（argmax）命中的样本占比。``micro`` 平均下就是整体正确率。

    Args:
        num_classes: 类别数。
    """
    return MetricCollection(
        {
            "f1_macro": MulticlassF1Score(num_classes=num_classes, average="macro"),
            "f1_micro": MulticlassF1Score(num_classes=num_classes, average="micro"),
            "top1_acc_micro": MulticlassAccuracy(num_classes=num_classes, top_k=1, average="micro"),
        }
    )


class ByteClassifierModule(pl.LightningModule):
    """训练包装，真正的计算都在主模型（``arch`` 决定是哪个）里。

    训练步只做「前向 + 交叉熵」，日志（``train/loss``）在本步末尾用 ``self.log`` 记成 epoch 级
    汇总；``torch.compile`` 只包住主模型的 ``forward``，所以日志本身不进编译图、不会把前反向图
    切断。F1 与 top1 准确率只在验证集与测试集上计算，指标名见 :func:`classification_metrics`。

    Note:
        训练入口不会调用 ``trainer.test``：测试集只留给 ``ml-test`` 手动评测，免得它被当成
        第二个验证集反复调参（那样报出来的测试指标是乐观偏差）。

    Args:
        model: 主模型，``forward(input_ids, padding_mask=None) -> logits``。
        cfg: 完整配置，用到 ``data`` / ``model`` / ``optim`` / ``compile`` / ``scheduler`` 小节。
    """

    # 需要显示在进度条上的指标（其余只写日志）。
    PROGRESS_METRICS = ("f1_macro", "top1_acc_micro")

    def __init__(self, model: nn.Module, cfg: Config) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        # 标签平滑由 [optim] label_smoothing 控制；0.0 时与普通交叉熵完全等价。
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optim.label_smoothing)
        num_classes = cfg.model.num_classes
        self.val_metrics = classification_metrics(num_classes)
        self.test_metrics = classification_metrics(num_classes)
        self._compiled_forward = (
            # 动态形状跟着补齐方式走：只有「会截断 + pad_to_max_length」才有恒定形状；
            # 变长补齐（含 max_length=-1 不截断）必须开动态，否则每个新 (B, L) 都会重编译，
            # fullgraph=True 下还会撞 recompile_limit 直接失败；固定补齐保持静态最快。
            # 编译引擎由 [compile] backend 选（inductor / aot_eager / eager ...）。
            torch.compile(
                model.forward,
                backend=resolve_backend(cfg.compile.backend),
                fullgraph=cfg.compile.fullgraph,
                dynamic=fixed_padding_length(cfg.data) is None,
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
        logits = self(batch["input_ids"], self._padding_mask(batch["input_ids"]))
        loss = self.criterion(logits, batch["labels"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["input_ids"], self._padding_mask(batch["input_ids"]))
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        self.val_metrics.update(logits.argmax(dim=-1), labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val", self.val_metrics)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        logits = self(batch["input_ids"], self._padding_mask(batch["input_ids"]))
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        self.test_metrics.update(logits.argmax(dim=-1), labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test", self.test_metrics)

    def _log_metrics(self, prefix: str, metrics: MetricCollection) -> None:
        """把指标集合算一遍、逐个写进日志（键如 ``val/f1_macro``），然后清空累计状态。"""
        for name, value in metrics.compute().items():
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in self.PROGRESS_METRICS,
            )
        metrics.reset()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """AdamW；在 CUDA 上用融合 kernel（CPU 不支持，自动回退）。

        36.2M 参数的 byte_film 上实测 ``optimizer.step()``：默认实现 16.6 ms/step，
        ``fused=True`` 5.8 ms/step —— 单个 step 的 20%~40% 花在这里。

        调度由 ``[scheduler] name`` 决定：``none`` 只返回优化器；``cosine`` 额外返回
        ``CosineAnnealingLR``（可带线性 warmup），周期按 ``max_epochs - warmup_epochs`` 个
        epoch 算，每个 epoch 更新一次。
        """
        parameters = list(self.parameters())
        optimizer = optim.AdamW(
            parameters,
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay,
            fused=bool(parameters) and parameters[0].is_cuda,
        )
        sched = self.cfg.scheduler
        if sched.name == "none":
            return optimizer

        # 余弦要吃掉 warmup 之后剩下的 epoch；max_epochs <= 0（无限）时无法确定周期。
        cosine_epochs = self.cfg.trainer.max_epochs - sched.warmup_epochs
        if cosine_epochs <= 0:
            raise ValueError(
                f"[scheduler] warmup_epochs={sched.warmup_epochs} 必须小于 "
                f"[trainer] max_epochs={self.cfg.trainer.max_epochs} 才能排余弦周期"
            )
        cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=sched.min_lr)
        if sched.warmup_epochs <= 0:
            scheduler: lr_scheduler.LRScheduler = cosine
        else:
            warmup = lr_scheduler.LinearLR(
                optimizer,
                start_factor=sched.warmup_start_factor,
                total_iters=sched.warmup_epochs,
            )
            scheduler = lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[sched.warmup_epochs],
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
