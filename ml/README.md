# ml

THUCNews 字节级文本分类的最小可行训练框架：主模型只取第一个 byte token 过一层 `nn.Linear`，
训练、验证、测试、指标与日志都交给 PyTorch Lightning。

## 运行

```powershell
uv run ml-train                                        # 使用 ml/config/config.toml
uv run ml-train --config ml/config/config.toml         # 显式指定配置
```

脚本名与配置里的名字都是"真实"的（`ml-train` → `ml.train:train`），后续接推理可以直接加
`ml-infer = "ml.infer:infer"`。TensorBoard 日志与 checkpoint 写在
`lightning_logs/byte-first-token/version_*/`，用 `uv run tensorboard --logdir lightning_logs` 查看。

数据用 `datasets` 从 Hub 拉取 `seamew/THUCNews`（`datasets` 5.x 已不支持 dataset script，
所以默认读取 Hub 自动转换的 `refs/convert/parquet` 分支）；网络受限时可自行设置 `HF_ENDPOINT`。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | 配置 dataclass 与 TOML 解析（未知字段会报错） |
| `data.py` | UTF-8 字节编码、padding、`ByteClassificationDataModule` |
| `model.py` | 主模型 `ByteFirstTokenClassifier`（继承 `BaseTensorModule`） |
| `module.py` | Lightning 包装 `ByteFirstTokenModule`，含 F1 指标与 `torch.compile` |
| `callbacks.py` | `LossLoggingCallback`：在 step 之外记录训练 loss |
| `compilation.py` | `torch.compile` 相关的 dynamo 开关（编译自动求导、标量追踪） |
| `train.py` | 训练入口 `train()`，组装 DataModule / Model / Logger / Trainer |

## 约定

- **byte 词表**：token id 即 UTF-8 字节值，`0` 号 token 兼作 padding（正常文本几乎不会出现 `0x00`）。
- **loss 日志**：`training_step` 只返回 loss，不调用 `self.log`；训练 loss 由
  `LossLoggingCallback` 在 `on_train_epoch_end` 记录，避免打断 `torch.compile` 的整图。
- **F1**：只在验证集与测试集上计算（`val/f1`、`test/f1`，macro 平均）。
- **编译**：`[compile] enabled` 控制是否对主模型 forward 做 `torch.compile(fullgraph=True)`，
  `compiled_autograd` / `capture_scalar_outputs` 分别对应编译自动求导与 `.item()` 标量追踪。
