# ml

THUCNews 字节级文本分类的训练框架：主模型可选 `first_token` 基线（只看第一个 byte token 过一层
`nn.Linear`）或 `byte_film`（ByteFiLM-DenseNet：单 token 特征递归 + FiLM 调制 V + 纯 dense 连接）。
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
| `model.py` | 基线模型 `ByteFirstTokenClassifier`（继承 `BaseTensorModule`） |
| `byte_film.py` | `ByteFiLMDenseNet`：latent query 注意力 + dense 增宽 + FiLM 调制 V |
| `module.py` | Lightning 包装 `ByteClassifierModule`，含 F1 指标、padding mask 与 `torch.compile` |
| `callbacks.py` | `LossLoggingCallback`：在 step 之外记录训练 loss |
| `compilation.py` | `torch.compile` 相关的 dynamo 开关（编译自动求导、标量追踪） |
| `train.py` | 训练入口 `train()`，按 `[model] arch` 组装 DataModule / Model / Logger / Trainer |

## 架构（`[model] arch = "byte_film"`）

1. byte 嵌入 + `RMSNorm -> MLP(GELU)` 得到 K / V；**只有 K** 过 RoPE；
2. 可学习 latent query 对 `(K, V)` 做注意力，得到 `(B, n_q*d_v, 1)` 的单 token 特征；
3. T 轮：`m` 个 dense block 纯 cat 增宽 → 用当前特征生成 Q、FiLM 调制 V、逐位置 `v_mix`
   → 注意力输出再 cat 回特征，宽度只增不减；
4. 分类头：`RMSNorm + 1x1 Conv`。

宽度为 `d0 = n_q*d_v`、`d_final = d0 + rounds*(m*d_block + n_q*d_v)`，超参都在 `[arch]` 段。

## 约定

- **byte 词表**：token id 即 UTF-8 字节值，`0` 号 token 兼作 padding（正常文本几乎不会出现 `0x00`）。
- **padding mask**：`[model] use_padding_mask` 默认开，`input_ids != 0` 作为 mask 挡住 pad 位置。
- **补齐与动态形状**：`[data] pad_to_max_length` 默认开，batch 形状恒定，`torch.compile` 用静态形状
  （`dynamic=False`）只需编译一次；改成按 batch 变长补齐时 `dynamic` 会自动切到 `True`（由
  `pad_to_max_length` 推导，不再单独配置），否则每个新长度都会重编译、`fullgraph=True` 下还会撞
  `recompile_limit` 直接失败。变长补齐时通道 MLP 走 `MlpNormAct1d(mapper_type="linear")`
  （内部 `nn.Linear` 而非 1x1 `Conv1d`）：inductor 会把 1d 卷积反向的符号长度固化成具体值，
  导致每个新长度都重编译。
- **loss 日志**：`training_step` 只返回 loss，不调用 `self.log`；训练 loss 由
  `LossLoggingCallback` 在 `on_train_epoch_end` 记录，避免打断 `torch.compile` 的整图。
- **F1**：只在验证集与测试集上计算（`val/f1`、`test/f1`，macro 平均）。
- **编译**：`[compile] enabled` 控制是否对主模型 forward 做 `torch.compile(fullgraph=True)`，
  `compiled_autograd` / `capture_scalar_outputs` 分别对应编译自动求导与 `.item()` 标量追踪；
  动态形状由 `[data] pad_to_max_length` 自动决定，不单独配置。
