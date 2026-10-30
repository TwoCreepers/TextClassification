# ml

THUCNews 字节级文本分类的训练框架：主模型可选 `first_token` 基线（只看第一个 byte token 过一层
`nn.Linear`）或 `byte_film`（ByteFiLM-DenseNet：单 token 特征递归 + FiLM 调制 V + 纯 dense 连接）。
训练、验证、测试、指标与日志都交给 PyTorch Lightning。

## 运行

```powershell
just prep      # 预处理：每个分类 8:1:1 切出 train/val/test（一次性）
just train     # 训练 + 测试
```

不用 `just` 时：

```powershell
uv run ml-prepare-thucnews --source data/THUCNews --output data/processed/THUCNews
uv run ml-train --config ml/config/config.toml
```

脚本名与配置里的名字都是"真实"的（`ml-train` → `ml.train:train`、`ml-prepare-thucnews` →
`ml.preprocess:main`），后续接推理可以直接加 `ml-infer = "ml.infer:infer"`。TensorBoard 日志与
checkpoint 写在 `lightning_logs/byte_film/version_*/`，用 `uv run tensorboard --logdir lightning_logs` 查看。

## 数据

原始 THUCNews 的格式是 `data/THUCNews/{分类}/{数字id}.txt`（14 个分类、83.6 万篇、约 2 GiB）。
预处理脚本 `ml/preprocess.py` 对**每个分类各自洗牌后 8:1:1 切分再合并**（所以三个集合都包含全部
14 个分类），并把文档按顺序打包成每个 split 一份的字节流：

```
data/processed/THUCNews/
    dataset_info.json           # 分类顺序、各集合 / 分类的文档数、划分参数
    packs/{split}.bin           # 该集合所有文档按顺序拼接的 UTF-8 字节
    packs/{split}.offsets.npy   # int64[n+1]，第 i 篇的起止偏移
    packs/{split}.labels.npy    # int64[n]，第 i 篇的类别编号
```

`THUCNewsDataset` 一次 `mmap` 整个文件、按偏移切片，所以每个 epoch 都是顺序 I/O，不用打开 83 万个
小文件（逐文件读在 Windows 上每篇约 3~5 ms，一个 epoch 要一个多小时）。

- 划分种子由 `--seed`（默认 42）与分类名一起派生，重复运行结果完全一致（pack 逐字节相同）；
  `--ratios` 可改比例。
- `--limit-per-class N` 每个分类只取前 N 篇，用于冒烟测试。
- `--workers` 是并发读文件的线程数（默认 16；单文件读取是延迟瓶颈，多线程能快 8 倍），全量 83.6 万篇实测约 2 分钟。
- `dataset_info.json` 里的 `class_names` 顺序就是标签编号（按分类名排序），训练时会与
  `[model] num_classes` 交叉校验；任何一个 split 缺分类、标签越界或篇数对不上都会直接报错。
- pack 里的文档按分类连续排列、val/test 不 shuffle，所以把 `limit_val_batches` / `limit_test_batches`
  调得很小时只会看到最前面几个分类，指标没有意义（冒烟测试看 loss 能不能正常下降即可）。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | 配置 dataclass 与 TOML 解析（未知字段会报错） |
| `preprocess.py` | `ml-prepare-thucnews`：按分类 8:1:1 切分并生成顺序字节 pack |
| `data.py` | `THUCNewsDataset`（mmap 顺序字节 pack）、`collate_byte_batch`、`ByteClassificationDataModule` |
| `model.py` | 基线模型 `ByteFirstTokenClassifier`（继承 `BaseTensorModule`） |
| `byte_film.py` | `ByteFiLMDenseNet`：latent query 注意力 + dense 增宽 + FiLM 调制 V |
| `module.py` | Lightning 包装 `ByteClassifierModule`，含 F1 指标、padding mask 与 `torch.compile` |
| `compilation.py` | `torch.compile` 相关的 dynamo 开关（编译自动求导、标量追踪） |
| `train.py` | 训练入口 `train()`，按 `[model] arch` 组装 DataModule / Model / Logger / Trainer |

## 架构（`[model] arch = "byte_film"`）

1. byte 嵌入 + `RMSNorm -> MLP(GELU)` 得到 K / V；**只有 K** 过 RoPE。该 MLP 可用
   `[arch] mlp_hidden` 加隐藏层（如 `[64, 128]` 即 `d -> 64 -> 128 -> d`），`[]` 表示保持单层；
2. 可学习 latent query 对 `(K, V)` 做注意力，得到 `(B, n_q*d_v, 1)` 的单 token 特征；
3. T 轮：`m` 个 dense block 纯 cat 增宽 → 用当前特征生成 Q、FiLM 调制 V、逐位置 `v_mix`
   → 注意力输出再 cat 回特征，宽度只增不减；`[arch] modulate_v = false` 时跳过 FiLM 与
   `v_mix`，每轮直接复用静态 V。每 `compress_every_rounds` 轮用 `RMSNorm + ChannelLinear`
   把主干宽度压到当前的 `1/compress_denominator`（下限 `n_q*d_v`，最后一轮之后不压）；
   `compress_denominator` 可以是 int（所有压缩点共用）或数组（按压缩点顺序取，如 `[2, 4]`），
   取值 >= 1，填 `1` 表示该压缩点不压（恒等映射）；
4. 分类头：`RMSNorm + 1x1 Conv`。

宽度为 `d0 = n_q*d_v`，每轮增宽 `m*d_block + n_q*d_v`，压缩点上除以 `compress_denominator`，
压缩后的实际宽度即 `d_final`；超参都在 `[arch]` 段。

## 约定

- **byte 词表**：token id 即 UTF-8 字节值，`0` 号 token 兼作 padding（正常文本几乎不会出现 `0x00`）。
- **padding mask**：`[model] use_padding_mask` 默认开，`input_ids != 0` 作为 mask 挡住 pad 位置。
- **补齐与动态形状**：`[data] pad_to_max_length` 默认开，batch 形状恒定，`torch.compile` 用静态形状
  （`dynamic=False`）只需编译一次；改成按 batch 变长补齐时 `dynamic` 会自动切到 `True`（由
  `pad_to_max_length` 推导，不再单独配置），否则每个新长度都会重编译、`fullgraph=True` 下还会撞
  `recompile_limit` 直接失败。变长补齐时通道 MLP 走 `MlpNormAct1d(mapper_type="linear")`
  （内部 `nn.Linear` 而非 1x1 `Conv1d`）：inductor 会把 1d 卷积反向的符号长度固化成具体值，
  导致每个新长度都重编译。
- **F1 / top1 准确率**：只在验证集与测试集上计算（`val/f1`、`test/f1`，macro 平均；
  `val/top1_acc`、`test/top1_acc`，top1 整体正确率）。
- **编译**：`[compile] enabled` 控制是否对主模型 forward 做 `torch.compile(fullgraph=True)`，
  `compiled_autograd` / `capture_scalar_outputs` 分别对应编译自动求导与 `.item()` 标量追踪；
  动态形状由 `[data] pad_to_max_length` 自动决定，不单独配置。
- **学习率调度**：`[scheduler] name` 为 `none` 时 lr 恒定，为 `cosine` 时按 epoch 余弦退火到
  `min_lr`；`warmup_epochs > 0` 时先用 `LinearLR` 从 `lr * warmup_start_factor` 线性升到 `lr`，
  再交给余弦。余弦周期取 `max_epochs - warmup_epochs`。
- **DataLoader**：`[data] pin_memory` 控制页锁定内存（`pin_memory_device` 在 torch 2.14 已
  deprecated，因此不提供该配置）。每个 split 的 DataLoader 都挂一个显式播种的 `torch.Generator`，
  种子为 `[trainer] seed` 加 split 顺序（train/val/test → +0/+1/+2），
  因此 shuffle 顺序可复现且不消耗全局 RNG。
