# ml

THUCNews 字节级文本分类的训练框架：主模型可选 `first_token` 基线（只看第一个 byte token 过一层
`nn.Linear`）或 `byte_film`（ByteFiLM-DenseNet：单 token 特征递归 + FiLM 调制 V + 纯 dense 连接）。
训练、验证、指标与日志交给 PyTorch Lightning；评测与推理复用同一个 Lightning 模块与同一份配置。

## 命令

| 命令 | 作用 |
| --- | --- |
| `just prep` | 预处理（纯 Python）：每个分类 8:1:1 划分，写出 pack。**不含去重** |
| `just prep-dedup` | 预处理（.NET）：先 SimHash 去重（相似度 > 0.9）再 8:1:1 划分 |
| `ml-train` | 训练 + 验证。**不跑测试集**；存 `best.ckpt`（`val/f1_macro` 最优）与 `last.ckpt`（最后一个 epoch） |
| `ml-test` | 载入指定 checkpoint，在验证集（`--split val`）或测试集（`--split test`）上评测 |
| `ml-infer` | 文本推理，输出全部分类的 softmax 概率（按概率降序） |

用 `just`：

```powershell
just prep          # 预处理（纯 Python，不去重）
just prep-dedup    # 预处理（.NET，先去重再划分）
just train         # 训练 + 验证
just test          # 测试集评测最新一次训练的 best.ckpt
just test val      # 换成验证集
just infer "商务部：前三季度社会消费品零售总额同比增长 8.2%"
```

不用 `just` 时（各命令都有自己的 `--help`）：

```powershell
uv run ml-prepare-thucnews --source data/THUCNews --output data/processed/THUCNews
dotnet run -c Release --project tools/thucnews-dedup -- prep data/THUCNews data/processed/THUCNews --overwrite
uv run ml-train --config ml/config/config.toml
uv run ml-test  --config ml/config/config.toml --split test
uv run ml-test  --config ml/config/config.toml --split val --checkpoint lightning_logs/byte_film/version_13/checkpoints/best.ckpt
uv run ml-test  --config ml/config/config.toml --checkpoint lightning_logs/byte_film/version_13/checkpoints/last.ckpt
uv run ml-infer --config ml/config/config.toml "一段文本"
```

TensorBoard 日志与 checkpoint 写在 `lightning_logs/byte_film/version_*/`，用
`uv run tensorboard --logdir lightning_logs` 查看。

## 评测（`ml-test`）

训练入口只做训练 + 验证，**不会**自动跑测试集：测试集若每轮都看，就等于被当成第二个验证集，
报出来的指标是乐观偏差。所以约定：

- 挑模型 / 调超参只看验证集（`--split val`，也就是 `ModelCheckpoint` 盯的 `val/f1_macro`）；
- 测试集（`--split test`，默认）只在需要汇报结果时手动跑一次。

指标名里写明平均方式，一行一个：

| 指标 | 含义 |
| --- | --- |
| `loss` | 交叉熵（按样本加权平均，受 `[optim] label_smoothing` 影响） |
| `f1_macro` | 先按类别算 F1 再算术平均，每类同权。小类不会被大类的高 F1 掩盖，**checkpoint 按它挑** |
| `f1_micro` | 先累计所有类别的 TP / FP / FN 再算 F1，等价于逐样本统计；单标签多分类下等于准确率 |
| `top1_acc_micro` | top1（argmax）命中的样本占比。准确率报 micro 就是常用的整体正确率（macro 会变成"各类召回的均值"，类别不均衡时含义不同） |

输出形如：

```text
=== test 集评测 ===
checkpoint: lightning_logs\byte_film\version_13\checkpoints\best.ckpt

test/loss                     0.7549  交叉熵（按样本加权平均）
test/f1_macro                 0.9034  macro F1：先按类别算 F1 再算术平均，每个类别同权
test/f1_micro                 0.9184  micro F1：先累计的 TP/FP/FN 再算 F1，单标签多分类下等于准确率
test/top1_acc_micro           0.9184  micro top1 准确率：argmax 命中的样本占比

选择模型看 val/f1_macro（训练时 ModelCheckpoint 的 monitor）；测试集只用来汇报，不参与调参。
```

`--batch-size` 覆盖 `[data] batch_size`、`--limit-batches` 只跑前 N 个 batch（调试用，只影响速度与
评测范围）。注意 pack 里的文档按分类连续排列、val/test 不 shuffle，所以 `--limit-batches` 很小时
只会看到最前面几个分类，`f1_macro` 会掉得很难看，这时指标没有意义。

## 推理（`ml-infer`）

输入按 UTF-8 编码成字节（训练读的就是 `.txt` 的原始字节），截到 `[data] max_length`（`-1` 表示
不截断），右侧补 0、用 `padding mask` 挡住 pad 位置——与训练完全一致。输出**全部**分类的 softmax
概率，按概率降序：

```text
输入 arg1: 95 字节
内容: 商务部：前三季度社会消费品零售总额同比增长 8.2%，消费市场持续恢复
  排名  分类      概率      logit
     1  股票    0.7097     2.0846
     2  家居    0.0582    -0.4162
     3  科技    0.0541    -0.4889
   ...（共 num_classes 行，按概率降序）
```

- 输入来源：位置参数（可给多段）、`--file PATH`（可重复，直接读原始字节，最贴近训练）、
  标准输入（`Get-Content 新闻.txt | uv run ml-infer`）。
- `--json` 输出机器可读结果（来源、字节数、每条的全部类别概率与 logit）。
- `--device auto|cpu|cuda` 选设备，默认有 CUDA 就用。
- 推理默认关掉 `torch.compile`：单条 / 少量文本不值得编译，CPU 上编译还依赖本地 C++ 编译器。
- 分类名取自 `[data] root/dataset_info.json`；找不到时退回 `class_<编号>`（不影响概率）。

## checkpoint 与配置

训练在每个 `version_*` 目录里存两份 checkpoint（都在 `checkpoints/` 下）：

- `best.ckpt`：`val/f1_macro` 最好的那个 epoch，评测 / 推理默认用它；
- `last.ckpt`：最后一个 epoch（`save_last=True`），用来看训练结束时的状态，或在被打断后接着跑。

两者都只存 `state_dict`（`[ema] enabled` 时存的就是 EMA 平均后的权重），不存超参，因此
`ml-test` / `ml-infer` 都是"按 `--config` 重建模型 + `load_state_dict(strict=True)`"：

- 架构超参（`[arch]` 里的宽度 / 轮数 / `n_q`……、`[model] num_classes`）必须与训练时一致，
  否则直接报错；
- `[data] max_length`、`[model] use_padding_mask`、`[data] pad_to_max_length` 这类不影响参数形状的
  字段不会报错，但对不上会改变输入，指标不可比；
- `--checkpoint` 省略时取**最新一次训练**（`lightning_logs/<name>/version_*` 里 mtime 最新的）目录下的
  `best.ckpt`；想评测最后一个 epoch 就显式传同目录的 `last.ckpt`。

## 数据

原始 THUCNews 的格式是 `data/THUCNews/{分类}/{数字id}.txt`（14 个分类、83.6 万篇、约 2 GiB）。
预处理有两套，产物格式完全一样，二选一：

| 命令 | 实现 | 近重复去重 |
| --- | --- | --- |
| `just prep` | `ml/preprocess.py`（纯 Python，`ml-prepare-thucnews`） | **不包含** |
| `just prep-dedup` | `tools/thucnews-dedup`（.NET） | SimHash 相似度 > 0.9，**先去重再划分** |

两者都是「每个分类各自洗牌后 8:1:1 切分再合并」，所以三个集合都包含全部 14 个分类；产物是：

```
data/processed/THUCNews/
    dataset_info.json           # 分类顺序、各集合 / 分类的文档数、划分参数
    packs/{split}.bin           # 该集合所有文档按顺序拼接的 UTF-8 字节
    packs/{split}.offsets.npy   # int64[n+1]，第 i 篇的起止偏移
    packs/{split}.labels.npy    # int64[n]，第 i 篇的类别编号
```

`THUCNewsDataset` 一次 `mmap` 整个文件、按偏移切片，所以每个 epoch 都是顺序 I/O，不用打开 83 万个
小文件。

### 性能提示：先关掉会拦截 IO 的杀毒软件

预处理要**逐个打开 83 万个小文件**。如果安全软件对每次 `open` 都做实时扫描——Windows Defender
的实时防护、以及很多第三方安全软件的「文件保护 / IO 过滤」驱动都属于这一类——读取吞吐会被压到
个位数 MB/s，整体慢一个数量级以上，而且看起来像是"卡住了"。

- 处理前把 `data/` 目录加进排除项，或者临时关掉实时防护（Defender 的「病毒和威胁防护 → 管理设置」
  里可以加排除路径）；
- 关掉不影响结果：同一份源数据重复跑出来的 pack 逐字节相同；
- 训练阶段读的是 `packs/` 里的大文件，不受这种逐文件扫描的影响。

### 去重通道（`just prep-dedup`，需要 dotnet）

THUCNews 里同一篇稿子被反复抓取的情况不少（模板化的行情表、四六级预测卷、新浪调查页）：逐字节完全
相同的有 1,116 篇，SimHash 相似度 > 0.9 的近重复涉及 27,542 篇。**`just prep` 不做去重**，要干净
训练集就走这条通道：

1. 把 83.6 万篇一次读进内存（2.09 GiB，按 1 GiB 分块存放，绕开 .NET 单个数组 2^31 的上限）；
2. 逐篇算 64 位 SimHash（去空白后的字符 3-gram，唯一 trigram 权重）；
3. 7 个重叠 16 位窗口做 LSH，找汉明距离 ≤ 6（相似度 > 0.9）的文档对，并查集聚类，**每簇只留编号最小的一篇**；
4. 剩下的 819,228 篇再按分类各自洗牌、8:1:1 划分，写出 pack。

先去重再划分（而不是先划分再去重）是为了让去重后的样本仍然均匀落在三个集合里 —— 先划分的话，
被去掉的重复稿会集中在某些分类/集合，比例会被拉歪。

当前磁盘上那份就是这条通道的产物（819,228 篇，去掉 16,847 篇 / 2.0%）：

| split | 篇数 |
| --- | ---: |
| train | 655,376 |
| val | 81,916 |
| test | 81,936 |

去重方法与统计写在 `dataset_info.json` 的 `dedup` 字段；已有的 pack 想再补一次去重可以用
`just dedup`（`dedup-pack` 命令，只读 3 个大文件，幂等）。

- 划分种子由 `--seed`（默认 42）与分类名一起派生，重复运行 pack 逐字节相同。
- `--limit-per-class N` 每个分类只取前 N 篇（冒烟测试）。
- `--workers N` 控制并发读文件的线程数（Python 默认 16；.NET 默认 CPU 核数）。
- `--no-dedup` 只有 .NET 版有。
- 已有产物时两个命令都需要覆盖：`just prep` 不会自动覆盖（会提示加 `--overwrite`），`just prep-dedup` 默认已带。
- `dataset_info.json` 里的 `class_names` 顺序就是标签编号（按分类名排序），训练时会与
  `[model] num_classes` 交叉校验；任何一个 split 缺分类、标签越界或篇数对不上都会直接报错。
- pack 里的文档按分类连续排列、val/test 不 shuffle，所以把 `limit_val_batches` / `limit_test_batches`
  调得很小时只会看到最前面几个分类，指标没有意义（冒烟测试看 loss 能不能正常下降即可）。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | 配置 dataclass 与 TOML 解析（未知字段会报错） |
| `preprocess.py` | 纯 Python 预处理（`ml-prepare-thucnews`）：8:1:1 切分 + 生成顺序字节 pack，不含去重 |
| `data.py` | `THUCNewsDataset`（mmap 顺序字节 pack）、`pad_byte_sequences` / `collate_byte_batch`、`ByteClassificationDataModule` |
| `model.py` | 基线模型 `ByteFirstTokenClassifier`（继承 `BaseTensorModule`） |
| `byte_film.py` | `ByteFiLMDenseNet`：latent query 注意力 + dense 增宽 + FiLM 调制 V |
| `module.py` | Lightning 包装 `ByteClassifierModule`：指标集合、padding mask、`torch.compile`、优化器 / 调度 |
| `compilation.py` | `torch.compile` 相关的 dynamo 开关（编译自动求导、标量追踪） |
| `train.py` | 训练入口 `train()`；同时提供 `ml-test` / `ml-infer` 共用的组装函数（DataModule、主模型、checkpoint 定位与载入） |
| `test.py` | `ml-test`：载入 checkpoint，在验证集或测试集上评测并打印指标 |
| `infer.py` | `ml-infer`：文本 -> 全部分类的 softmax 概率（降序），支持 `--json` |

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
4. 分类头：`Dropout + RMSNorm + 1x1 Conv`。

宽度为 `d0 = n_q*d_v`，每轮增宽 `m*d_block + n_q*d_v`，压缩点上除以 `compress_denominator`，
压缩后的实际宽度即 `d_final`；超参都在 `[arch]` 段。

主干特征在送入 `q_gen`、每个 `dense_block`、`film`（作为条件）以及分类头（`head_norm` 之前）
之前各有一处 dropout，四处互相独立：`[arch] q_gen_dropout` / `dense_block_dropout` /
`film_dropout` / `head_dropout`（`0` 表示关闭；`modulate_v = false` 时 `film_dropout` 不生效）。

## 约定

- **byte 词表**：token id 即 UTF-8 字节值，`0` 号 token 兼作 padding（正常文本几乎不会出现 `0x00`）。
- **padding mask**：`[model] use_padding_mask` 默认开，`input_ids != 0` 作为 mask 挡住 pad 位置。
- **截断与补齐**：`[data] max_length` 是**截断**长度——`THUCNewsDataset.__getitem__` 读取时截到
  这里，与 `pad_to_max_length` 无关（`ml-infer` 也按同一规则截断）；填 `-1` 表示**不截断**（整篇
  喂给模型，长文显存 / 耗时都会上去，只建议实验用）。`pad_to_max_length` 只决定**补齐**长度：开则
  每个 batch 补到 `max_length`（batch 形状恒定，`torch.compile` 用 `dynamic=False` 的静态形状，只需
  编译一次），关则只补到 batch 内最长序列（变长，`dynamic` 自动切到 `True`，不单独配置），否则每个
  新长度都会重编译、`fullgraph=True` 下还会撞 `recompile_limit` 直接失败。`max_length = -1` 时没有
  固定长度可补，`pad_to_max_length` 不生效，一律变长补齐。变长补齐时通道 MLP 走
  `MlpNormAct1d(mapper_type="linear")`（内部 `nn.Linear` 而非 1x1 `Conv1d`）：inductor 会把 1d 卷积
  反向的符号长度固化成具体值，导致每个新长度都重编译。
- **F1 / top1 准确率**：只在验证集与测试集上算，指标名写明平均方式——`val/f1_macro`、`test/f1_macro`
  （macro F1，checkpoint 的 monitor）、`f1_micro`（micro F1）、`top1_acc_micro`（micro top1
  准确率）。训练集不算指标；测试集只由 `ml-test` 手动触发，训练过程不碰（见「评测」）。
- **编译**：`[compile] enabled` 控制是否对主模型 forward 做 `torch.compile(fullgraph=True)`，
  `compiled_autograd` / `capture_scalar_outputs` 分别对应编译自动求导与 `.item()` 标量追踪；
  动态形状由"输入是否定长"自动决定（`max_length` 会截断且 `pad_to_max_length` 打开才是定长），
  不单独配置（`ml-infer` 固定关掉编译）。
- **学习率调度**：`[scheduler] name` 为 `none` 时 lr 恒定，为 `cosine` 时按 epoch 余弦退火到
  `min_lr`；`warmup_epochs > 0` 时先用 `LinearLR` 从 `lr * warmup_start_factor` 线性升到 `lr`，
  再交给余弦。余弦周期取 `max_epochs - warmup_epochs`。
- **DataLoader**：`[data] pin_memory` 控制页锁定内存（`pin_memory_device` 在 torch 2.14 已
  deprecated，因此不提供该配置）。每个 split 的 DataLoader 都挂一个显式播种的 `torch.Generator`，
  种子为 `[trainer] seed` 加 split 顺序（train/val/test → +0/+1/+2），
  因此 shuffle 顺序可复现且不消耗全局 RNG。
- **EMA**：`[ema] enabled` 挂上 Lightning 自带的 `EMAWeightAveraging` callback（`decay` 默认
  0.999、`use_buffers = false` 只平均参数），验证与 checkpoint 都用平均后的权重；`ml-test` /
  `ml-infer` 载入的正是这份 EMA 权重。
