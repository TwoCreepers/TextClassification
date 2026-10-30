# TextClassification

字节级文本分类实验仓库：不做分词，token id 直接就是 UTF-8 字节值。在 THUCNews（14 类中文新闻、
83.6 万篇）上对比两个共用同一条链路的模型——

- `first_token`：只取第一个字节过一层线性层的极简基线，用来打通数据 / 训练 / 评测链路；
- `byte_film`：`ByteFiLM-DenseNet`，纯 dense 连接 + latent query 注意力 + FiLM 调制 V。

超参集中在 `ml/config/config.toml`；模型结构与实现细节见 [`ml/README.md`](ml/README.md)。

## 结构

| 路径 | 说明 |
| --- | --- |
| `ml/` | 训练工程：数据预处理、模型、训练 / 评测 / 推理入口（细节见 [`ml/README.md`](ml/README.md)） |
| `ml/config/config.toml` | 所有超参与开关（数据、模型、优化器、EMA、日志、Trainer、编译） |
| `packages/FishEdTorchBaseLib/` | 内部 PyTorch 模块库 `fishetbl`（`NormAct` / `FiLM` / `TokenAttention` / `ChannelRmsNorm` 等） |
| `justfile` | 常用配方：`just train` / `just test` / `just infer` / `just prep` / `just prep-dedup` / `just lint` / `just cslint` |
| `pyproject.toml` | uv workspace 根，成员是 `ml` 与 `packages/FishEdTorchBaseLib` |

## 快速开始

需要 Python 3.13、[uv](https://docs.astral.sh/uv/) 与 [just](https://github.com/casey/just)
（仓库在 Windows 下开发，配方统一走 `pwsh`）。

```pwsh
uv sync       # 装齐 workspace 全部依赖（fishetbl / ml / ruff / pyright）
just prep     # 预处理（纯 Python，不去重）：原始 THUCNews -> data/processed/THUCNews 的顺序字节 pack
just train    # 训练 + 验证，checkpoint 存进 lightning_logs/（best + last 两份）
just test     # 用最新一次训练的 best.ckpt 在测试集上评测
just infer "商务部：前三季度社会消费品零售总额同比增长 8.2%"
just lint     # ruff 自动修复 + 格式化，再跑 pyright
```

需要去掉重复稿（SimHash 相似度 > 0.9）就用 .NET 通道，产物格式完全一样：

```pwsh
just prep-dedup   # 先去重、再 8:1:1 划分；需要 dotnet
just cslint       # dotnet format：按 SDK 风格格式化 tools/thucnews-dedup
```

> **跑预处理前先把杀毒软件的实时扫描关掉**（或把 `data/` 加进排除项），尤其是会拦截 IO 的
> 那些 —— Windows Defender 的实时防护、第三方安全软件的「文件保护 / IO 过滤」驱动。预处理要
> 逐个打开 83 万个小文件，实时扫描会把吞吐压到个位数 MB/s，看起来像卡死；关掉不影响结果，
> 同一份源数据重复跑出来的 pack 逐字节相同。

不用 `just` 时把配方换成对应的 `uv run` 命令即可。

## 命令

| 命令 | 作用 |
| --- | --- |
| `ml-prepare-thucnews` | 原始 `data/THUCNews/{分类}/{id}.txt` 按分类 8:1:1 切分，打包成顺序字节 pack（**不去重**；要先去重请用 `just prep-dedup`） |
| `ml-train` | 训练 + 验证。**不跑测试集**；按 `val/f1_macro` 存 `best.ckpt`，同时存最后一轮的 `last.ckpt` |
| `ml-test` | 用指定 checkpoint 在验证集或测试集上评测（`--split val` / `--split test`） |
| `ml-infer` | 文本推理，输出**全部分类**的 softmax 概率并按概率降序排列 |

各命令都支持 `--help`；`ml-test` / `ml-infer` 省略 `--checkpoint` 时自动用 `lightning_logs/` 下最新一次训练的
`best.ckpt`（想评测最后一个 epoch 就显式传同一目录里的 `last.ckpt`）。

指标名里写明平均方式，避免两种平均混用：

- `f1_macro`：先按类别算 F1 再算术平均，每类同权，**模型选择看它**；
- `f1_micro`：先累计所有类别的 TP / FP / FN 再算 F1，单标签多分类下等于准确率；
- `top1_acc_micro`：top1（argmax）命中的样本占比，micro 即整体正确率。

训练不再自动跑测试集：调参只看验证集，测试集只在要汇报结果时用 `ml-test --split test` 跑一次，
避免测试集被当成第二个验证集、报出乐观偏差的指标。

## 输出示例

`ml-infer` 会给出一条输入的全部分类，按 softmax 概率降序（下面是截断后的片段）：

```text
输入 arg1: 95 字节
内容: 商务部：前三季度社会消费品零售总额同比增长 8.2%，消费市场持续恢复
  排名  分类      概率      logit
     1  股票    0.7097     2.0846
     2  家居    0.0582    -0.4162
     3  科技    0.0541    -0.4889
   ...（共 14 个分类）
```

`ml-test` 每行一个指标，例如：

```text
=== test 集评测 ===
checkpoint: lightning_logs\byte_film\version_13\checkpoints\best.ckpt

test/loss                     0.7549  交叉熵（按样本加权平均）
test/f1_macro                 0.9034  macro F1：先按类别算 F1 再算术平均，每个类别同权
test/f1_micro                 0.9184  micro F1：先累计的 TP/FP/FN 再算 F1，单标签多分类下等于准确率
test/top1_acc_micro           0.9184  micro top1 准确率：argmax 命中的样本占比
```

（数值随 checkpoint 变化，仅作格式示例。）

## 开发方式

本仓库由人类开发者与 AI 编码代理共同维护。代理是 DeepSeek Harness 内置的
`deepseek-flash` 助手，长期参与这个仓库。

每处改动一律以 commit 与 diff 为准，人类开发者负责方向与最终审阅。
