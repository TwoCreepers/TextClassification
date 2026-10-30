# TextClassification

字节级文本分类实验仓库：在 THUCNews 上把"只看第一个字节"的极简基线，和纯 dense 连接 +
latent query 注意力的 `ByteFiLM-DenseNet` 放在同一条训练链路里对比。

## 结构

| 路径 | 说明 |
| --- | --- |
| `ml/` | 训练工程：数据预处理、模型、训练入口，超参都在 `ml/config/config.toml`；细节见 [`ml/README.md`](ml/README.md) |
| `packages/FishEdTorchBaseLib/` | 内部 PyTorch 模块库 `fishetbl`（`NormAct` / `FiLM` / `TokenAttention` / `ChannelRmsNorm` 等） |
| `justfile` | 常用配方：`just train`、`just prep`、`just lint` |

## 运行

```pwsh
just prep    # 一次性预处理 THUCNews，生成顺序字节 pack
just train   # 用 ml/config/config.toml 训练（TensorBoard 日志在 lightning_logs/）
just lint    # ruff 自动修复 + 格式化，再跑 pyright
```

## 开发方式

本仓库由人类开发者与 AI 编码代理共同维护。代理是 DeepSeek Harness 内置的
`deepseek-flash` 助手，长期参与这个仓库。

每处改动一律以 commit 与 diff 为准，人类开发者负责方向与最终审阅。
