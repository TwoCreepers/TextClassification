# 本仓库在 Windows 下开发，系统没有 sh，统一用 pwsh 执行配方。
set windows-shell := ["pwsh", "-NoProfile", "-Command"]
set shell := ["pwsh", "-NoProfile", "-Command"]

# 默认打印 --list。
default: 
    @just --list

# 用 ml/config/config.toml 训练（只训练 + 验证，不碰测试集；TensorBoard 日志写在 lightning_logs/）。
train:
    uv run ml-train --config ml/config/config.toml

# 在测试集上评测最新一次训练的 best.ckpt（两个位置参数：split、checkpoint）。
#   just test        # 最新一次训练的 best.ckpt + test 集
#   just test val    # 换成验证集
#   just test test lightning_logs/byte_film/version_13/checkpoints/last.ckpt
test split="test" checkpoint="":
    uv run ml-test --config ml/config/config.toml --split {{split}} {{ if checkpoint != "" { "--checkpoint " + quote(checkpoint) } else { "" } }}

# 推理一段文本：输出全部分类的 softmax 概率（降序）。
#   just infer "商务部：前三季度社零总额同比增长 8.2%"
#   echo "一篇新闻" | uv run ml-infer     # 标准输入 / --file 等其余用法直接调 uv run ml-infer
infer text="":
    uv run ml-infer --config ml/config/config.toml {{ if text != "" { quote(text) } else { "" } }}

# 预处理 THUCNews（纯 Python）：每个分类 8:1:1 划分，写出 pack。
# **不含近重复去重**；需要去重用 just prep-dedup。
prep:
    uv run ml-prepare-thucnews --source data/THUCNews --output data/processed/THUCNews

# 预处理 THUCNews（.NET）：先去重（SimHash 相似度 > 0.9），再每个分类 8:1:1 划分，写出 pack。
# C# 实现（tools/thucnews-dedup），需要 dotnet；跑之前建议先把 data/ 加进杀毒软件的排除项。
prep-dedup:
    dotnet run -c Release --project tools/thucnews-dedup -- prep data/THUCNews data/processed/THUCNews --overwrite

# 对已有的 pack 再补一次去重（先去重后划分的产物不需要，重复执行是幂等的）。
dedup:
    dotnet run -c Release --project tools/thucnews-dedup -- dedup-pack data/processed/THUCNews 12

# C# 工具（tools/thucnews-dedup）的格式检查：dotnet format 直接按 SDK 风格重写文件。
# 只想检查不落盘：dotnet format tools/thucnews-dedup --verify-no-changes
cslint:
    dotnet format tools/thucnews-dedup

# 代码检查：ruff 自动修复 + 格式化，最后跑 pyright 类型检查。
lint:
    uv run ruff check --fix .
    uv run ruff format .
    uv run pyright
