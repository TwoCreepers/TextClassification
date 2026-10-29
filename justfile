# 本仓库在 Windows 下开发，系统没有 sh，统一用 pwsh 执行配方。
set windows-shell := ["pwsh", "-NoProfile", "-Command"]
set shell := ["pwsh", "-NoProfile", "-Command"]

# 默认打印 --list。
default: 
    @just --list

# 用 ml/config/config.toml 快速跑一次训练（TensorBoard 日志写在 lightning_logs/）。
train:
    uv run ml-train --config ml/config/config.toml

# 预处理 THUCNews：每个分类按 8:1:1 切出 train/val/test 并生成顺序字节 pack（一次性）。
prep:
    uv run ml-prepare-thucnews --source data/THUCNews --output data/processed/THUCNews

# 代码检查：ruff 自动修复 + 格式化，最后跑 pyright 类型检查。
lint:
    uv run ruff check --fix .
    uv run ruff format .
    uv run pyright
