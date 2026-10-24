# 本仓库在 Windows 下开发，系统没有 sh，统一用 pwsh 执行配方。
set windows-shell := ["pwsh", "-NoProfile", "-Command"]
set shell := ["pwsh", "-NoProfile", "-Command"]

# 默认执行代码检查。
default: lint

# 代码检查：ruff 自动修复 + 格式化，最后跑 pyright 类型检查。
lint:
    uv run ruff check --fix .
    uv run ruff format .
    uv run pyright
