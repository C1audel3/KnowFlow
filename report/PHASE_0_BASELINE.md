# 阶段 0：基线环境验收报告

> 完成日期：2026-09-15  
> 开发分支：`feat/simple-multimodal-rag`  
> 结论：通过

## 1. 阶段目标

本阶段建立可重复使用的 Python 开发环境，统一依赖和配置入口，并在不调用外部模型 API 的前提下确认上游代码基线稳定。

## 2. 环境基线

| 项目 | 结果 |
|---|---|
| Conda | Miniconda `26.7.1`，系统级安装于 `/home/administrator/miniconda3` |
| Conda 环境 | `raganything-dev` |
| Python | `3.12.14` |
| pip | `26.2.1` |
| RAG-Anything | `1.4.1`，editable 安装 |
| LightRAG | `1.4.16` |
| PyTorch | `2.14.0+cpu`，CUDA disabled |
| 包源策略 | Conda 使用 `conda-forge`；PyTorch 使用官方 CPU wheel 源 |

Conda 已关闭 base 环境自动激活。项目使用 `.python-version` 和 `environment.yml` 明确 Python 3.12 基线，避免依赖当前系统 Python。

## 3. 环境复现

```bash
conda env create --file environment.yml
conda activate raganything-dev

# 先安装 CPU 版 PyTorch，避免无 GPU 的开发机下载 CUDA 运行时。
python -m pip install "torch>=2.6,<3" torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

cp env.example .env
```

若新版 Miniconda 因默认 Anaconda channels 的 ToS 检查而中断，可只使用 conda-forge 创建环境：

```bash
conda create --name raganything-dev --override-channels \
  --channel conda-forge python=3.12 pip
```

然后继续执行上面的两条 pip 安装命令。

## 4. 配置约定

- `env.example` 只保留 MVP 所需的目录、解析器、模态开关、OpenAI-compatible 模型和日志配置。
- 本地密钥只写入被 Git 忽略的 `.env`，示例文件仅包含占位值。
- LLM、视觉模型、Embedding 模型和服务地址均可通过环境变量切换。
- 数据目录默认使用项目相对路径，不绑定开发者的本机绝对路径。
- 首版启用文本、图片、表格和公式；音频、视频保持关闭。

## 5. 验证结果

### 5.1 导入与依赖

```text
raganything=1.4.1
torch=2.14.0+cpu, cuda=False
No broken requirements found.
```

`raganything`、核心依赖和 CPU PyTorch 均可正常导入，`pip check` 未发现依赖冲突。

### 5.2 上游测试基线

```text
396 passed in 4.33s
```

完整测试在 WSL 的常规执行环境中通过。受限命令沙箱无法正常结束 `asyncio.to_thread()` 创建的线程，因此测试需要在沙箱外执行；该现象属于执行容器限制，不是项目测试失败。

### 5.3 静态检查

按仓库 pre-commit 规则执行：

```bash
ruff check raganything tests --ignore=E402
```

结果：`All checks passed!`

严格 Ruff 检查会在 `raganything/raganything.py` 报告 12 个既有 `E402`。这些位置因加载 `.env`、调整导入路径后再导入项目模块而触发，仓库现有 pre-commit 配置已明确忽略该规则。本阶段不修改上游核心初始化顺序。

## 6. 阶段产物

- `environment.yml`：Conda Python 3.12 环境定义。
- `.python-version`：编辑器和 Python 版本管理工具的版本提示。
- `pyproject.toml`：统一 Python 支持范围、运行依赖和开发依赖。
- `requirements.txt`、`setup.py`：与项目元数据保持关键版本约束一致。
- `env.example`：轻量级多模态 RAG 的配置模板。
- `report/DEVELOPMENT_GUIDE.md`：更新本地环境搭建命令。

## 7. 边界与下一步

本阶段没有配置真实 API Key，也没有调用外部模型，因此尚未验证模型服务、MinerU 真实文档解析和多模态问答质量。下一阶段将使用最小 `content_list` 打通文本、图片、表格、公式的入库与查询闭环。
