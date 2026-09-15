# 轻量级多模态 RAG 开发文档

> 本文描述计划建设的应用层。标记为“计划”的模块尚未实现，不能视为当前仓库已有能力。

## 1. 当前基础

本项目基于 RAG-Anything，核心入口是 `RAGAnything`。现有代码已经提供：

- `process_document_complete()`：解析文件并完成文本及多模态入库。
- `insert_content_list()`：直接插入预解析内容，适合最小验证和外部数据源。
- `aquery()`：执行普通知识库查询。
- `aquery_with_multimodal()`：将用户提供的图片、表格或公式与文本问题组合查询。
- `aquery_vlm_enhanced()`：将检索上下文中的图片交给视觉模型回答。
- `finalize_storages()`：关闭并持久化相关存储。

首版应用层将组合这些公开方法，不复制 RAG-Anything 的内部入库实现。

## 2. 总体架构

```text
┌──────────────────────┐
│ Streamlit Demo UI    │
└──────────┬───────────┘
           │ HTTP
┌──────────▼───────────┐
│ FastAPI              │
│ upload/status/query  │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ KnowledgeBaseService │
│ lifecycle + errors   │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ RAG-Anything         │
│ parse/modal/query    │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ LightRAG local store │
│ KV/vector/graph      │
└──────────────────────┘
```

### 设计原则

1. **保持简单**：首版单进程运行，使用本地存储。
2. **隔离上游**：FastAPI 和 UI 不直接依赖 LightRAG 私有属性。
3. **配置外置**：模型地址、密钥、目录和限制全部来自环境变量。
4. **可测试**：服务层通过依赖注入接收 RAG 实例，测试时使用 Fake。
5. **可复现**：样例数据和评测问题固定，结果带运行配置。

## 3. 数据流

### 3.1 文档入库

```text
上传文件
  → 校验扩展名、大小和文件名
  → 保存至隔离目录
  → 创建 task_id
  → process_document_complete()
  → MinerU 生成 content_list
  → 文本写入 LightRAG
  → 图片/表格/公式生成描述和实体
  → 更新任务状态
```

### 3.2 快速样例入库

```text
预定义 content_list
  → 结构校验
  → insert_content_list()
  → 写入本地知识库
```

该路径用于开发早期排除 MinerU、LibreOffice 和真实文档质量的影响。

### 3.3 查询

```text
用户问题
  → 参数校验
  → aquery(mode="mix")
  → 向量 + 图谱混合检索
  → LLM/VLM 生成答案
  → 提取来源和耗时
  → 标准化响应
```

## 4. 多模态内容模型

应用层统一使用 RAG-Anything 的 `content_list` 结构。

### 文本

```json
{
  "type": "text",
  "text": "正文内容",
  "page_idx": 0
}
```

### 图片

```json
{
  "type": "image",
  "img_path": "/absolute/path/to/image.png",
  "image_caption": ["图片标题"],
  "image_footnote": ["图片注释"],
  "page_idx": 1
}
```

`img_path` 必须是服务端可访问的绝对路径。客户端提供的原始文件名不能直接拼接为保存路径。

### 表格

```json
{
  "type": "table",
  "table_body": "| 指标 | 数值 |\n|---|---|\n| 准确率 | 92% |",
  "table_caption": ["实验结果"],
  "page_idx": 2
}
```

### 公式

```json
{
  "type": "equation",
  "text": "score(q,d)=similarity(q,d)",
  "text_format": "latex",
  "page_idx": 3
}
```

## 5. 配置设计

计划在 `app/core/config.py` 中集中读取以下环境变量：

| 环境变量 | 示例 | 用途 |
|---|---|---|
| `APP_HOST` | `127.0.0.1` | API 监听地址 |
| `APP_PORT` | `8000` | API 端口 |
| `WORKING_DIR` | `./rag_storage_app` | LightRAG 数据目录 |
| `UPLOAD_DIR` | `./data/uploads` | 上传文件目录 |
| `MAX_UPLOAD_MB` | `20` | 上传大小限制 |
| `LLM_MODEL` | `gpt-4o-mini` | 文本模型 |
| `VISION_MODEL` | `gpt-4o-mini` | 视觉模型 |
| `LLM_BINDING_HOST` | `https://api.openai.com/v1` | OpenAI-compatible 地址 |
| `LLM_BINDING_API_KEY` | 空 | API 密钥 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding 模型 |
| `EMBEDDING_DIM` | `1536` | 向量维度 |
| `EMBEDDING_BINDING_HOST` | 同 LLM | Embedding 地址 |
| `EMBEDDING_BINDING_API_KEY` | 同 LLM | Embedding 密钥 |

规则：

- `.env` 不提交，只提交无密钥的 `.env.example`。
- 启动时检查必要变量和 Embedding 维度。
- 对外配置接口不得返回 API Key。
- 首版固定 `PARSER=mineru`、查询模式默认 `mix`。

## 6. 服务层设计

计划新增 `KnowledgeBaseService`，职责如下：

```python
class KnowledgeBaseService:
    async def initialize(self) -> None: ...
    async def ingest_file(self, file_path: Path, task_id: str) -> str: ...
    async def ingest_content_list(self, content_list: list[dict]) -> str: ...
    async def query(self, question: str, mode: str = "mix") -> QueryResult: ...
    async def close(self) -> None: ...
```

约束：

- 全局只维护一个 RAGAnything 实例。
- FastAPI startup/lifespan 中初始化，shutdown 时关闭。
- 路由只处理 HTTP 输入输出，不包含模型组装和 RAG 业务逻辑。
- 服务层将底层异常转换为少量稳定的应用异常。
- 不在多个请求中反复初始化存储。

## 7. API 设计

### `GET /health`

响应示例：

```json
{
  "status": "ok",
  "rag_initialized": true,
  "parser": "mineru",
  "vision_enabled": true
}
```

### `POST /documents`

使用 `multipart/form-data` 上传单个文件。

响应：

```json
{
  "task_id": "uuid",
  "status": "pending",
  "file_name": "demo.pdf"
}
```

首版任务状态：

```text
pending → processing → completed
                     └→ failed
```

### `GET /documents/{task_id}`

```json
{
  "task_id": "uuid",
  "status": "completed",
  "document_id": "doc-xxx",
  "error": null
}
```

### `POST /query`

请求：

```json
{
  "question": "表格中准确率是多少？",
  "mode": "mix",
  "vlm_enhanced": false
}
```

响应：

```json
{
  "answer": "表格中的准确率为 92%。",
  "sources": ["demo.pdf"],
  "duration_ms": 1280,
  "mode": "mix"
}
```

如果当前 LightRAG 版本无法稳定返回结构化来源，首版允许 `sources` 为空，但必须在 README 的限制中说明，不通过字符串猜测来源。

## 8. 上传安全约束

- 仅允许 `.pdf`、`.md`、`.png`、`.jpg`、`.jpeg`。
- 默认限制为 20 MB。
- 使用服务端生成的 UUID 作为实际文件名。
- 使用 `Path.resolve()` 验证最终路径位于上传目录内。
- 不接受客户端传入的任意本地图片路径。
- 拒绝软链接、空文件和扩展名不匹配文件。
- 日志只记录安全文件名，不记录密钥和完整请求正文。

## 9. 并发与生命周期

首版采用单个 Uvicorn worker，原因是：

- LightRAG 本地存储和进程内任务表不适合多进程共享。
- 多模态解析与模型调用本身已经是异步流程。
- 单进程更容易复现和演示。

应用关闭时必须执行：

```python
await rag.finalize_storages()
```

同一时间只允许一个文档执行重型解析；查询可以通过信号量限制并发。若未来需要多 worker，应先将任务状态、存储和锁迁移到外部服务。

## 10. 错误响应

统一错误结构：

```json
{
  "error": {
    "code": "DOCUMENT_PARSE_FAILED",
    "message": "文档解析失败",
    "request_id": "uuid"
  }
}
```

计划错误码：

| 错误码 | HTTP 状态 | 含义 |
|---|---:|---|
| `INVALID_FILE_TYPE` | 415 | 文件类型不支持 |
| `FILE_TOO_LARGE` | 413 | 文件超过限制 |
| `DOCUMENT_NOT_FOUND` | 404 | 任务不存在 |
| `DOCUMENT_PARSE_FAILED` | 422 | 解析失败 |
| `MODEL_UNAVAILABLE` | 503 | 模型服务不可用 |
| `RAG_NOT_READY` | 503 | RAG 尚未初始化 |
| `INTERNAL_ERROR` | 500 | 未分类内部错误 |

对外返回简洁信息，完整异常仅写入服务端日志。

## 11. 测试策略

### 11.1 默认测试不调用模型

使用 Fake RAG 服务验证：

- 生命周期是否正确。
- 上传文件是否安全保存。
- 任务状态是否正确变化。
- 查询参数是否正确传递。
- 异常是否映射为稳定响应。

### 11.2 真实模型测试显式开启

真实模型测试使用独立标记，例如：

```bash
pytest -m integration
```

未设置 API Key 时自动跳过，避免 CI 意外产生费用。

### 11.3 固定评测集

建议使用 JSONL：

```json
{"id":"table-01","question":"准确率是多少？","expected_keywords":["92%"],"type":"table"}
{"id":"image-01","question":"图片展示什么？","expected_keywords":["架构"],"type":"image"}
```

评测输出至少包括：

- 总问题数和成功请求数。
- 关键词命中率。
- 各模态命中率。
- 平均及 P95 查询耗时。
- 失败问题及原始答案。

关键词命中率只能作为简易回归指标，不应表述为严格语义准确率。

## 12. 本地运行方式（计划）

完成应用层后，预期命令如下：

```bash
conda env create --file environment.yml
conda activate raganything-dev

# Install CPU PyTorch first so pip does not pull the CUDA runtime.
python -m pip install "torch>=2.6,<3" torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

cp env.example .env

# The API and UI commands become available after the application phase.
uvicorn app.main:app --reload --workers 1
streamlit run ui/streamlit_app.py
```

正式实现时应以最终 README 和 `pyproject.toml` 为准。

## 13. Git 工作流

- `main`：保持稳定，跟随个人 Fork 的默认分支。
- `upstream/main`：原始 HKUDS 项目，只用于同步。
- `feat/simple-multimodal-rag`：当前项目开发分支。
- 每个阶段使用独立的小提交，避免将模型输出和本地存储提交到 Git。
- 功能完成并测试后，再通过 Pull Request 合并到个人 `main`。

推荐提交类型：

```text
feat: 新功能
fix: 缺陷修复
test: 测试
docs: 文档
refactor: 不改变行为的重构
chore: 工程维护
```

## 14. 可观测指标

每次入库建议记录：

- 文件类型和安全文件名。
- 文档处理状态。
- 解析总耗时。
- 各内容类型数量。
- 错误阶段和错误码。

每次查询建议记录：

- 请求 ID。
- 查询模式。
- 是否启用 VLM。
- 查询耗时。
- 是否成功。

日志中禁止记录 API Key、完整图片 Base64 和敏感文档全文。

## 15. 已知限制

- MinerU 初次使用可能需要下载模型，启动成本较高。
- 本地存储模式主要用于单机演示，不面向高并发生产环境。
- 多模态描述质量依赖视觉模型。
- 表格和公式质量受文档解析结果影响。
- 首版任务状态保存在内存中，服务重启后不会保留。
- 来源返回能力取决于 LightRAG 查询结果格式，需要在实现阶段验证。

这些限制应保留在最终 README 中，便于面试时准确解释技术边界。
