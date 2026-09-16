# 轻量级多模态 RAG 开发文档

> 本文描述轻量级多模态 RAG 应用层。阶段 0～5 已实现，当前系统已具备 API、Streamlit 页面和固定评测闭环。

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
│ Streamlit Demo UI    │  ← 阶段 4 已实现
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

### 3.3 阶段 1 可运行样例

先验证本地样例结构，不调用模型 API：

```bash
conda activate raganything-dev
python examples/simple_multimodal_rag.py --validate-only
```

在 `.env` 中配置 LLM、视觉模型和 Embedding 服务后，执行真实入库与四类固定问题：

```bash
python examples/simple_multimodal_rag.py
```

使用同一文档 ID 连续执行两次以手工检查幂等性：

```bash
python examples/simple_multimodal_rag.py --repeat 2
```

脚本只使用公开的 `insert_content_list()`、`aquery()` 和 `finalize_storages()` 接口。每个问题采用 `mix` 模式，输出答案、关键词检查结果和耗时；无论成功或异常都会释放存储资源。

### 3.4 阶段 2 真实文档样例

阶段 2 提供可重复生成的小型 PDF、Markdown 和 PNG：

```bash
python scripts/generate_phase2_samples.py
python examples/process_real_documents.py --validate-only
```

真实运行前应确保 `.env` 已配置、Ollama 正在运行且存在 `bge-m3`：

```bash
ollama serve
ollama pull bge-m3
python examples/process_real_documents.py
```

脚本调用公开的 `process_document_complete()` 完成解析和入库，并再次调用 `parse_document()` 的缓存结果输出内容类型、文档 ID 和各阶段耗时。PDF 与 PNG 默认使用适合 CPU 环境的 MinerU `pipeline` 后端，Markdown 使用直接解析。首次解析可能下载 MinerU 模型，所需时间不计入稳定性能基线。

默认输出目录为 `output_phase2/`，知识库存储为 `rag_storage_phase2/`。两者均为运行时产物，不提交到 Git。需要隔离运行时可指定：

```bash
python examples/process_real_documents.py \
  --working-dir ./rag_storage_phase2_test \
  --output-dir ./output_phase2_test
```

### 3.5 查询

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

`app/core/config.py` 集中读取以下环境变量：

| 环境变量 | 示例 | 用途 |
|---|---|---|
| `APP_HOST` | `127.0.0.1` | API 监听地址 |
| `APP_PORT` | `8000` | API 端口 |
| `WORKING_DIR` | `./rag_storage_app` | LightRAG 数据目录 |
| `UPLOAD_DIR` | `./data/uploads` | 上传文件目录 |
| `MAX_UPLOAD_MB` | `20` | 上传大小限制 |
| `LLM_MODEL` | `deepseek-flash` | 文本模型 |
| `VISION_MODEL` | `deepseek-flash` | 视觉处理所用模型 |
| `LLM_BINDING_HOST` | `https://api.deepseek.com` | OpenAI-compatible 地址 |
| `LLM_BINDING_API_KEY` | 空 | API 密钥 |
| `EMBEDDING_MODEL` | `bge-m3` | Ollama Embedding 模型 |
| `EMBEDDING_DIM` | `1024` | 向量维度 |
| `EMBEDDING_BINDING_HOST` | `http://localhost:11434/v1` | Embedding 地址 |
| `EMBEDDING_BINDING_API_KEY` | `ollama` | 本地兼容接口占位值 |
| `PARSER_BACKEND` | `pipeline` | MinerU CPU 解析后端 |
| `PARSER_TIMEOUT` | `900` | 单文档解析超时秒数 |
| `MAX_QUERY_CONCURRENCY` | `2` | 查询并发上限 |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Streamlit 连接的 API |
| `UI_REQUEST_TIMEOUT` | `30` | 单次 UI 请求超时秒数 |
| `UI_POLL_INTERVAL` | `1` | 文档状态轮询间隔秒数 |
| `UI_TASK_TIMEOUT` | `900` | 文档任务等待上限秒数 |

规则：

- `.env` 不提交，只提交无密钥的 `.env.example`。
- 启动时检查必要变量和 Embedding 维度。
- 对外配置接口不得返回 API Key。
- 首版固定 `PARSER=mineru`、查询模式默认 `mix`。

## 6. 服务层设计

已实现 `KnowledgeBaseService`，职责如下：

```python
class KnowledgeBaseService:
    async def initialize(self) -> None: ...
    async def submit_document(self, upload: UploadFile) -> DocumentTaskResponse: ...
    def get_document(self, task_id: str) -> DocumentTaskResponse: ...
    async def query(self, request: QueryRequest) -> QueryResponse: ...
    async def close(self) -> None: ...
```

约束：

- 全局只维护一个 RAGAnything 实例。
- FastAPI lifespan 中初始化，shutdown 时等待活动任务并关闭。
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

真实模型测试使用独立标记和显式开关：

```bash
RUN_REAL_INTEGRATION=1 python -m pytest -m integration
```

未设置开关、API Key、阶段 2 知识库或 Ollama 不可用时跳过或给出明确失败，避免 CI 意外产生费用。

### 11.3 固定评测集

固定问题集位于 `data/evaluation/phase5_questions.jsonl`，使用 JSONL：

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

API 使用阶段 2 知识库启动后，一条命令执行两轮真实评测：

```bash
python -m scripts.evaluate \
  --api-url http://127.0.0.1:8000 \
  --rounds 2 \
  --fail-on-check
```

默认结果写入 `report/artifacts/phase5_results.json`。文件包含数据集 SHA-256、模型名、运行参数、逐题答案、错误码、来源、后端耗时和端到端耗时，但不写入 API Key 或 API 地址。`--fail-on-check` 在任一规则失败时返回非零退出码，适合回归检查。

## 12. 本地运行方式

完成应用层后，预期命令如下：

```bash
conda env create --file environment.yml
conda activate raganything-dev

# Install CPU PyTorch first so pip does not pull the CUDA runtime.
python -m pip install "torch>=2.6,<3" torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,api,ui]"

cp env.example .env

uvicorn app.main:app --reload --workers 1

# 终端 2
streamlit run ui/streamlit_app.py
```

打开 `http://127.0.0.1:8000/docs` 使用 OpenAPI 页面，打开 `http://127.0.0.1:8501` 使用演示界面。真实 API 查询回归可复用阶段 2 的本地知识库：

```bash
python -m scripts.verify_phase3_api --working-dir ./rag_storage_phase2
```

在 API 已使用阶段 2 知识库启动时，可自动操作 Streamlit 页面完成真实问答验收：

```bash
python -m scripts.verify_phase4_ui --api-url http://127.0.0.1:8000
```

Streamlit 仅通过 HTTP 调用后端，不加载 RAGAnything、模型或本地存储。页面状态和问答历史保存在当前浏览器会话中，刷新或服务重启后不保证恢复。

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
