# 阶段 3：服务层与 API 报告

> 完成日期：2026-09-15
> API：FastAPI 0.141.1、Pydantic 2.13.5
> 运行模式：单进程、单 RAG 实例、本地内存任务表

## 1. 阶段成果

阶段 3 将阶段 2 的命令行闭环封装为可通过 HTTP 调用的后端：

```text
HTTP 上传 / 查询
  → FastAPI 参数与错误处理
  → KnowledgeBaseService
  → 内存任务状态 + 并发限制
  → RAGAnything / LightRAG
```

主要实现：

- `app/core/config.py`：环境配置、目录解析、正整数和扩展名校验、公开配置白名单。
- `app/core/errors.py`：稳定且可安全公开的应用错误。
- `app/schemas/models.py`：任务、健康检查和问答的 Pydantic 模型。
- `app/services/knowledge_base.py`：RAG 生命周期、流式上传、任务编排和查询。
- `app/main.py`：FastAPI lifespan、路由和统一异常响应。
- `scripts/verify_phase3_api.py`：使用真实知识库验证 HTTP 应用层。
- `tests/app/`：配置、服务和接口自动化测试。

`pyproject.toml` 新增 `api` 可选依赖，并将这些依赖同时加入开发环境；`app*` 已加入 setuptools 包发现范围。

## 2. API 清单

| 方法 | 路径 | 成功状态 | 用途 |
|---|---|---:|---|
| `GET` | `/health` | 200 | 返回 RAG 生命周期状态 |
| `GET` | `/config/public` | 200 | 返回不含密钥和内部目录的公开配置 |
| `POST` | `/documents` | 202 | 保存文件并创建后台处理任务 |
| `GET` | `/documents/{task_id}` | 200 | 查询任务状态、文档 ID、内容统计和耗时 |
| `POST` | `/query` | 200 | 执行固定 `mix` 模式知识库问答 |

文档状态机：

```text
pending → processing → completed
                     └→ failed
```

任务状态仅保存在当前进程内存中，重启后不恢复。`sources` 字段已固定提供；当前 LightRAG 文本查询只稳定返回答案字符串，因此该字段暂为空列表，不从答案文本中猜测来源。

## 3. 上传与错误安全

上传实现按 1 MiB 分块读取，在写入时执行大小限制，避免先把完整文件载入内存。服务端使用 UUID 作为实际文件名，同时保留经过 `Path.name` 清理的展示名。

校验包括：

- 仅接受 PDF、Markdown、PNG、JPG、JPEG。
- 拒绝空文件和超出 `MAX_UPLOAD_MB` 的文件。
- 校验 PDF、PNG、JPEG 魔数；Markdown 校验 UTF-8 且拒绝 NUL 字节。
- 上传路径解析后必须直接位于配置的上传目录。
- 校验失败时删除已写入的部分文件。

错误响应使用稳定结构：

```json
{
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "不支持的文件类型",
    "request_id": "uuid"
  }
}
```

模型异常和解析异常只在服务端日志记录异常类型，不记录可能携带密钥的异常正文；对外分别返回通用模型错误或任务 `failed` 状态，不包含 API Key、内部路径和堆栈。

## 4. 生命周期与并发

- FastAPI lifespan 创建一个 `KnowledgeBaseService` 和一个 RAGAnything 实例。
- 文档处理使用容量为 1 的信号量，防止多个 MinerU 重任务同时运行。
- 查询使用 `MAX_QUERY_CONCURRENCY` 配置的信号量。
- shutdown 等待已提交的文档任务结束，再调用 `finalize_storages()`。
- 所有查询固定 `mode="mix"` 且关闭 rerank，客户端不能绕过首版范围。

真实 API 验收暴露并修复了一个生命周期缺陷：新进程加载已有知识库时，纯文本 `aquery()` 原先不会初始化 LightRAG。现在它与入库和多模态查询入口一致，先执行惰性初始化，因此可以直接查询持久化知识库。

## 5. 自动化测试

阶段 3 专项测试覆盖：

- 配置读取、公开字段白名单和非法配置。
- 服务初始化、目录创建和存储关闭。
- 文件名清理、UUID 保存和文件签名校验。
- 空文件、非法类型、伪造扩展名和超大文件。
- `pending`、`processing`、`completed`、`failed` 状态。
- Markdown 与二进制文件的解析参数差异。
- 查询参数、来源字段和模型错误脱敏。
- 5 个 OpenAPI 路径及稳定 HTTP 错误结构。
- 新进程查询已有知识库的 LightRAG 初始化回归。

专项结果：

```text
22 passed, 1 third-party deprecation warning
437 passed, 1 third-party deprecation warning（全量）
```

该警告来自 Starlette `TestClient` 对 AnyIO 旧别名的引用，不来自项目代码。

质量检查结果：全仓库 96 个 Python 文件通过格式检查，本阶段新增和修改文件通过 Ruff。全仓库 Ruff 保留阶段 2 已记录的 18 个上游 `E402`，本阶段没有新增 lint 问题。

## 6. 真实 API 验收

运行：

```bash
python -m scripts.verify_phase3_api --working-dir ./rag_storage_phase2
```

脚本通过 FastAPI `TestClient` 完整执行 lifespan、`GET /health` 和 `POST /query`，加载阶段 2 的真实本地存储：86 个图节点、251 条边和 7 个向量块。

验收问题为“Atlas 项目计划在哪个季度发布？”，接口返回包含 **2027 Q3** 的非空答案：

- HTTP 应用状态：`ok`
- RAG 初始化：成功
- 查询模式：`mix`
- 查询耗时：12,523 ms
- 结果：通过

本次答案命中已有 LLM 缓存，但向量模型连接、索引加载、混合检索、HTTP 序列化和存储释放均实际执行。

此外使用单 worker Uvicorn 在 `127.0.0.1:8765` 完成进程级启动检查，`GET /health` 和 `GET /openapi.json` 均返回 HTTP 200；随后通过正常关闭流程释放服务资源。

## 7. 阶段结论与限制

阶段 3 验收完成。项目现在具备可启动的 HTTP 后端、安全文件上传、异步任务状态、真实知识库问答、统一错误响应和自动化测试，可进入阶段 4 的 Streamlit 演示界面开发。

现有限制：

- 任务表只存在于单进程内存中。
- 本地 LightRAG 存储要求 Uvicorn 使用一个 worker。
- shutdown 会等待正在处理的重型文档完成。
- 查询来源字段暂为空，等待后续稳定的结构化检索结果接口。
- `/health` 表示应用和 RAG 对象已初始化，不主动产生付费模型探测请求。
