# 阶段 4：Streamlit 演示界面报告

> 完成日期：2026-09-16
> UI：Streamlit 1.64.0、HTTPX
> 联调模型：DeepSeek V4.1 Flash、Ollama `bge-m3`

## 1. 阶段成果

阶段 4 为阶段 3 的 HTTP 后端增加了可直接演示的中文 Web 界面：

```text
Streamlit 页面
  → 安全 HTTP 客户端
  → FastAPI
  → KnowledgeBaseService
  → LightRAG + DeepSeek + Ollama
```

新增产物：

- `ui/client.py`：配置、响应模型、上传预检、错误脱敏、任务轮询和 API 客户端。
- `ui/streamlit_app.py`：服务状态、文件上传、处理结果和知识问答页面。
- `scripts/verify_phase4_ui.py`：自动操作真实 Streamlit 页面完成问答验收。
- `tests/ui/`：HTTP 客户端、错误路径、轮询和页面离线测试。

`pyproject.toml` 新增 `ui` 可选依赖并将 `ui*` 加入包发现范围；开发依赖包含 Streamlit。

## 2. 页面功能

页面包含以下区域：

### 服务状态

- 调用 `GET /health` 和 `GET /config/public`。
- 展示 API/RAG 状态、MinerU 后端、支持格式和上传大小。
- 后端离线时显示启动提示，页面继续正常渲染。

### 文档上传

- 浏览器侧预检扩展名、空文件和文件大小。
- 调用 `POST /documents` 后轮询 `GET /documents/{task_id}`。
- 展示等待、处理、成功和失败状态。
- 完成后展示安全文件名、文档 ID、内容块、类型统计和耗时。
- 最近一次任务保存在 `st.session_state`。

### 知识问答

- 提供阶段 2 样例对应的四个推荐问题，也允许自定义问题。
- 请求固定使用 `mix` 模式，调用 `POST /query`。
- 使用聊天样式展示当前会话的问答历史。
- 展示模式、毫秒耗时和来源状态。
- 来源为空时明确显示“当前版本暂未提供结构化来源”，不解析答案文本猜测来源。

## 3. 配置与运行

新增环境变量：

```env
API_BASE_URL=http://127.0.0.1:8000
UI_REQUEST_TIMEOUT=30
UI_POLL_INTERVAL=1
UI_TASK_TIMEOUT=900
```

启动方式：

```bash
# 终端 1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

# 终端 2
streamlit run ui/streamlit_app.py
```

浏览器访问 `http://127.0.0.1:8501`。

## 4. 安全与错误处理

UI 只接受规范的 HTTP(S) API 地址，不读取 LLM 或 Embedding API Key。

自动化验证覆盖：

- API 地址规范化和非法协议、查询参数拒绝。
- 文件扩展名、空文件和大小预检。
- 上传时只发送清理后的文件名。
- 后端安全错误码和消息解析。
- 非 JSON、结构错误、网络中断和请求超时。
- 后端原始响应、主机信息和异常正文不会进入用户提示。
- 空问题不会发送网络请求。
- 页面在 FastAPI 离线时仍可渲染且不产生 Streamlit 异常。

## 5. 自动化测试

阶段 4 专项结果：

```text
20 passed
```

专项测试不调用真实模型，使用 HTTPX `MockTransport` 和 Streamlit `AppTest`，因此可在 CI 中稳定运行。

全量结果：

```text
457 passed, 1 third-party deprecation warning
```

该警告仍来自 Starlette `TestClient` 对 AnyIO 旧别名的引用。项目依赖检查无冲突，103 个 Python 文件通过格式检查，本阶段新增和修改的 Python 文件通过 Ruff。全仓库仍保留前一阶段记录的 18 个上游 `E402`，本阶段没有新增 lint 问题。

## 6. 真实端到端联调

联调使用阶段 2 已建立的本地知识库，包含 86 个图节点、251 条边和 7 个向量块。过程为：

1. 启动本机 Ollama `bge-m3`。
2. FastAPI 使用 `rag_storage_phase2` 启动。
3. Streamlit `AppTest` 加载真实页面并确认 API/RAG 就绪。
4. 页面填写此前未出现的自定义问题“请只用一句话回答：Atlas项目的目标发布季度是什么？”并点击“开始问答”。
5. UI 客户端调用真实 `/query`，答案写入页面会话状态。

结果：

- 答案包含：`2027 Q3`
- 检索模式：`mix`
- API 报告耗时：14,626 ms
- 来源字段：空列表，页面显示明确的来源缺失提示
- 验收：通过

本次问题此前未进入缓存。运行日志记录了新的 `mix:keywords` 和 `mix:query` 缓存写入，确认实际调用 DeepSeek 完成关键词提取和答案生成，并使用真实 Ollama Embedding 与 LightRAG 混合检索。

独立进程检查使用 Streamlit 端口 8766：

- `GET /_stcore/health`：HTTP 200，响应 `ok`
- `GET /`：HTTP 200，页面 HTML 7,260 字节
- FastAPI 与 Streamlit 均通过正常关闭流程停止

## 7. 已知限制

- 页面任务和问答历史只保存在当前 Streamlit 会话中。
- 上传后轮询为同步演示流程，不适合大量并发用户。
- 后端任务表仍为单进程内存状态。
- `sources` 暂为空列表，等待后端提供稳定的结构化检索来源。
- Streamlit `AppTest` 在 bare mode 下会输出缺少 `ScriptRunContext` 的提示，该提示不影响页面运行。

## 8. 阶段结论

阶段 4 验收完成。项目已具备从真实文档上传、解析和索引，到多模态知识问答及状态展示的完整可视化演示链路，可用于截图、录屏和面试现场展示。
