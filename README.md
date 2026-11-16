# KnowFlow Paper：多模态论文阅读助手

基于 RAG-Anything / LightRAG 扩展的单机论文阅读应用，支持 PDF / Markdown 入库、
单篇问答、章节概览、图表与公式浏览、页码证据引用、阅读历史及记录导出。

论文检索先限定当前论文，再使用 BM25 + 向量 RRF；明确图号时优先定位图表、子图和
附近正文。生成阶段可传入实际原图，页面展示本次返回的文字与图像来源。
原通用知识库页面及 `/documents`、`/query` 接口保留。

## 快速启动

已有环境和模型配置时，在项目根目录运行：

```bash
conda activate raganything-dev
python -m scripts.check_paper_setup
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

另一个终端：

```bash
conda activate raganything-dev
streamlit run ui/streamlit_app.py
```

首次使用时创建 Python 3.10–3.12 环境，安装依赖：

```bash
python -m pip install -e '.[dev,image,text]'
```

仅在还没有 `.env` 时从 `env.example` 复制模板，填写文本模型、支持图片输入的视觉模型、
API 地址和密钥，以及 embedding 模型与实际维度。使用 Ollama bge-m3 时先启动 Ollama
并准备该模型。解析模型缓存可参考 [离线准备说明](docs/offline_setup.md)。

启动检查只验证配置、依赖元信息、目录权限及可选的只读API，不证明模型或解析服务
端到端可用。可以通过 `python -m scripts.check_paper_setup --api-url http://127.0.0.1:8000`
检查已运行的API。
旧知识库页面使用 `streamlit run ui/knowledge_base_app.py`。

原创演示PDF可通过 `python -m scripts.generate_paper_demo` 生成；其中实验数值为虚构测试数据。

## 论文阅读流程

上传PDF或Markdown后，页面每两秒更新状态，处理完成后刷新论文库。选择论文，输入
问题或选择结构化概览；可通过“请结合 Figure 2 和正文解释方法流程”定位图表证据。
回答展示E编号、PDF页序和原图；引用编号对应当前证据，不代表已自动验证语义支持。

每次阅读会保存问题、选项和回答快照，可在“阅读历史”中分页载入并导出JSON，载入
不会再次调用模型。上传任务状态只在当前服务进程保留，论文与历史会跨重启保存。
备份时应同时保留 `WORKING_DIR`、`INPUT_DIR` 和 `OUTPUT_DIR`；重建索引不会更新旧回答，
历史图片仍依赖解析资产。

| 接口 | 用途 |
|---|---|
| `POST /papers` | 上传PDF或Markdown，返回任务编号 |
| `GET /documents/{task_id}` | 查询当前进程的处理状态 |
| `GET /papers` | 列出持久化论文 |
| `POST /papers/read` | 问答或概览，请求包含paper_id、question、task、use_images、top_k |
| `GET /papers/{paper_id}/content` | 分页读取来源证据 |
| `GET /papers/{paper_id}/readings` | 按论文分页查询阅读历史 |
| `GET /papers/{paper_id}/readings/{history_id}` | 读取历史请求与回答 |
| `GET /papers/{paper_id}/original` | 下载原文 |

完整请求与响应模型见运行中API的 `/docs`。技术参考保存在 `docs/`，`README_zh.md`
为上游框架说明。开发报告和面试草稿仅本地保留，不纳入仓库。

## 目录与数据

```text
app/               API、服务、论文检索与评测模块
ui/                论文工作区、旧知识库页面及HTTP客户端
raganything/       复用的上游框架
scripts/           生成样例、评测、审计和环境检查
tests/            单元、API、UI及显式开启的真实模型集成测试
data/             样例、题集、来源清单与实验工件
docs/             必要的技术参考
assets/           图片与架构素材
```

实验脚本默认将JSON、逐题CSV和审阅包写入 `data/experiments/`，该目录整体Git忽略。
`.env`、上传文件、论文下载、模型缓存、`rag_storage*/`、`output*/` 和本地报告均不上传。

## 评测工具

```bash
python -m scripts.prepare_paper_benchmark --help
python -m scripts.benchmark_paper_reading --help
python -m scripts.review_paper_reading --help
```

来源清单在 `data/benchmarks/papers/manifest.json`，固定题集在 `data/evaluation/`。
真实实验需配置模型服务；论文PDF按清单另行下载，不随仓库分发。审阅工具读取已生成的
本地实验结果，评分由实际审阅者填写，不自动产生人工评价。

## 验证边界

已接入3篇真实论文、36页、241个证据块，以及30道原始问题和6道视觉开发题。
文字/图文对照检查相同证据；规则命中与来源定位不等于语义正确或引用支持。
视觉题目前是单轮开发诊断，独立人工复核仍待完成。历史通用知识库实验不能代表新增
论文阅读功能的准确率。

```bash
python -m pytest -q
```

默认测试使用离线替身，真实模型集成需显式开启。文件整理后的完整回归为537通过、
1跳过。项目尚无多用户权限、分布式任务或
生产级吞吐承诺。
