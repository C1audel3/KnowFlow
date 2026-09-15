# 阶段 1：最小多模态 RAG 闭环报告

> 完成日期：2026-09-15  
> 开发分支：`feat/simple-multimodal-rag`  
> 状态：通过

## 1. 实现结果

阶段 1 已形成一个独立、可运行的最小链路：

```text
固定 content_list
  → 校验 text/image/table/equation
  → insert_content_list()
  → 四类固定问题（mode="mix"）
  → 非空答案与关键词检查
  → finalize_storages()
```

本阶段刻意跳过 MinerU 文档解析，用固定输入隔离解析质量和模型效果，先验证应用对 RAG-Anything 公开接口的调用方式。

## 2. 固定样例与问题

样例文件为 `data/samples/phase1_content.json`，包含四个内容块：

| 类型 | 固定事实 | 验收问题 | 关键词 |
|---|---|---|---|
| 文本 | 项目代号 Aurora | 项目代号是什么 | `Aurora` |
| 图片 | 架构包含知识图谱构建 | 包含哪种图结构构建 | `知识图谱` |
| 表格 | Q2 文档数量为 150 | Q2 文档数量是多少 | `150` |
| 公式 | F1 综合精确率与召回率 | F1 综合哪两个指标 | `精确率`、`召回率` |

图片复用仓库中的 `assets/rag_anything_framework.png`，运行时会转换成绝对路径，仓库内没有开发者本机路径。

## 3. 关键设计

- 模型名称、API 地址、Embedding 维度和密钥全部从 `.env` 读取。
- 空密钥和示例占位符会在调用模型前被拒绝。
- 插入使用稳定文档 ID `phase1-multimodal-demo-v1`，便于检查重复执行行为。
- 四个问题统一使用 LightRAG `mix` 查询模式。
- 查询关闭额外的 VLM-enhanced 回答步骤；视觉模型仍在图片入库时生成描述，减少不必要的重复视觉调用。
- 未配置 rerank 模型，因此查询显式设置 `enable_rerank=False`。
- 每个问题记录耗时并执行确定性关键词检查。
- `try/finally` 保证插入或查询失败时仍执行 `finalize_storages()`。
- DeepSeek Chat Completions 只接受 `json_object` JSON 模式；适配层会将 LightRAG 的 Pydantic `response_format` 转换为 DeepSeek 支持的格式。

## 4. 自动化测试

新增 10 个测试，覆盖：

1. 四种模态齐全且图片文件存在。
2. 缺少模态时拒绝样例。
3. 缺少类型必填字段时拒绝样例。
4. 缺少模型凭证时在初始化前失败。
5. 插入参数、稳定文档 ID、四次 `mix` 查询及答案检查。
6. 入库异常时释放存储。
7. 查询异常时释放存储。
8. 相同文档 ID 重复运行两次。
9. DeepSeek 关键词提取参数转换为 `json_object`。
10. 其他 OpenAI-compatible 服务的结构化输出参数保持不变。

测试使用 Fake RAG 隔离外部模型，不产生 API 费用，也不把预设回答描述成真实检索效果。

## 5. 验证记录

样例离线校验：

```text
valid: true
modalities: equation, image, table, text
blocks: 4
```

阶段 1 测试：

```text
10 passed in 1.31s
```

完整项目回归：

```text
406 passed in 2.57s
```

代码质量检查：

```text
76 files already formatted
All checks passed!
```

执行的检查命令：

```bash
python examples/simple_multimodal_rag.py --validate-only
python -m pytest -q
ruff format --check raganything examples tests
ruff check raganything examples tests --ignore=E402
```

## 6. 真实模型验收

实际配置：

| 组件 | 配置 |
|---|---|
| LLM | DeepSeek V4.1 Flash，API 模型名 `deepseek-flash` |
| 视觉模型 | `deepseek-flash` 原生视觉能力 |
| Embedding | Ollama `bge-m3`，1024 维 |
| 检索模式 | `mix`，关闭 rerank |
| 本地存储 | LightRAG 默认 JSON、NanoVectorDB、GraphML |

真实入库结果：

```text
内容块：4（text/image/table/equation 各 1）
多模态描述：3/3 成功
向量 chunks：4
知识图谱：124 nodes / 285 edges
资源释放：成功完成 12 个 storage finalization
```

最终代码的单轮真实查询结果：

| 问题类型 | 关键词 | 命中 | 查询耗时 |
|---|---|---:|---:|
| 文本 | `Aurora` | 是 | 2.406 s |
| 图片 | `知识图谱` | 是 | 2.649 s |
| 表格 | `150` | 是 | 1.683 s |
| 公式 | `精确率`、`召回率` | 是 | 2.300 s |

```text
固定问题命中：4/4（100%）
本轮总耗时：13.042 s
```

这里的 100% 只是 4 个固定事实问题的关键词命中率，不代表通用准确率。

重复运行验收也已完成：两轮均为 4/4 命中，第二轮复用了查询缓存并正常退出。LightRAG 会把相同文档 ID 识别为 duplicate 并跳过重复写入，但同时保留 duplicate 状态记录；后续服务层展示任务状态时需要将其正确归类，而不是当作新的解析失败。

首次查询暴露了 DeepSeek 与 LightRAG 的结构化输出兼容问题：DeepSeek 返回 `400 response_format unavailable`。适配为官方支持的 `{"type": "json_object"}` 后，关键词提取与四类查询均成功。DeepSeek JSON Output 约束见其[官方文档](https://api-docs.deepseek.com/guides/json_mode/)。

## 7. 运行方式

```bash
conda activate raganything-dev
cp env.example .env
# 编辑 .env，填入真实的 LLM_BINDING_API_KEY。
ollama serve
python examples/simple_multimodal_rag.py
```

重复运行验收：

```bash
python examples/simple_multimodal_rag.py --repeat 2
```

成功时脚本输出每个问题的原始答案、关键词是否命中、单次查询耗时和总耗时。

## 8. 验收结论与边界

阶段 1 的样例、执行程序、模型适配、生命周期控制、自动化回归和真实在线验证均已完成。`.env` 受 Git 忽略保护，报告没有记录 API Key。

当前结果可以支持简历中的事实描述：“使用 DeepSeek V4.1 Flash 与本地 BGE-M3 Embedding 打通文本、图片、表格、公式的入库及混合检索链路，10 项专项测试和 406 项完整回归通过；4 个固定问题全部命中。”不可将 4 个问题的关键词命中率扩大表述为通用准确率。
