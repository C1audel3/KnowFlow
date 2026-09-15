# 阶段 1：最小多模态 RAG 闭环报告

> 完成日期：2026-09-15  
> 开发分支：`feat/simple-multimodal-rag`  
> 状态：代码与离线工程验收通过；真实模型验收待配置 API Key

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
- 每个问题记录耗时并执行确定性关键词检查。
- `try/finally` 保证插入或查询失败时仍执行 `finalize_storages()`。

## 4. 自动化测试

新增 8 个测试，覆盖：

1. 四种模态齐全且图片文件存在。
2. 缺少模态时拒绝样例。
3. 缺少类型必填字段时拒绝样例。
4. 缺少模型凭证时在初始化前失败。
5. 插入参数、稳定文档 ID、四次 `mix` 查询及答案检查。
6. 入库异常时释放存储。
7. 查询异常时释放存储。
8. 相同文档 ID 重复运行两次。

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
8 passed in 0.88s
```

完整项目回归：

```text
404 passed in 2.76s
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

## 6. 运行真实闭环

```bash
conda activate raganything-dev
cp env.example .env
# 编辑 .env，填入真实的 LLM_BINDING_API_KEY 和 EMBEDDING_BINDING_API_KEY。
python examples/simple_multimodal_rag.py
```

重复运行验收：

```bash
python examples/simple_multimodal_rag.py --repeat 2
```

成功时脚本输出每个问题的原始答案、关键词是否命中、单次查询耗时和总耗时。

## 7. 验收结论与边界

阶段 1 的样例、执行程序、配置检查、生命周期控制和自动化回归已经完成。当前工作区不存在 `.env`，两项 API Key 均未配置，因此本次没有执行真实模型调用，也没有生成可用于简历的准确率或延迟数字。

真实模型验收必须在配置凭证后补跑，并将原始结果追加到本报告。在此之前，可以在简历中描述“实现最小多模态 RAG 链路和自动化测试”，但不应声称已经验证真实模型的问答准确率。
