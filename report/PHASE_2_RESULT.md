# 阶段 2：真实文档处理报告

> 完成日期：2026-09-15  
> 运行环境：WSL、Python 3.12、Conda `raganything-dev`、CPU MinerU `pipeline`  
> 模型：DeepSeek V4.1 Flash（OpenAI-compatible）、Ollama `bge-m3`（1024 维）

## 1. 完成内容

阶段 2 已将阶段 1 的预构造 `content_list` 验证扩展为真实文件链路：

```text
PDF / Markdown / PNG
  → MinerU 或直接解析
  → content_list
  → 文本与多模态统一入库
  → bge-m3 向量索引 + LightRAG 图谱
  → mix 模式问答
```

新增产物：

- `examples/process_real_documents.py`：真实文档入库、统计、固定问答与生命周期清理。
- `scripts/generate_phase2_samples.py`：确定性生成 PDF 和 PNG 测试材料。
- `data/samples/phase2_multimodal.pdf`：包含正文、架构图、传感器表格和 F1 公式。
- `data/samples/phase2_document.md`：包含正文、Markdown 表格和相对路径图片。
- `data/samples/phase2_image.png`：独立流程图，包含标识 `PIXEL-42`。
- `tests/test_phase2_document_pipeline.py`：输入校验、真实 Markdown 解析、编排及失败清理测试。

## 2. 真实处理结果

运行命令：

```bash
conda activate raganything-dev
python examples/process_real_documents.py
```

本次结果：

| 文件 | 大小 | 内容块 | 类型统计 | 解析 | 文本入库 | 多模态入库 | 总耗时 |
|---|---:|---:|---|---:|---:|---:|---:|
| `phase2_multimodal.pdf` | 42,408 B | 7 | text 4、image 1、table 1、equation 1 | 21.867 s | 83.395 s | 185.330 s | 294.366 s |
| `phase2_document.md` | 403 B | 7 | text 6、image 1 | 0.003 s | 102.186 s | 89.788 s | 191.979 s |
| `phase2_image.png` | 30,480 B | 1 | image 1 | 46.150 s | 0 s | 87.142 s | 133.301 s |

三份文档均成功生成稳定文档 ID。全部入库后图谱包含 **86 个节点、251 条边**，真实处理及问答总耗时为 **658.434 秒**。

耗时是单次本机 CPU 与在线 API 实测值，受模型缓存、网络和服务负载影响，不应解释为框架性能上限。首次 MinerU 模型下载发生在前一次尝试中，不包含在表内。

## 3. 固定问答结果

所有查询均使用 `mode="mix"`，关闭 rerank，并对非空答案和预期关键词进行断言。

| 类型 | 问题摘要 | 预期事实 | 耗时 | 结果 |
|---|---|---|---:|---|
| PDF 正文 | Atlas 计划在哪个季度发布 | 2027 Q3 | 3.693 s | 通过 |
| PDF 表格 | Sensor-B 延迟 | 18 ms | 3.793 s | 通过 |
| PDF 图片 | 视觉网关名称 | ORION | 5.340 s | 通过 |
| PDF 公式 | F1 综合的指标 | 精确率、召回率 | 16.488 s | 通过 |
| Markdown | 知识库试点负责人 | Lin Qiao | 4.841 s | 通过 |
| 独立图片 | IMAGE CODE | PIXEL-42 | 4.632 s | 通过 |

验收结果：**3/3 文档成功，6/6 固定问答通过**。

当前检查是确定性关键词检查，不等同于语义正确率评测。更完整的准确率、拒答和来源质量评估将在阶段 5 建立固定 JSONL 数据集后完成。

## 4. 异常处理与真实缺陷修复

自动化测试覆盖：

- 文件不存在时抛出 `FileNotFoundError`。
- 不支持的扩展名在模型初始化前拒绝。
- 文档解析失败和模型查询超时向上抛出明确异常。
- 正常或异常路径均执行 `finalize_storages()`。
- Markdown 的相对图片路径被解析为存在的绝对路径。

首次真实 PDF 运行中，Hugging Face 下载发生一次 SSL 中断后自动重试并最终成功，MinerU 以退出码 0 完成且产出了结果文件。然而原包装器只要在 stderr 历史中看到 `error` 字样便抛出异常，形成虚假失败。

本阶段将判定逻辑改为以 MinerU 进程退出码为准；stderr 错误仍被记录，非零退出码仍会生成诊断提示，后续输出文件校验继续防止虚假成功。同时增加了“stderr 含重试错误但退出码为 0”的回归测试。

## 5. 自动化验证

阶段 2 专项测试：

```text
8 passed
```

新增 MinerU 重试回归测试后已运行全量测试：

```bash
ruff format --check .
ruff check .
pytest -q
```

验证结果：

- 阶段 2 及 MinerU 相关测试：`20 passed`。
- 全量测试：`415 passed in 2.38s`，相较阶段 1 的 406 项增加 9 项，无回归。
- 全仓库格式检查：81 个 Python 文件均已格式化。
- 本阶段新增与修改的 Python 文件通过 Ruff 检查。
- 全仓库 Ruff 仍报告 18 个既有 `E402`，位于 `examples/ollama_integration_example.py`、`examples/vllm_integration_example.py` 和 `raganything/raganything.py`，不属于本阶段改动。

## 6. 已知限制

- 当前视觉模型与文本模型使用同一个 OpenAI-compatible DeepSeek 配置；能处理本样例，但并非通用原生视觉模型方案。
- MinerU CPU 首次启动需要下载较大的模型，并明显增加初始化时间。
- 独立图片由 MinerU 输出为图片块，文字事实来自多模态描述，不是独立 OCR 文本块。
- 关键词命中只能验证固定事实，不能发现所有幻觉、遗漏或错误引用。
- 当前是命令行演示；上传、任务状态和 HTTP API 将在阶段 3 实现。

## 7. 阶段结论

阶段 2 验收完成。项目已经具备从真实 PDF、Markdown、PNG 到解析、索引、图谱融合和混合检索问答的可复现闭环，并留下真实运行数据、异常测试及 MinerU 兼容性修复，可进入阶段 3 的服务层与 API 开发。
