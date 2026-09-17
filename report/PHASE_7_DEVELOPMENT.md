# 阶段七开发报告：大规模可复现实验框架

## 1. 阶段目标

本阶段把项目从“小样例功能验证”扩展为可写入简历、可重复执行的实验工程。核心目标是：扩大评测集、拆分检索与生成指标、增加消融实验，并保留逐题原始结果。

## 2. 完成内容

### 2.1 分层基准数据集

- 建设 12 份项目档案，共 48 个带稳定 `evidence_id` 的证据块。
- 建设 120 题评测集，包含文本 25 题、表格 20 题、图像语义 20 题、公式 15 题、跨文档 20 题、拒答 20 题。
- 难度分布固定为 easy / medium / hard 各 40 题。
- 每道题包含参考答案、关键词评分规则和证据真值，生成脚本通过断言校验规模与分布。
- 数据集 SHA-256：`9a7f4e8b4a90a9de734808e8d9ed4bfafe2ef218e09306307327b6b0ea009fed`。

本基准在“文档解析后的证据块”层进行评测，用于隔离检索与生成能力；真实 PDF、PNG 的解析链路已在此前阶段覆盖，因此本阶段不把 OCR 波动混入检索消融结果。

### 2.2 检索与指标模块

新增独立评测模块，提供：

- 无第三方依赖的 BM25 基线；
- `bge-m3` 稠密向量余弦检索；
- BM25 与稠密检索的 RRF 融合；
- Recall@1/3/5、Precision@1/3/5、MRR、nDCG@1/3/5；
- 引用抽取、引用精确率与引用召回率；
- 按问题类型与难度自动分组汇总。

### 2.3 真实模型实验流水线

实验脚本支持：

- Ollama `bge-m3` 批量生成文档和查询向量；
- 使用配置文件中的真实 `deepseek-flash` 接口；
- RAG 与无 RAG 配对实验，温度固定为 0；
- 4 路并发、逐请求 checkpoint、失败后断点恢复；
- 自动输出 JSON 总工件以及检索、生成两份逐题 CSV；
- 计算准确率、分组准确率、拒答率、幻觉率、引用质量、时延、Token、Wilson 置信区间与 McNemar 配对检验。

## 3. 主要文件

- `app/benchmark.py`：检索、RRF 与指标实现。
- `scripts/generate_phase7_benchmark.py`：确定性生成 12 文档 / 120 题。
- `scripts/phase7_experiment.py`：真实检索和生成实验入口。
- `data/benchmarks/phase7/documents/`：12 份基准档案。
- `data/evaluation/phase7_benchmark.jsonl`：120 题标注集。
- `tests/test_phase7_benchmark.py`：BM25、RRF、检索和引用指标单元测试。
- `report/artifacts/phase7_experiment.json`：配置、汇总和全部逐题结果。
- `report/artifacts/phase7_retrieval_observations.csv`：360 条检索观测。
- `report/artifacts/phase7_generation_observations.csv`：240 条生成观测。

## 4. 复现方式

```bash
conda activate raganything-dev
ollama serve
ollama pull bge-m3
python scripts/generate_phase7_benchmark.py
python scripts/phase7_experiment.py --workers 4
```

只验证检索、不调用生成 API：

```bash
python scripts/phase7_experiment.py --skip-generation
```

## 5. 验收结果

- 数据规模校验：12 文档、48 证据块、120 个唯一问题。
- 分层校验：六类问题数量符合设计，三档难度各 40 题。
- 新增单元测试：3 项全部通过。
- 真实实验：完成 120 条查询向量、三种检索策略共 360 条观测，以及 RAG / 无 RAG 共 240 次真实生成。
- 所有结果均绑定数据集哈希并保留逐题记录，不使用人工估算值。

## 6. 后续建议

下一阶段可把同一套评测接口接入公开多模态数据集，并增加 OCR 噪声、图片模糊、同义改写和 Top-K 参数敏感性实验。当前合成集带有明显项目代号，BM25 因此较强；公开集和扰动集能更客观地检验语义检索的优势。
