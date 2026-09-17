# 阶段 6：交付与简历材料报告

> 完成日期：2026-09-17

## 1. 交付成果

阶段 6 将前五个阶段的工程结果整理成面向使用者和面试展示的最终交付：

- 重写根目录 `README.md`，从上游框架介绍切换为 KnowFlow 应用说明。
- 新增 `assets/knowflow_architecture.svg`，展示输入、应用层、RAG 编排、模型、本地存储和评测链路。
- 新增真实 Streamlit 页面截图 `report/screenshots/knowflow_ui.png`。
- 新增 `report/RESUME_AND_INTERVIEW.md`，提供简历描述、讲解提纲、常见问题和演示顺序。
- 更新开发计划与开发文档，将阶段 6 标记为完成。
- 更新项目元数据，使主页、仓库、问题追踪和上游地址边界清晰。
- 将运行时上传目录 `data/uploads/` 加入 Git 忽略规则。

## 2. 快速启动验收

README 已覆盖以下完整路径：

```text
Conda 环境
  → 安装项目依赖
  → 安装并启动 Ollama
  → 拉取 bge-m3
  → 配置 DeepSeek API Key
  → 启动 FastAPI
  → 启动 Streamlit
  → 上传样例文档
  → 执行问答与固定评测
```

所有命令均从项目根目录执行，运行时知识库、解析输出、上传文件和 `.env` 不进入 Git。

## 3. 展示材料

### 系统架构

![KnowFlow 架构](../assets/knowflow_architecture.svg)

### Streamlit 页面

![KnowFlow Streamlit 页面](./screenshots/knowflow_ui.png)

截图使用真实 Streamlit 进程和真实 FastAPI 健康/公开配置接口生成，不执行模型问答，不包含 API Key、本地绝对路径或用户文档内容。

## 4. 简历指标口径

可以安全使用的指标：

- 3 份真实样例文档，覆盖 PDF、Markdown、PNG。
- PDF 解析出 7 个内容块，包含文本、图片、表格和公式。
- 16 题固定回归集，覆盖 6 类问题。
- 两轮 32 次真实 HTTP 请求，请求成功率与固定规则命中率均为 100%。
- 470 项自动化测试通过，1 项真实模型测试默认跳过。

不能扩大表述为：

- “模型语义准确率 100%”。
- “首次生成平均耗时 131.88 ms”。
- “支持生产级高并发或分布式部署”。
- “所有回答均具有可靠引用来源”。

## 5. 安全与仓库审计

交付前检查范围包括：

- `.env`、本地知识库、解析输出和上传文件被 Git 忽略。
- 提交文件中不存在真实 API Key、个人绝对路径和模型权重。
- 评测结果只保存模型名称，不保存密钥与服务 URL。
- README 明确上游项目、二次开发范围和 MIT 许可证。
- 页面和 API 错误不展示完整内部异常。

## 6. 验证记录

阶段 6 最终验收执行：

```text
python -m pytest -q
python -m ruff check app ui scripts tests
python -m ruff format --check app ui scripts tests
python examples/process_real_documents.py --validate-only
git diff --check
```

最终测试数字和检查结果以本次提交记录为准。

本次实际结果：

```text
样例输入校验：3/3 有效
Ruff check：All checks passed
Ruff format：69 files already formatted
依赖检查：No broken requirements found
全量测试：470 passed, 1 skipped, 1 warning
FastAPI /health：HTTP 200，RAG ready
Streamlit /_stcore/health：HTTP 200，ok
```

跳过项仍是需要显式授权的真实模型集成测试；阶段 5 已通过独立真实 HTTP 评测完成模型链路验收。唯一警告来自 Starlette `TestClient` 使用的 AnyIO 旧别名。

## 7. 已知限制

- MinerU、DeepSeek 和 Ollama 的首次配置仍需要网络与本机依赖。
- 本地任务状态在服务重启后丢失。
- 页面截图只用于展示布局和服务就绪状态，不代表执行了新的模型问答。
- 仓库保留大量上游文档和核心代码；README 已明确区分 KnowFlow 应用层与上游能力。

## 8. 阶段结论

阶段 6 完成后，项目具备面向新用户的启动文档、可视化架构与页面材料、可追溯评测结果、简历描述和面试讲解稿，可以作为完整作品集项目展示。
