"""Streamlit interface for the KnowFlow multimodal RAG API."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ui.client import (
    APIClient,
    DEFAULT_EXTENSIONS,
    DocumentTaskView,
    PublicConfigView,
    UIError,
    UISettings,
    sources_text,
    validate_upload,
)

SUGGESTED_QUESTIONS = (
    "Atlas 项目计划在哪个季度发布？",
    "试验结果中 Sensor-B 的延迟是多少？",
    "架构图中的视觉网关叫什么？",
    "报告中的 F1 公式综合哪两个指标？",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def get_api_client(base_url: str, timeout: float) -> APIClient:
    return APIClient(base_url, timeout)


def _initialize_state() -> None:
    st.session_state.setdefault("recent_task", None)
    st.session_state.setdefault("chat_history", [])


def _task_status_text(status: str) -> str:
    return {
        "pending": "等待处理",
        "processing": "正在解析与索引",
        "completed": "处理完成",
        "failed": "处理失败",
    }.get(status, "状态未知")


def _render_task(task: DocumentTaskView) -> None:
    st.subheader("最近一次文档任务")
    left, middle, right = st.columns(3)
    left.metric("状态", _task_status_text(task.status))
    middle.metric(
        "内容块", task.content_blocks if task.content_blocks is not None else "—"
    )
    right.metric(
        "处理耗时",
        f"{task.duration_ms / 1000:.2f} 秒" if task.duration_ms is not None else "—",
    )
    st.caption(f"文件：{task.file_name}")
    if task.document_id:
        st.code(task.document_id, language=None)
    if task.content_types:
        st.dataframe(
            [
                {"内容类型": content_type, "数量": count}
                for content_type, count in task.content_types.items()
            ],
            hide_index=True,
            use_container_width=True,
        )
    if task.status == "failed":
        st.error(task.error or "文档处理失败")


def _render_service_status(client: APIClient) -> PublicConfigView | None:
    st.sidebar.header("服务状态")
    try:
        health = client.health()
        config = client.public_config()
    except UIError as exc:
        st.sidebar.error(exc.message)
        st.sidebar.caption("请先启动 FastAPI：uvicorn app.main:app --workers 1")
        return None

    if health.status == "ok" and health.rag_initialized:
        st.sidebar.success("API 与 RAG 已就绪")
    else:
        st.sidebar.warning("API 在线，RAG 尚未就绪")
    st.sidebar.caption(
        f"解析器：{health.parser} / {config.parser_backend}  ·  "
        f"检索：{config.query_mode}"
    )
    st.sidebar.write("支持格式：" + "、".join(config.supported_extensions))
    st.sidebar.write(f"最大文件：{config.max_upload_mb} MB")
    return config


def _render_upload(
    client: APIClient, settings: UISettings, config: PublicConfigView | None
) -> None:
    st.header("1. 上传知识文档")
    st.write("支持 PDF、Markdown 和图片。服务端会解析文本、图片、表格与公式。")
    uploaded = st.file_uploader(
        "选择文件",
        type=[item.lstrip(".") for item in DEFAULT_EXTENSIONS],
        accept_multiple_files=False,
    )
    if st.button("开始处理", type="primary", disabled=uploaded is None):
        if uploaded is None:
            return
        try:
            supported = (
                config.supported_extensions if config else list(DEFAULT_EXTENSIONS)
            )
            max_mb = config.max_upload_mb if config else 20
            content = uploaded.getvalue()
            validate_upload(
                uploaded.name,
                len(content),
                supported_extensions=supported,
                max_upload_mb=max_mb,
            )
            with st.spinner("正在上传文档……"):
                task = client.upload_document(
                    uploaded.name,
                    content,
                    uploaded.type or "application/octet-stream",
                )
            status_box = st.empty()

            def show_progress(current: DocumentTaskView) -> None:
                status_box.info(f"任务状态：{_task_status_text(current.status)}")

            task = client.poll_document(
                task.task_id,
                timeout=settings.task_timeout,
                interval=settings.poll_interval,
                on_update=show_progress,
            )
            st.session_state.recent_task = task.model_dump(mode="json")
            if task.status == "completed":
                status_box.success("文档解析和索引完成")
            else:
                status_box.error(task.error or "文档处理失败")
        except UIError as exc:
            st.error(exc.message)

    if st.session_state.recent_task:
        try:
            _render_task(DocumentTaskView.model_validate(st.session_state.recent_task))
        except ValueError:
            st.session_state.recent_task = None
            st.warning("最近任务数据已失效，请重新上传")


def _render_query(client: APIClient) -> None:
    st.header("2. 多模态知识问答")
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            st.caption(
                f"模式：{entry['mode']} · 耗时：{entry['duration_ms']} ms · "
                f"来源：{sources_text(entry['sources'])}"
            )

    with st.form("query_form"):
        suggestion = st.selectbox("推荐问题", ("自定义问题", *SUGGESTED_QUESTIONS))
        question = st.text_input(
            "问题", placeholder="例如：表格中 Sensor-B 的延迟是多少？"
        )
        submitted = st.form_submit_button("开始问答", type="primary")

    if submitted:
        final_question = question.strip() or (
            suggestion if suggestion != "自定义问题" else ""
        )
        try:
            with st.spinner("正在执行混合检索……"):
                result = client.query(final_question)
            st.session_state.chat_history.append(
                {
                    "question": final_question,
                    "answer": result.answer,
                    "sources": result.sources,
                    "duration_ms": result.duration_ms,
                    "mode": result.mode,
                }
            )
            st.rerun()
        except UIError as exc:
            st.error(exc.message)


def render_app() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    st.set_page_config(
        page_title="KnowFlow 多模态知识库",
        page_icon="🔎",
        layout="wide",
    )
    _initialize_state()
    st.title("KnowFlow 多模态知识库")
    st.caption("真实文档解析 · 多模态索引 · 图谱与向量混合检索")

    try:
        settings = UISettings.from_env()
        client = get_api_client(settings.api_base_url, settings.request_timeout)
    except (ValueError, UIError):
        st.error("前端配置无效，请检查 API_BASE_URL 和 UI 超时设置")
        return

    config = _render_service_status(client)
    _render_upload(client, settings, config)
    st.divider()
    _render_query(client)


render_app()
