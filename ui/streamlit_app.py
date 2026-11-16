"""Single-paper multimodal reading workspace."""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from app.schemas.papers import PaperEvidence, PaperReadingRequest, PaperReadingResponse
from ui.client import APIClient, UIError, UISettings, validate_upload
from ui.paper_tasks import render_task_result, render_upload_tracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def get_api_client(base_url: str, timeout: float) -> APIClient:
    return APIClient(base_url, timeout)


def render_evidence(client: APIClient, item: PaperEvidence, label: str = "") -> None:
    page = f"第 {item.page} 页" if item.page else "页码未知"
    if item.page and item.page_end:
        page = f"第 {item.page}–{item.page_end} 页（跨页段落）"
    with st.expander(f"{label} {page} · {item.modality} · {item.section[:70]}"):
        st.text(item.text)
        if item.image_url:
            try:
                st.image(client.paper_asset(item), caption=page)
            except UIError:
                st.warning("原图暂不可读取，仍可查看文字证据。")
        st.caption(f"证据 ID：{item.evidence_id}")


def render_answer(client: APIClient, result: PaperReadingResponse) -> None:
    st.markdown(result.answer)
    st.caption(
        f"耗时 {result.duration_ms} ms · 检索方式 {result.retrieval} · "
        + ("本次使用原图与文字" if result.vision_used else "本次使用文字证据")
    )
    for warning in result.warnings:
        st.warning(warning)
    if result.status == "uncited":
        st.warning("回答未给出有效引用，请核对检索片段。")
    if result.status == "no_evidence":
        st.info("请尝试论文中的原文术语或图表标题。")
    st.caption("已引用表示模型引用了该片段，不代表系统已自动核验结论。")
    for index, item in enumerate(result.evidence, 1):
        label = f"[E{index}] " + (
            "已引用" if f"E{index}" in result.citations else "检索片段"
        )
        render_evidence(client, item, label)


def render_app() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    st.set_page_config(
        page_title="KnowFlow 论文阅读助手", page_icon="📚", layout="wide"
    )
    st.title("KnowFlow 论文阅读助手")
    st.caption("按论文检索 · 图表与公式理解 · 阅读概览 · 页码与原图引用")
    try:
        settings = UISettings.from_env()
        client = get_api_client(settings.api_base_url, settings.request_timeout)
        health = client.health()
        config = client.public_config()
    except (ValueError, UIError) as exc:
        st.sidebar.error(exc.message if isinstance(exc, UIError) else "前端配置无效")
        st.info(
            "请先启动 API：uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1"
        )
        return

    st.sidebar.header("论文库")
    if health.rag_initialized:
        st.sidebar.success("API 与解析服务已就绪")
    else:
        st.sidebar.warning("解析服务尚未就绪")
    st.sidebar.caption("论文库可跨重启保留；上传任务状态保存在当前服务进程。")
    uploaded = st.sidebar.file_uploader("上传论文 PDF 或 Markdown", type=["pdf", "md"])
    if st.sidebar.button("加入论文库", type="primary", disabled=uploaded is None):
        try:
            assert uploaded is not None
            data = uploaded.getvalue()
            validate_upload(
                uploaded.name,
                len(data),
                supported_extensions=[".pdf", ".md"],
                max_upload_mb=config.max_upload_mb,
            )
            task = client.upload_paper(
                uploaded.name, data, uploaded.type or "application/octet-stream"
            )
            st.session_state["paper_task_id"] = task.task_id
            st.session_state["paper_task_started"] = time.monotonic()
            st.session_state.pop("paper_task_result", None)
            st.sidebar.success("上传成功，正在后台解析与索引。")
        except UIError as exc:
            st.sidebar.error(exc.message)

    with st.sidebar:
        if st.session_state.get("paper_task_id"):
            render_upload_tracker(client, settings.task_timeout)
        render_task_result()
    st.sidebar.button("刷新任务与论文库")
    try:
        papers = client.papers()
    except UIError as exc:
        st.error(exc.message)
        return
    if not papers:
        st.info(
            "论文库为空。请上传论文，处理状态会自动更新。PDF 首次解析可能需要下载模型。"
        )
        return
    by_id = {paper.paper_id: paper for paper in papers}
    paper_id = st.selectbox(
        "当前论文", list(by_id), format_func=lambda value: by_id[value].title
    )
    paper = by_id[paper_id]
    st.caption(
        f"{paper.file_name} · {paper.block_count} 个证据块 · "
        + ("向量索引已建立" if paper.vector_indexed else "关键词索引")
    )
    left, right = st.columns(2)
    with left:
        st.subheader("原文与图文内容")
        if st.button("准备原文下载"):
            try:
                st.session_state["paper_download"] = (
                    paper_id,
                    client.paper_original(paper_id),
                )
            except UIError as exc:
                st.error(exc.message)
        download = st.session_state.get("paper_download")
        if download and download[0] == paper_id:
            st.download_button("下载原文", download[1], file_name=paper.file_name)
        batch = st.number_input(
            "内容批次（每批 20 块）",
            min_value=1,
            max_value=max(1, (paper.block_count + 19) // 20),
            value=1,
            step=1,
            key=f"batch_{paper_id}",
        )
        st.caption("页码使用 PDF 文件页序，从 1 开始；Markdown 或缺失页码显示未知。")
        try:
            for item in client.paper_content(paper_id, (int(batch) - 1) * 20):
                render_evidence(client, item)
        except UIError as exc:
            st.error(exc.message)
    with right:
        st.subheader("论文阅读")
        with st.form("paper_reading"):
            task_name = st.selectbox("阅读任务", ["针对问题阅读", "结构化概览"])
            question = st.text_area(
                "问题或概览关注点",
                placeholder="例如：请结合 Figure 2 和正文解释方法流程。",
            )
            use_images = st.checkbox("检索到图表时读取原图（最多三张）", value=True)
            top_k = st.slider("主要检索片段数", 2, 10, 5)
            submitted = st.form_submit_button("开始阅读", type="primary")
        if submitted:
            try:
                payload = PaperReadingRequest(
                    paper_id=paper_id,
                    question=question,
                    task="question" if task_name == "针对问题阅读" else "overview",
                    use_images=use_images,
                    top_k=top_k,
                )
                with st.spinner("正在检索证据并生成回答……"):
                    result = client.paper_read(payload)
                st.session_state["paper_answer"] = (
                    paper_id,
                    result.model_dump(),
                    payload.model_dump(),
                )
            except ValueError:
                st.error("请先输入问题；结构化概览可以不填写关注点。")
            except UIError as exc:
                st.error(exc.message)
        with st.expander("阅读历史（跨重启保留）"):
            history_page = st.number_input(
                "历史页码（每页20条）",
                min_value=1,
                value=1,
                step=1,
                key=f"history_page_{paper_id}",
            )
            try:
                history = client.paper_readings(paper_id, (int(history_page) - 1) * 20)
                by_history = {item.history_id: item for item in history}
                selected = st.selectbox(
                    "已保存的阅读",
                    [None, *by_history],
                    format_func=lambda value: "请选择记录"
                    if value is None
                    else (
                        f"{by_history[value].created_at} · "
                        + (by_history[value].question[:80] or "结构化概览")
                    ),
                    key=f"history_{paper_id}_{history_page}",
                )
                if not history:
                    st.caption("这一页暂无阅读记录。")
                if st.button("载入记录", disabled=selected is None):
                    record = client.paper_reading(paper_id, selected)
                    st.session_state["paper_answer"] = (
                        paper_id,
                        record.response.model_dump(),
                        record.request.model_dump(),
                    )
            except UIError as exc:
                st.warning(exc.message)
        saved = st.session_state.get("paper_answer")
        if saved and saved[0] == paper_id:
            if len(saved) > 2:
                st.caption("本次问题：" + (saved[2]["question"] or "结构化概览"))
            render_answer(client, PaperReadingResponse.model_validate(saved[1]))
            st.download_button(
                "导出本次阅读记录",
                data=json.dumps(
                    {
                        "schema_version": 1,
                        "paper_title": paper.title,
                        "request": saved[2] if len(saved) > 2 else None,
                        "response": saved[1],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="paper_reading.json",
                mime="application/json",
            )


render_app()
