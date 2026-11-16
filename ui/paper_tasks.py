"""Nonblocking periodic upload tracking; terminal states trigger one catalogue refresh."""

import time

import streamlit as st

from ui.client import APIClient, UIError


def finish_tracking(result: dict) -> None:
    st.session_state["paper_task_result"] = result
    st.session_state.pop("paper_task_id", None)
    st.rerun()


@st.fragment(run_every=2)
def render_upload_tracker(client: APIClient, timeout: float) -> None:
    task_id = st.session_state.get("paper_task_id")
    if not task_id:
        return
    started = st.session_state.setdefault("paper_task_started", time.monotonic())
    if time.monotonic() - started >= timeout:
        finish_tracking({"status": "tracking_paused", "task_id": task_id})
        return
    try:
        task = client.document_status(task_id)
    except UIError as exc:
        if exc.code in {"DOCUMENT_NOT_FOUND", "HTTP_404"}:
            finish_tracking(
                {
                    "status": "expired",
                    "message": "任务状态已失效，请查看论文库确认是否完成；未完成时可重新上传。",
                }
            )
        else:
            st.warning(exc.message + "；将继续自动查询。")
        return
    if task.status in {"completed", "failed"}:
        finish_tracking(task.model_dump())
        return
    st.info(f"{task.file_name}：正在处理，每 2 秒自动更新。")


def render_task_result() -> None:
    result = st.session_state.get("paper_task_result") or {}
    status = result.get("status")
    if status == "completed":
        st.success(f"{result['file_name']} 已加入论文库。")
    elif status == "failed":
        st.error(result.get("error") or "论文处理失败，请检查文件后重新上传。")
        st.caption("可通过上方“加入论文库”重新提交。")
    elif status == "expired":
        st.warning(result["message"])
    elif status == "tracking_paused":
        st.warning("等待较久，自动跟踪已暂停；后台处理可能仍在进行。")
        if st.button("继续跟踪任务"):
            st.session_state["paper_task_id"] = result["task_id"]
            st.session_state["paper_task_started"] = time.monotonic()
            st.session_state.pop("paper_task_result", None)
            st.rerun()
