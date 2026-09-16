"""Exercise the Streamlit page against a running real phase-three API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = "Atlas 项目计划在哪个季度发布？"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--expected-keyword", default="2027 Q3")
    parser.add_argument("--timeout", type=float, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["API_BASE_URL"] = args.api_url
    os.environ["UI_REQUEST_TIMEOUT"] = str(args.timeout)
    app = AppTest.from_file(str(PROJECT_ROOT / "ui" / "streamlit_app.py")).run(
        timeout=30
    )
    if app.exception:
        raise RuntimeError("Streamlit initial render failed")
    if not app.sidebar.success:
        raise RuntimeError("Streamlit did not detect a ready API")

    options = app.selectbox[0].options
    if args.question in options:
        app.selectbox[0].select(args.question)
    else:
        app.text_input[0].input(args.question)
    query_button = next(button for button in app.button if button.label == "开始问答")
    query_button.click()
    app.run(timeout=args.timeout)
    if app.exception:
        raise RuntimeError("Streamlit query render failed")

    history = app.session_state["chat_history"]
    if not history:
        raise RuntimeError("Streamlit did not store a query response")
    result = history[-1]
    if args.expected_keyword.lower() not in result["answer"].lower():
        raise RuntimeError("Streamlit answer did not contain the expected keyword")

    print(
        json.dumps(
            {
                "api_ready": True,
                "question": result["question"],
                "answer": result["answer"],
                "mode": result["mode"],
                "duration_ms": result["duration_ms"],
                "sources": result["sources"],
                "passed": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
