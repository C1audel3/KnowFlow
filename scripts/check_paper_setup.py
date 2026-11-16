"""Check local paper-demo prerequisites without uploading files or invoking models."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from app.core.config import PROJECT_ROOT, AppSettings
from ui.client import APIClient, UIError, UISettings

PACKAGES = (
    "fastapi",
    "uvicorn",
    "httpx",
    "streamlit",
    "python-dotenv",
    "openai",
    "mineru",
    "lightrag-hku",
    "Pillow",
    "reportlab",
)


def local_checks() -> list[dict]:
    checks = []

    def add(name, ok, message):
        checks.append(
            {"name": name, "status": "pass" if ok else "fail", "message": message}
        )

    add("python", (3, 10) <= sys.version_info[:2] < (3, 13), "需要 Python 3.10–3.12。")
    for package in PACKAGES:
        try:
            installed = version(package)
            add(f"package:{package}", True, f"已安装 {installed}")
        except PackageNotFoundError:
            add(f"package:{package}", False, "缺少依赖，请安装项目演示依赖。")
    try:
        settings = AppSettings.from_env()
        UISettings.from_env()
    except ValueError:
        add(
            "configuration",
            False,
            "应用或页面配置无效，检查正整数、文件类型和 API_BASE_URL。",
        )
    else:
        add("configuration", True, "应用与页面配置可解析。")
        add(
            "paper_extensions",
            {".pdf", ".md"} <= settings.supported_extensions,
            "论文演示需要启用 .pdf 和 .md。",
        )
        for name, directory in (
            ("storage", settings.working_dir),
            ("uploads", settings.upload_dir),
            ("output", settings.output_dir),
        ):
            existing = directory
            while not existing.exists() and existing != existing.parent:
                existing = existing.parent
            add(
                f"directory:{name}",
                existing.is_dir() and os.access(existing, os.W_OK | os.X_OK),
                "目录或最近的已存在父目录需可写；本检查不创建文件。",
            )
    for key in (
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "LLM_BINDING_HOST",
        "EMBEDDING_BINDING_HOST",
        "LLM_BINDING_API_KEY",
        "EMBEDDING_BINDING_API_KEY",
    ):
        value = os.getenv(key, "").strip()
        valid = bool(value) and not value.casefold().startswith(
            ("replace-", "your-", "<")
        )
        add(
            f"configured:{key}",
            valid,
            "已配置。" if valid else "尚未填写或仍为示例占位值。",
        )
    for label, relative in (
        ("sample", "data/samples/papers/demo_paper.pdf"),
        ("dataset", "data/evaluation/paper_demo.jsonl"),
    ):
        add(
            label,
            (PROJECT_ROOT / relative).is_file(),
            "原创演示材料需存在；可运行生成脚本准备。",
        )
    return checks


def api_checks(client: APIClient) -> list[dict]:
    try:
        health = client.health()
        config = client.public_config()
        papers = client.papers()
    except UIError:
        # Never echo exception details, endpoint URLs, credentials or response bodies.
        return [
            {
                "name": "api",
                "status": "fail",
                "message": "API不可连接或响应不兼容，请检查服务启动与配置。",
            }
        ]
    return [
        {
            "name": "api",
            "status": "pass"
            if health.rag_initialized and health.status == "ok"
            else "fail",
            "message": "API初始化状态检查。",
        },
        {
            "name": "api_paper_extensions",
            "status": "pass"
            if {".pdf", ".md"} <= set(config.supported_extensions)
            else "fail",
            "message": "服务端需支持论文演示文件类型。",
        },
        {
            "name": "catalogue",
            "status": "pass",
            "message": f"论文目录接口可读取，当前有 {len(papers)} 篇。",
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", help="额外检查运行中的API；仅发送只读请求")
    parser.add_argument("--output", type=Path, help="保存不含密钥和服务地址的JSON结果")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    checks = local_checks()
    if args.api_url:
        try:
            client = APIClient(args.api_url, timeout=5)
        except ValueError:
            checks.append(
                {"name": "api", "status": "fail", "message": "API地址格式无效。"}
            )
        else:
            try:
                checks.extend(api_checks(client))
            finally:
                client.close()
    result = {
        "scope": "Configuration, installed package metadata, directory permissions and optional read-only API checks. No model calls or parser execution; not a clean-install or end-to-end verification.",
        "passed": all(item["status"] == "pass" for item in checks),
        "api_checked": bool(args.api_url),
        "model_calls": 0,
        "checks": checks,
        "unverified": [
            "模型可达性、鉴权和视觉能力",
            "embedding实际维度",
            "MinerU模型缓存及PDF解析",
            "新环境完整安装",
        ],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
