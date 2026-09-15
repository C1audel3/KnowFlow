"""Environment-backed configuration for the single-process demo API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTENSIONS = frozenset({".pdf", ".md", ".png", ".jpg", ".jpeg"})


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _project_path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Validated non-secret application settings."""

    host: str
    port: int
    working_dir: Path
    upload_dir: Path
    output_dir: Path
    max_upload_bytes: int
    supported_extensions: frozenset[str]
    parser_backend: str
    parser_timeout: int
    max_query_concurrency: int

    @classmethod
    def from_env(cls) -> AppSettings:
        extensions = frozenset(
            item.strip().lower()
            for item in os.getenv(
                "SUPPORTED_FILE_EXTENSIONS", ".pdf,.md,.png,.jpg,.jpeg"
            ).split(",")
            if item.strip()
        )
        if not extensions or any(not item.startswith(".") for item in extensions):
            raise ValueError("SUPPORTED_FILE_EXTENSIONS must contain dotted extensions")
        unknown = extensions - DEFAULT_EXTENSIONS
        if unknown:
            raise ValueError(f"unsupported configured extensions: {sorted(unknown)}")

        return cls(
            host=os.getenv("APP_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=_positive_int("APP_PORT", 8000),
            working_dir=_project_path("WORKING_DIR", "./rag_storage_app"),
            upload_dir=_project_path("INPUT_DIR", "./data/uploads"),
            output_dir=_project_path("OUTPUT_DIR", "./output"),
            max_upload_bytes=_positive_int("MAX_UPLOAD_MB", 20) * 1024 * 1024,
            supported_extensions=extensions,
            parser_backend=os.getenv("PARSER_BACKEND", "pipeline").strip()
            or "pipeline",
            parser_timeout=_positive_int("PARSER_TIMEOUT", 900),
            max_query_concurrency=_positive_int("MAX_QUERY_CONCURRENCY", 2),
        )

    def public_dict(self) -> dict[str, object]:
        return {
            "max_upload_mb": self.max_upload_bytes // (1024 * 1024),
            "supported_extensions": sorted(self.supported_extensions),
            "parser": "mineru",
            "parser_backend": self.parser_backend,
            "query_mode": "mix",
            "max_query_concurrency": self.max_query_concurrency,
        }
