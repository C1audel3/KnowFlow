"""Stable application errors that are safe to expose over HTTP."""

from __future__ import annotations


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class InvalidFileTypeError(AppError):
    def __init__(self) -> None:
        super().__init__("INVALID_FILE_TYPE", "不支持的文件类型", 415)


class InvalidFileError(AppError):
    def __init__(self, message: str = "文件内容无效") -> None:
        super().__init__("INVALID_FILE", message, 400)


class FileTooLargeError(AppError):
    def __init__(self) -> None:
        super().__init__("FILE_TOO_LARGE", "文件超过大小限制", 413)


class TaskNotFoundError(AppError):
    def __init__(self) -> None:
        super().__init__("DOCUMENT_NOT_FOUND", "文档任务不存在", 404)


class RAGNotReadyError(AppError):
    def __init__(self) -> None:
        super().__init__("RAG_NOT_READY", "RAG 服务尚未初始化", 503)


class ModelUnavailableError(AppError):
    def __init__(self) -> None:
        super().__init__("MODEL_UNAVAILABLE", "模型服务暂时不可用", 503)
