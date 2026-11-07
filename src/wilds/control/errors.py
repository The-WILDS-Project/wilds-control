"""The one error shape (plan §3.14).

HTTP carries the outcome (status code); the body is always ``{code, message, details}``, so a
caller never parses text. The same body is a failed job's ``error``.
"""

from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ErrorCode = Literal[
    "lease_required",  # 403
    "not_found",  # 404
    "busy",  # 409
    "invalid",  # 422
    "refused",  # the TCS or a device said no
    "timeout",
    "cancelled",
    "internal",
]


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str  # for people
    details: dict[str, Any] | None = None  # structured, e.g. the blocking job_id


class ApiError(Exception):
    """Raise anywhere (a handler, a job) to answer with ``status`` and an :class:`ErrorBody`."""

    def __init__(
        self, status: int, code: ErrorCode, message: str, details: dict[str, Any] | None = None
    ):
        super().__init__(message)
        self.status = status
        self.body = ErrorBody(code=code, message=message, details=details)


def install(app: FastAPI) -> None:
    async def handle(_: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, ApiError)
        return JSONResponse(exc.body.model_dump(), status_code=exc.status)

    app.add_exception_handler(ApiError, handle)
