"""The one error shape (plan §3.14).

HTTP carries the outcome (status code); the body is always ``{code, message, details}``, so a
caller never parses text. The same body is a failed job's ``error``.
"""

from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

ErrorCode = Literal[
    "lease_required",  # 403
    "not_found",  # 404
    "busy",  # 409
    "stale",  # 409: the settings changed since the caller loaded them
    "invalid",  # 422
    "conflict",  # 409: the request does not fit the state (already done, nothing to cancel)
    "light_not_blocked",  # 409: a dark needs the light path blocked (plan §3.17)
    "not_operational",  # 409: the component is not started up or cannot take tasks now
    "cooler_not_stable",  # 409: a dark needs the sensor at its setpoint
    "solve_failed",  # 409: the plate solver found no match (or has no index files)
    "slit_not_found",  # 409: no gap in the difference of the closed and open frames
    "not_calibrated",  # 409: a calibration value this needs has not been measured yet
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

    async def handle_validation(_: Request, exc: Exception) -> JSONResponse:
        # FastAPI's own 422 (a body or parameter that does not parse) gets the same shape.
        assert isinstance(exc, RequestValidationError)
        errors = [
            {"where": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()
        ]
        first = errors[0] if errors else {"where": "request", "message": "invalid"}
        body = ErrorBody(
            code="invalid",
            message=f"{first['where']}: {first['message']}",
            details={"errors": errors},
        )
        return JSONResponse(body.model_dump(), status_code=422)

    async def handle_model(_: Request, exc: Exception) -> JSONResponse:
        # A settings change that fails its model's limits (conf.patch), raised inside a handler.
        assert isinstance(exc, ValidationError)
        errors = [
            {"where": ".".join(str(p) for p in e["loc"]), "message": e["msg"]} for e in exc.errors()
        ]
        first = errors[0] if errors else {"where": "request", "message": "invalid"}
        body = ErrorBody(
            code="invalid",
            message=f"{first['where']}: {first['message']}",
            details={"errors": errors},
        )
        return JSONResponse(body.model_dump(), status_code=422)

    app.add_exception_handler(ApiError, handle)
    app.add_exception_handler(RequestValidationError, handle_validation)
    app.add_exception_handler(ValidationError, handle_model)
