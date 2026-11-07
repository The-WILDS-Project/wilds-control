"""Sessions and the control lease (plan §3.11)."""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from .. import config, provenance
from ..lease import LeaseError, Session, state

router = APIRouter(tags=["control"])


# -- dependencies -----------------------------------------------------------------------


def _http(e: LeaseError) -> HTTPException:
    return HTTPException(status_code=e.status, detail=e.message)


def current_session(x_session_id: Annotated[str | None, Header()] = None) -> Session:
    try:
        return state.get(x_session_id)
    except LeaseError as e:
        raise _http(e) from e


def require_lease(x_lease_token: Annotated[str | None, Header()] = None) -> None:
    """Dependency for every command and configuration change: 403 unless the lease token is valid."""
    try:
        state.check_token(x_lease_token)
    except LeaseError as e:
        raise _http(e) from e


def require_engineering(x_lease_token: Annotated[str | None, Header()] = None) -> None:
    """For WILDS mechanisms and calibration: the lease must be held in engineering mode."""
    require_lease(x_lease_token)
    if not state.engineering:
        raise HTTPException(403, "Engineering mode required")


def require_to(x_to_passphrase: Annotated[str | None, Header()] = None) -> None:
    """A valid TO request also shows the TO is online (their page polls with the passphrase)."""
    if not config.check_to(x_to_passphrase):
        raise HTTPException(403, "Telescope Operator passphrase required")
    state.touch_to()


Me = Annotated[Session, Depends(current_session)]
ToOnly = Depends(require_to)


# -- models -----------------------------------------------------------------------------


class NewSession(BaseModel):
    name: str
    hints: provenance.Hints = provenance.Hints()


class SessionCreated(BaseModel):
    session_id: str
    code: str
    provenance: str  # plain-text tooltip lines


class Heartbeat(BaseModel):
    name: str
    code: str
    holder: bool
    engineering: bool
    requested: bool
    refused: bool = False  # the TO refused your last request; asking again clears it
    refusal_reason: str | None = None
    provenance: str
    lease_token: str | None = None  # only the holder is given it


class Engineering(BaseModel):
    passphrase: str


class Refusal(BaseModel):
    reason: str = ""


class SessionInfo(BaseModel):
    code: str
    name: str
    requested: bool
    provenance: str  # plain-text tooltip lines
    details: provenance.Provenance | None = None


class SessionsView(BaseModel):
    holder: SessionInfo | None
    engineering: bool
    requests: list[SessionInfo]
    viewers: list[SessionInfo]


def _info(s: Session) -> SessionInfo:
    return SessionInfo(
        code=s.code,
        name=s.name,
        requested=s.requested_at is not None,
        provenance=s.prov.summary() if s.prov else "",
        details=s.prov,
    )


# -- sessions ---------------------------------------------------------------------------


@router.post("/sessions", response_model=SessionCreated)
def create_session(body: NewSession, request: Request) -> SessionCreated:
    client = request.client
    server = request.scope.get("server")
    prov = provenance.collect(
        ip=client.host if client else "unknown",
        port=client.port if client else None,
        server_port=server[1] if server else None,
        user_agent=request.headers.get("user-agent"),
        hints=body.hints,
    )
    try:
        s = state.create_session(body.name, prov)
    except LeaseError as e:
        raise _http(e) from e
    return SessionCreated(session_id=s.id, code=s.code, provenance=prov.summary())


@router.post("/sessions/{session_id}/heartbeat", response_model=Heartbeat)
def heartbeat(session_id: str, request: Request) -> Heartbeat:
    try:
        s = state.get(session_id)
    except LeaseError as e:
        raise _http(e) from e
    if s.prov and request.client:
        provenance.moved(s.prov, request.client.host)
    holder = state.holder_id == s.id
    return Heartbeat(
        name=s.name,
        code=s.code,
        holder=holder,
        engineering=holder and state.engineering,
        requested=s.requested_at is not None,
        refused=s.refused,
        refusal_reason=s.refusal_reason,
        provenance=s.prov.summary() if s.prov else "",
        lease_token=state.token if holder else None,
    )


# -- requests and release ---------------------------------------------------------------


@router.post("/control/requests", status_code=202)
def request_control(me: Me) -> dict[str, str]:
    state.request(me)
    return {"code": me.code}


@router.delete("/control")
def release_control(me: Me) -> dict[str, bool]:
    try:
        state.release(me)
    except LeaseError as e:
        raise _http(e) from e
    return {"released": True}


@router.post("/control/engineering")
def engineering(body: Engineering, me: Me) -> dict[str, bool]:
    if not config.check_engineering(body.passphrase):
        raise HTTPException(403, "Wrong engineering passphrase")
    state.take_engineering(me)
    return {"engineering": True}


# -- Telescope Operator only ------------------------------------------------------------


@router.get("/control/sessions", response_model=SessionsView, dependencies=[ToOnly])
def list_sessions() -> SessionsView:
    state.expire()
    holder = state.holder
    return SessionsView(
        holder=_info(holder) if holder else None,
        engineering=state.engineering,
        requests=[_info(s) for s in state.requests()],
        viewers=[_info(s) for s in state.sessions.values() if s.requested_at is None],
    )


@router.post("/control/requests/{code}/grant", dependencies=[ToOnly])
def grant(code: str) -> dict[str, str]:
    try:
        s = state.grant(code)
    except LeaseError as e:
        raise _http(e) from e
    return {"holder": s.name}


@router.post("/control/requests/{code}/refuse", dependencies=[ToOnly])
def refuse(code: str, body: Refusal) -> dict[str, bool]:
    try:
        state.refuse(code, body.reason)
    except LeaseError as e:
        raise _http(e) from e
    return {"refused": True}


@router.post("/control/revoke", dependencies=[ToOnly])
def revoke() -> dict[str, bool]:
    state.revoke()
    return {"revoked": True}
