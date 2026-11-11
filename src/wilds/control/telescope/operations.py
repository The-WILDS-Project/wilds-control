"""Watching a telescope command through to its outcome (plan §3.12).

A command is sent; whether it was accepted (with External Command Preview on, the TO has to
accept it in the TCS) and whether it has been carried out is read from the telemetry, never
assumed from having sent it. The job's message follows the states

    sent  ->  waiting for the TO  ->  in progress  ->  done

or ends as ``refused`` ("not accepted": the TO did not accept it in time, or rejected it: the
command stopped waiting but never took effect) or ``timeout`` (accepted but the telescope did
not arrive). Cancelling takes back a command still waiting for the TO.
"""

import time
from collections.abc import Callable

from ..errors import ApiError
from ..jobs import JobCancelled, JobContext
from .base import Telemetry, TelescopeBase

POLL_S = 0.25
# After a command stops waiting for the TO, the telemetry should show it taking effect within
# this long; if not, the TO rejected it.
EFFECT_GRACE_S = 3.0


async def confirm(
    ctx: JobContext,
    telescope: TelescopeBase,
    done: Callable[[Telemetry], bool],
    what: str,
    *,
    took_effect: Callable[[Telemetry], bool] | None = None,
) -> None:
    """Follow a command until ``done``. ``took_effect`` says the TCS has taken the command up: if a
    command is not waiting for the TO and still has not taken effect after a few seconds, the TO
    rejected it (or it never arrived). Omit it for a command that shows no such sign."""
    conf = telescope.conf.current
    started = time.monotonic()
    accepted_at: float | None = None
    left_preview: float | None = None
    ctx.progress(0.05, "sent")
    while True:
        try:
            ctx.check()
        except JobCancelled:
            telescope.withdraw()
            raise
        t = telescope.telemetry()
        now = time.monotonic()
        if t.pending is not None:
            left_preview = None  # still waiting for the TO
            if now - started > conf.accept_timeout_s:
                telescope.withdraw()
                raise ApiError(
                    409,
                    "refused",
                    f"{what} was not accepted by the TO within {conf.accept_timeout_s:g} s",
                    {"waited_s": round(now - started)},
                )
            ctx.progress(0.1, f"waiting for the TO to accept {t.pending.what}")
        elif done(t):
            ctx.progress(1.0, "done")
            return
        else:
            if took_effect is not None and not took_effect(t):
                left_preview = left_preview or now
                if now - left_preview > EFFECT_GRACE_S:
                    raise ApiError(409, "refused", f"{what} was rejected by the TO")
            accepted_at = accepted_at or now
            if now - accepted_at > conf.motion_timeout_s:
                raise ApiError(
                    409,
                    "timeout",
                    f"the telescope did not arrive within {conf.motion_timeout_s:g} s",
                )
            ctx.progress(0.5, "in progress")
        try:
            await ctx.sleep(POLL_S)
        except JobCancelled:
            telescope.withdraw()
            raise
