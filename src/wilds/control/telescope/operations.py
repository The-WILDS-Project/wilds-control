"""Watching a telescope command through to its outcome (plan §3.12).

A command is sent and nobody approves it: the TCS acts on it while it accepts external commands,
and checks it is valid and within the telescope's range. Whether it was carried out is read from
the telemetry, never assumed from having sent it. The job's message follows

    sent  ->  in progress  ->  done

or ends as ``refused`` (the TCS did not take the command up: it was invalid or out of range, or
the TCS is not accepting external commands) or ``timeout`` (taken up but the telescope did not
arrive).
"""

import time
from collections.abc import Callable

from ..errors import ApiError
from ..jobs import JobContext
from .base import Telemetry, TelescopeBase

POLL_S = 0.25
# After a command is sent, the telemetry should show the TCS taking it up within this long; if
# not, the TCS refused it (or it never arrived).
EFFECT_GRACE_S = 3.0


async def confirm(
    ctx: JobContext,
    telescope: TelescopeBase,
    done: Callable[[Telemetry], bool],
    what: str,
    *,
    took_effect: Callable[[Telemetry], bool] | None = None,
) -> None:
    """Follow a command until ``done``. ``took_effect`` says the TCS has taken the command up: if
    it has not after a few seconds, the TCS refused it (or it never arrived). Omit it for a
    command that shows no such sign."""
    conf = telescope.conf.current
    started = time.monotonic()
    ctx.progress(0.05, "sent")
    while True:
        ctx.check()
        t = telescope.telemetry()
        now = time.monotonic()
        if done(t):
            ctx.progress(1.0, "done")
            return
        if took_effect is not None and not took_effect(t) and now - started > EFFECT_GRACE_S:
            if t.accepts_commands is False:
                why = "the TCS is not accepting external commands"
            else:
                why = "the TCS did not take it up (invalid, or out of range?)"
            raise ApiError(409, "refused", f"{what} was refused: {why}")
        if now - started > conf.motion_timeout_s:
            raise ApiError(
                409,
                "timeout",
                f"the telescope did not arrive within {conf.motion_timeout_s:g} s",
            )
        ctx.progress(0.5, "in progress")
        await ctx.sleep(POLL_S)
