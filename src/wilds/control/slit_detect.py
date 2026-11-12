"""Find the slit in the guide camera's image (plan §3.3).

The slit is a gap in a reflective plate, so it is dark in the guide camera: the light that falls
in it goes on to the spectrograph. Stars and sky are in both a frame with the slit **closed** and
one with it **open**, and only the gap differs, so the closed frame minus the open one is the gap
alone: stars cancel (what is left of them is their shot noise), and the gap stands out as a
bright strip whatever the sky is doing. That is more robust than looking for a dark strip in one
frame, where a star in the gap or a star just beside it looks like an edge.

``measure_slit`` takes the two frames (same settings, taken one after the other) and returns
where the gap is, its direction and its size, all in the pixels of the frames given. The caller
maps them to the unbinned sensor and the sky (``plate_scale``). Nothing here knows about the
camera, the stage or FastAPI.

Angles are counter-clockwise on the screen from the top of the image (the convention of the
calibration and of ``slit.js``); x runs right and y down. The gap's direction is a line, not an
arrow: the angle is given in (-90, 90].
"""

import math
from dataclasses import dataclass

import numpy as np


class SlitNotFound(ValueError):
    """There is no gap in the difference of the two frames (is the slit stage where it was
    asked to be, and is the light on?)."""


@dataclass(frozen=True)
class SlitMeasurement:
    x: float  # the centre of the gap, pixels of the frames given
    y: float
    angle_deg: float  # the long axis, counter-clockwise from the top of the image, (-90, 90]
    length_px: float
    width_px: float
    snr: float  # the gap's contrast over the pixel noise (one pixel, not the whole gap)
    significance: float  # how many sigma the whole gap stands out by in the fit


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur, separable, edges repeated (numpy only: scipy is not a dependency)."""
    r = max(1, math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(a, pad, mode="edge")
        out = np.zeros_like(a)
        n = a.shape[axis]
        for i, w in enumerate(k):
            out += w * (p[i : i + n, :] if axis == 0 else p[:, i : i + n])
        a = out
    return a


def _noise(a: np.ndarray) -> float:
    """The pixel noise of a mostly empty difference image (median absolute deviation)."""
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) or 1e-9


def _weights(closed: np.ndarray, open_: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """One over the noise variance of each pixel of the difference. Shot noise grows with the
    light, so pixels in and around stars are noisy and count for little; the relation (variance
    against the sum of the two frames) is measured from the frames themselves, so it holds
    whatever the gain."""
    s = _smooth(closed.astype(np.float64) + open_.astype(np.float64), 1.0)
    edges = np.quantile(s, np.linspace(0, 1, 13))
    mids, var = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (s >= lo) & (s <= hi)
        if sel.sum() > 100:
            mids.append(float(np.mean(s[sel])))
            var.append(_noise(raw[sel]) ** 2)
    a, b = np.polyfit(mids, var, 1) if len(mids) >= 3 else (0.0, _noise(raw) ** 2)
    a = max(float(a), 0.0)
    floor = max(float(np.min(var)) if var else 1.0, 1e-6)
    return 1.0 / np.maximum(a * s + b, floor)


def _axes(angle_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Unit vectors (x right, y down) along the gap and across it for an angle from the top."""
    a = math.radians(angle_deg)
    return (-math.sin(a), -math.cos(a)), (math.cos(a), -math.sin(a))


def _angle_of(vx: float, vy: float) -> float:
    """Angle from the top of the image, counter-clockwise, of a direction (x right, y down),
    as a line: in (-90, 90]."""
    if vy > 0 or (vy == 0 and vx < 0):
        vx, vy = -vx, -vy
    a = math.degrees(math.atan2(-vx, -vy))
    return 90.0 if a <= -90.0 else a


def _cover(d: np.ndarray, half: float) -> np.ndarray:
    """What a one-pixel-wide pixel at ``d`` from the centre has of a strip ``half`` each side."""
    return np.clip(np.minimum(d + 0.5, half) - np.maximum(d - 0.5, -half), 0.0, 1.0)


class _Fit:
    """A box (the gap) fitted to the difference image by weighted matched filtering: for a box
    shape the best brightness is linear, so the score is ``(sum w m r)^2 / sum w m^2`` for the
    model ``m``, the difference ``r`` and the weights ``w`` (one over the pixel's noise variance),
    and it is the shape (centre, angle, length, width) that is searched."""

    def __init__(self, raw: np.ndarray, weight: np.ndarray, x: float, y: float, reach: float):
        h, w = raw.shape
        x0, x1 = max(0, int(x - reach)), min(w, int(x + reach) + 1)
        y0, y1 = max(0, int(y - reach)), min(h, int(y + reach) + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        sel = weight[y0:y1, x0:x1] > 0
        self.xx, self.yy = xx[sel].astype(np.float64), yy[sel].astype(np.float64)
        self.r = raw[y0:y1, x0:x1][sel]
        self.w = weight[y0:y1, x0:x1][sel]

    def model(self, p: np.ndarray) -> np.ndarray:
        x, y, angle, length, width = p
        (ux, uy), (vx, vy) = _axes(angle)
        dx, dy = self.xx - x, self.yy - y
        return _cover(dx * ux + dy * uy, length / 2) * _cover(dx * vx + dy * vy, width / 2)

    def score(self, p: np.ndarray) -> float:
        if p[3] < 2 or p[4] < 0.3:
            return -1.0
        m = self.model(p)
        den = float(m @ (self.w * m))
        return float((m @ (self.w * self.r)) ** 2 / den) if den > 0 else -1.0

    def brightness(self, p: np.ndarray) -> float:
        m = self.model(p)
        return float(m @ (self.w * self.r) / (m @ (self.w * m)))

    def search(self, p0: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """Pattern search: nudge each parameter (the centre along and across the strip) while that
        improves the score, then halve the steps, until they are fine."""
        p, best = p0.astype(np.float64).copy(), self.score(p0)
        steps = steps.astype(np.float64).copy()
        for _ in range(9):
            moved = True
            while moved:
                moved = False
                u, v = _axes(p[2])  # the centre moves along and across the strip, not in x and y
                for i in range(len(p)):
                    for sign in (1.0, -1.0):
                        q = p.copy()
                        if i < 2:
                            d = u if i == 0 else v
                            q[0] += sign * steps[i] * d[0]
                            q[1] += sign * steps[i] * d[1]
                        else:
                            q[i] += sign * steps[i]
                        sc = self.score(q)
                        if sc > best:
                            p, best, moved = q, sc, True
                            break
            steps /= 2
        return p


def measure_slit(
    closed: np.ndarray,
    open_: np.ndarray,
    *,
    sigma: float = 1.5,
    min_snr: float = 4.0,
    min_area: int = 40,
) -> SlitMeasurement:
    """The gap's centre, direction, length and width from a frame with the slit closed and one
    with it open. Raises :class:`SlitNotFound` when no strip stands clear of the noise."""
    if closed.shape != open_.shape:
        raise ValueError("the closed and open frames differ in size")
    import sep

    raw = closed.astype(np.float64) - open_.astype(np.float64)
    # Stars cancel in the difference but leave their shot noise, large beside a faint gap: the
    # refinement ignores the pixels where either frame is well above its sky.
    starry = np.zeros(raw.shape, bool)
    for f in (closed, open_):
        f = f.astype(np.float64)
        starry |= f > np.median(f) + 6.0 * _noise(f)
    good = ~starry
    noise = _noise(raw[good])
    smooth = _smooth(np.where(good, raw, 0.0), sigma)  # no star residuals to break the strip
    noise_s = _noise(smooth)

    # 1. the biggest blob in the smoothed difference: stars leave only small ones
    objects, seg = sep.extract(
        np.ascontiguousarray(smooth),
        4.0,
        err=noise_s,
        minarea=min_area,
        deblend_cont=1.0,
        filter_kernel=None,
        segmentation_map=True,
    )
    if len(objects) == 0:
        raise SlitNotFound("no gap in the difference of the frames")
    k = int(np.argmax(objects["npix"]))
    ys, xs = np.nonzero(seg == k + 1)
    w = np.clip(smooth[ys, xs], 0, None)
    cx, cy = np.average(xs, weights=w), np.average(ys, weights=w)
    cov = np.cov(np.vstack([xs - cx, ys - cy]), aweights=w)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    if evals[1] < 1.3**2 * max(evals[0], 1e-9):
        raise SlitNotFound("the brightest blob in the difference is not a strip")
    angle = _angle_of(float(evecs[0, 1]), float(evecs[1, 1]))
    length0 = math.sqrt(12 * evals[1])
    width0 = math.sqrt(12 * max(evals[0], 0.25))

    # 2. fit a box to the raw difference, each pixel weighted by one over its noise variance
    weight = _weights(closed, open_, raw)
    reach = 0.5 * (length0 + width0) + 20.0
    fit = _Fit(raw, weight, cx, cy, reach)
    p0 = np.array([cx, cy, angle, length0, width0])
    steps = np.array([2.0, 2.0, 1.5, max(2.0, 0.05 * length0), max(1.0, 0.1 * width0)])
    p = fit.search(p0, steps)
    cx, cy, angle, length, width = (float(v) for v in p)
    plateau = fit.brightness(p)
    if plateau <= 0:
        raise SlitNotFound("no gap in the difference of the frames")
    angle = _angle_of(*_axes(angle)[0])

    snr = plateau / noise
    if snr < min_snr * 0.25:  # a faint strip is still a strip; only the hopeless are refused
        raise SlitNotFound(f"the gap is too faint: {snr:.1f} sigma per pixel")
    return SlitMeasurement(
        significance=float(math.sqrt(max(fit.score(p), 0.0))),
        x=float(cx),
        y=float(cy),
        angle_deg=float(angle),
        length_px=float(length),
        width_px=float(width),
        snr=float(snr),
    )
