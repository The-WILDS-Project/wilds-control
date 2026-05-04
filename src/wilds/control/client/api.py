import httpx


class WildsApi:
    """Async HTTP/2 client for the wilds-control API.

    Usage::

        async with WildsApi("http://localhost:8001") as api:
            await api.acquire(name="M31", ra_hours=0.6625, dec_deg=41.27)
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "WildsApi":
        self._client = httpx.AsyncClient(base_url=self._base_url, http2=True)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    async def camera_start(self) -> None:
        await self._post("/camera/start")

    async def camera_stop(self) -> None:
        await self._post("/camera/stop")

    # ------------------------------------------------------------------
    # Slit
    # ------------------------------------------------------------------

    async def slit_move(self, *, mode: str, value_mm: float) -> None:
        await self._post("/slit/move", {"mode": mode, "value_mm": value_mm})

    # ------------------------------------------------------------------
    # ADC
    # ------------------------------------------------------------------

    async def adc_move(self, n: int, *, angle_deg: float) -> None:
        await self._post(f"/adc/{n}/move", {"angle_deg": angle_deg})

    async def adc_track(self, *, enable: bool) -> None:
        await self._post("/adc/track", {"enable": enable})

    # ------------------------------------------------------------------
    # Focus
    # ------------------------------------------------------------------

    async def focus_move(self, *, steps: int) -> None:
        await self._post("/focus/move", {"steps": steps})

    async def focus_auto(self) -> None:
        await self._post("/focus/auto")

    # ------------------------------------------------------------------
    # Acquire
    # ------------------------------------------------------------------

    async def acquire(
        self,
        *,
        name: str,
        ra_hours: float,
        dec_deg: float,
        tolerance_arcsec: float = 5.0,
    ) -> None:
        await self._post("/acquire", {
            "name": name,
            "ra_hours": ra_hours,
            "dec_deg": dec_deg,
            "tolerance_arcsec": tolerance_arcsec,
        })

    async def acquire_cancel(self) -> None:
        await self._post("/acquire/cancel")

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    async def solve(self) -> dict:
        r = await self._post("/solve")
        return r.json()

    # ------------------------------------------------------------------

    async def _post(self, path: str, body: dict | None = None) -> httpx.Response:
        assert self._client is not None, "WildsApi must be used as an async context manager"
        r = await self._client.post(path, json=body)
        r.raise_for_status()
        return r
