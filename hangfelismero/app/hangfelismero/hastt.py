"""Home Assistant saját beszéd-szöveg szolgáltatása (pl. Nabu Casa felhő) mint továbbítási cél.

A Wyoming Whisper helyett a hangot élőben a Home Assistant STT API-jának küldjük:
  POST /api/stt/<entitás vagy szolgáltató>   (pl. stt.home_assistant_cloud)
Az add-onon belül a Supervisor proxyján keresztül érjük el (http://supervisor/core/api, SUPERVISOR_TOKEN),
fejlesztéskor a HA_URL és HA_TOKEN környezeti változókkal.

Cím formája az add-on beállításaiban:
  ha://stt.home_assistant_cloud                 (nyelv: hu-HU)
  ha://stt.home_assistant_cloud?language=hu-HU
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import parse_qs, urlparse

import aiohttp
import numpy as np

from .engine import RATE, pcm_to_float

_LOGGER = logging.getLogger(__name__)

SCHEME = "ha://"
DEFAULT_LANGUAGE = "hu-HU"
TIMEOUT = 60.0


def is_ha_uri(uri: str) -> bool:
    return uri.startswith(SCHEME)


@dataclass(frozen=True)
class HaTarget:
    engine: str
    language: str
    base_url: str
    token: str | None

    @property
    def url(self) -> str:
        return f"{self.base_url}/stt/{self.engine}"

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def language_for(self, requested: str | None) -> str:
        """A pipeline gyakran csak „hu”-t küld; a felhő szolgáltatónak a teljes kód kell (hu-HU)."""
        if requested and "-" in requested:
            return requested
        if requested and not self.language.lower().startswith(requested.lower()):
            return requested
        return self.language


def parse_target(uri: str) -> HaTarget:
    parsed = urlparse(uri)
    engine = (parsed.netloc + parsed.path).strip("/")
    if not engine:
        raise ValueError(f"Hiányzik a szolgáltató a címből: {uri} (pl. ha://stt.home_assistant_cloud)")
    language = parse_qs(parsed.query).get("language", [DEFAULT_LANGUAGE])[0]
    base_url = os.environ.get("HA_URL")
    token = os.environ.get("HA_TOKEN")
    if base_url:
        base_url = base_url.rstrip("/") + "/api"
    else:
        base_url = "http://supervisor/core/api"
        token = token or os.environ.get("SUPERVISOR_TOKEN")
    return HaTarget(engine=engine, language=language, base_url=base_url, token=token)


async def describe(target: HaTarget) -> list[str] | None:
    """A szolgáltató által támogatott nyelvek, vagy None, ha nem érhető el."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.get(target.url, headers=target.headers) as resp:
                if resp.status >= 400:
                    _LOGGER.warning("A beszédfelismerő (%s) nem érhető el: HTTP %s", target.url, resp.status)
                    return None
                body = await resp.json(content_type=None)
                return list(body.get("languages") or [target.language])
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
        _LOGGER.warning("A beszédfelismerő (%s) nem érhető el: %s", target.url, err)
        return None


class HaSttSession:
    """Egy felismerés: a hangdarabokat élőben továbbítja, a végén visszaadja a szöveget."""

    def __init__(self, target: HaTarget, language: str | None, fmt: tuple[int, int, int]) -> None:
        self.target = target
        self.language = target.language_for(language)
        self.fmt = fmt
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._task: asyncio.Task[str] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="ha-stt")

    def write(self, audio: bytes, fmt: tuple[int, int, int] | None = None) -> None:
        if fmt is not None:
            self.fmt = fmt
        if audio:
            self._queue.put_nowait(self._to_16k_mono(audio))

    def end(self) -> None:
        self._queue.put_nowait(None)

    async def result(self) -> str:
        assert self._task is not None
        return await self._task

    async def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    # ------------------------------------------------------------------------------------
    def _to_16k_mono(self, audio: bytes) -> bytes:
        rate, width, channels = self.fmt
        if (rate, width, channels) == (RATE, 2, 1):
            return audio
        samples = pcm_to_float(audio, rate, width, channels)
        return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()

    async def _body(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk

    async def _run(self) -> str:
        headers = dict(self.target.headers)
        headers["X-Speech-Content"] = (
            f"format=wav; codec=pcm; sample_rate={RATE}; bit_rate=16; channel=1; language={self.language}"
        )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as session:
            async with session.post(self.target.url, data=self._body(), headers=headers) as resp:
                body = await resp.json(content_type=None)
                if resp.status < 400 and isinstance(body, dict) and body.get("result") == "error":
                    # a szolgáltató nem ismert fel beszédet (csend, zaj): ez nem hiba, csak üres szöveg
                    _LOGGER.info("A beszédfelismerő (%s) nem ismert fel szöveget", self.target.engine)
                    return ""
                if resp.status >= 400 or not isinstance(body, dict) or body.get("result") != "success":
                    raise ConnectionError(f"a beszédfelismerő ({self.target.engine}) hibát jelzett: "
                                          f"HTTP {resp.status} {body}")
                return str(body.get("text") or "")
