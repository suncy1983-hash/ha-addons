"""Wyoming STT proxy: a hangot továbbítja a valódi beszédfelismerőnek (pl. Whisper), közben azonosítja a beszélőt.

Home Assistant felől ez egy közönséges beszéd-szöveg szolgáltatás. Minden hangparancsnál:
  1. a hangot élőben továbbküldi a Whispernek (nincs plusz késleltetés a felvétel alatt),
  2. a felvétel végén kiszámolja, ki beszél (~0,1 mp, párhuzamosan a Whisperrel),
  3. az eredményt MQTT-n közli (a Sentient ebből tudja, kivel beszél) MIELŐTT a szöveget visszaadja,
  4. visszaadja a Whisper szövegét.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Awaitable, Callable

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler

from . import hastt
from .engine import RATE, Result, SpeakerStore, pcm_to_float

_LOGGER = logging.getLogger(__name__)

UPSTREAM_TIMEOUT = 60.0
MAX_SECONDS = 30  # ennél hosszabb hangot nem tartunk memóriában

Publish = Callable[[Result, str, str | None], Awaitable[None]]


class UpstreamInfo:
    """A valódi beszédfelismerő leírása (nyelvek, modellek), gyorsítótárazva."""

    def __init__(self, uri: str, version: str) -> None:
        self.uri = uri
        self.version = version
        self.ha_target = hastt.parse_target(uri) if hastt.is_ha_uri(uri) else None
        self._info: Info | None = None
        self._fetched = 0.0

    async def get(self) -> Info:
        if self.ha_target is not None:
            return await self._get_ha()
        if self._info is None or time.monotonic() - self._fetched > 300:
            try:
                async with AsyncClient.from_uri(self.uri, connect_timeout=5, read_timeout=10) as client:
                    await client.write_event(Describe().event())
                    while True:
                        event = await asyncio.wait_for(client.read_event(), timeout=10)
                        if event is None:
                            raise ConnectionError("a beszédfelismerő bontotta a kapcsolatot")
                        if Info.is_type(event.type):
                            self._info, self._fetched = Info.from_event(event), time.monotonic()
                            break
            except (OSError, asyncio.TimeoutError, ConnectionError) as err:
                _LOGGER.warning("A beszédfelismerő (%s) nem érhető el: %s", self.uri, err)
        return self._wrap(self._info)

    async def _get_ha(self) -> Info:
        """Home Assistant STT célnál a modelleket a szolgáltató nyelveiből rakjuk össze."""
        target = self.ha_target
        assert target is not None
        if self._info is None or time.monotonic() - self._fetched > 300:
            languages = await hastt.describe(target)
            if languages is not None:
                # a pipeline „hu”-t és „hu-HU”-t is küldhet: mindkettőt felajánljuk
                codes = sorted({*languages, *(lang.split("-")[0] for lang in languages)})
                attribution = Attribution(name="Home Assistant", url="https://www.home-assistant.io")
                model = AsrModel(name=target.engine, description=target.engine, attribution=attribution,
                                 installed=True, version=None, languages=codes)
                self._info = Info(asr=[AsrProgram(name=target.engine, description=target.engine,
                                                  attribution=attribution, installed=True, version=None,
                                                  models=[model])])
                self._fetched = time.monotonic()
        return self._wrap(self._info)

    def _wrap(self, upstream: Info | None) -> Info:
        models = []
        if upstream is not None:
            for program in upstream.asr:
                models.extend(program.models)
        attribution = Attribution(name="Hangfelismerő", url="https://github.com/suncy1983-hash")
        program = AsrProgram(
            name="hangfelismero",
            description="Beszélőfelismerés + " + (upstream.asr[0].name if upstream and upstream.asr else "beszédfelismerő"),
            attribution=attribution,
            installed=True,
            version=self.version,
            models=models,
            supports_transcript_streaming=False,
        )
        return Info(asr=[program])


class ProxyHandler(AsyncEventHandler):
    """Egy kapcsolat Home Assistanttel. A Wyoming egy kapcsolaton egy felismerést kezel egyszerre."""

    def __init__(self, reader, writer, *, store: SpeakerStore, upstream: UpstreamInfo, publish: Publish) -> None:
        super().__init__(reader, writer)
        self.store = store
        self.upstream = upstream
        self.publish = publish
        self._client: AsyncClient | None = None
        self._ha: hastt.HaSttSession | None = None
        self._transcribe: Transcribe | None = None
        self._format: tuple[int, int, int] = (RATE, 2, 1)
        self._pcm = bytearray()
        self._started = 0.0

    # ------------------------------------------------------------------------------------
    async def handle_event(self, event: Event) -> bool:
        try:
            if Describe.is_type(event.type):
                await self.write_event((await self.upstream.get()).event())
                return True
            if Transcribe.is_type(event.type):
                self._transcribe = Transcribe.from_event(event)
                await self._forward(event)
                return True
            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                self._format = (start.rate, start.width, start.channels)
                self._pcm.clear()
                self._started = time.monotonic()
                await self._forward(event)
                return True
            if AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                self._format = (chunk.rate, chunk.width, chunk.channels)
                rate, width, channels = self._format
                if len(self._pcm) < MAX_SECONDS * rate * width * channels:
                    self._pcm.extend(chunk.audio)
                await self._forward(event)
                return True
            if AudioStop.is_type(event.type):
                await self._forward(event)
                await self._finish()
                return True
            # minden mást (pl. ping) szó nélkül továbbítunk
            await self._forward(event)
            return True
        except Exception as err:  # noqa: BLE001 - a hangsegédnek mindig válaszolni kell
            _LOGGER.exception("Hiba a hangparancs feldolgozásakor")
            await self.write_event(Error(text=f"Hangfelismerő hiba: {err}", code="hangfelismero").event())
            await self._close_client()
            return False

    async def disconnect(self) -> None:
        await self._close_client()

    # ------------------------------------------------------------------------------------
    async def _forward(self, event: Event) -> None:
        if self.upstream.ha_target is not None:
            self._forward_ha(event)
            return
        if self._client is None:
            self._client = AsyncClient.from_uri(self.upstream.uri, connect_timeout=5)
            await self._client.connect()
            if self._transcribe is not None and not Transcribe.is_type(event.type):
                await self._client.write_event(self._transcribe.event())
        await self._client.write_event(event)

    def _forward_ha(self, event: Event) -> None:
        """Home Assistant STT cél: a hangot élőben töltjük fel, a többi eseményre nincs szükség."""
        target = self.upstream.ha_target
        assert target is not None
        if AudioStart.is_type(event.type) or (AudioChunk.is_type(event.type) and self._ha is None):
            if self._ha is not None:
                asyncio.create_task(self._ha.cancel())
            language = self._transcribe.language if self._transcribe else None
            self._ha = hastt.HaSttSession(target, language, self._format)
            self._ha.start()
        if AudioChunk.is_type(event.type):
            assert self._ha is not None
            self._ha.write(AudioChunk.from_event(event).audio, self._format)
        elif AudioStop.is_type(event.type) and self._ha is not None:
            self._ha.end()

    async def _read_transcript(self) -> Transcript:
        if self.upstream.ha_target is not None:
            if self._ha is None:
                return Transcript(text="")
            return Transcript(text=await self._ha.result())
        assert self._client is not None
        while True:
            event = await self._client.read_event()
            if event is None:
                raise ConnectionError("a beszédfelismerő a szöveg előtt bontotta a kapcsolatot")
            if Transcript.is_type(event.type):
                return Transcript.from_event(event)
            if Error.is_type(event.type):
                raise ConnectionError("a beszédfelismerő hibát jelzett: " + Error.from_event(event).text)
            # TranscriptStart/Chunk/Stop: mi egyben adjuk vissza a végeredményt

    async def _finish(self) -> None:
        rate, width, channels = self._format
        pcm, self._pcm = bytes(self._pcm), bytearray()
        loop = asyncio.get_running_loop()

        def work() -> tuple[np.ndarray, Result, np.ndarray | None]:
            audio = pcm_to_float(pcm, rate, width, channels)
            result, vec = self.store.identify(audio)
            return audio, result, vec

        # a Whisper és az azonosítás egyszerre fut
        transcript_task = asyncio.create_task(asyncio.wait_for(self._read_transcript(), UPSTREAM_TIMEOUT))
        try:
            audio, result, vec = await loop.run_in_executor(None, work)
        except Exception:  # noqa: BLE001 - ha az azonosítás elhasal, a szöveg akkor is menjen
            _LOGGER.exception("A beszélő azonosítása nem sikerült")
            audio, vec = np.zeros(0, dtype=np.float32), None
            result = Result(None, 0.0, None, 0.0, 0.0, "hiba")
        try:
            transcript = await transcript_task
        finally:
            await self._close_client()

        satellite = self._transcribe.name if self._transcribe else None
        text = transcript.text or ""
        _LOGGER.info("„%s” -> %s (%.2f, %s, %.1f mp beszéd, %.0f ms)", text, result.speaker or "ismeretlen",
                     result.score, result.reason, result.seconds, (time.monotonic() - self._started) * 1000)

        # először közöljük, ki beszél, csak utána adjuk vissza a szöveget: mire a Sentient
        # megkapja a kérést, a beszélő-szenzor már friss
        try:
            await self.publish(result, text, satellite)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Az eredményt nem sikerült közölni")

        if self.store.settings.block_unknown and result.speaker is None and result.reason != "nincs_profil":
            _LOGGER.info("Ismeretlen hang, a parancsot elnyelem (block_unknown)")
            out = replace(transcript, text="")
        else:
            out = transcript
        await self.write_event(out.event())

        def keep() -> None:
            try:
                self.store.log_recent(audio, vec, result, text, satellite)
                self.store.maybe_auto_enroll(audio, vec, result)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("A hangot nem sikerült elmenteni")

        if len(audio):
            loop.run_in_executor(None, keep)

    async def _close_client(self) -> None:
        if self._ha is not None:
            await self._ha.cancel()
            self._ha = None
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
        self._transcribe = None
