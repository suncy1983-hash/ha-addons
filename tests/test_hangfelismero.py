"""Tesztek modell nélkül: a hangbeágyazást egy determinisztikus ál-beágyazó helyettesíti.

Az ál-beágyazó a hang spektrumából képez vektort, így két „beszélő” két különböző hangmagasság.
Futtatás: PYTHONPATH=hangfelismero/app pytest -q tests
"""

from __future__ import annotations

import asyncio
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from wyoming.asr import Transcribe, Transcript, TranscriptStart, TranscriptStop
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from hangfelismero.engine import RATE, Result, Settings, SpeakerStore, float_to_wav, pcm_to_float, read_wav
from hangfelismero.proxy import ProxyHandler, UpstreamInfo


# --------------------------------------------------------------------------------------------
class FakeEmbedder:
    """Vektor = a spektrum 16 sávja; hangmagasságonként jól elkülönül."""

    dim = 16

    def embed(self, audio):
        speech = audio[np.abs(audio) > 0.01]
        seconds = len(speech) / RATE
        if seconds < 0.8:
            return None, seconds
        spec = np.abs(np.fft.rfft(speech[:RATE]))[:4000]
        bands = spec.reshape(16, -1).sum(axis=1)
        return (bands / np.linalg.norm(bands)).astype(np.float32), seconds


def tone(freq: float, seconds: float = 2.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    rng = np.random.default_rng(seed)
    return (0.3 * np.sin(2 * np.pi * freq * t) + noise * rng.standard_normal(len(t))).astype(np.float32)


def pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


@pytest.fixture
def store(tmp_path: Path) -> SpeakerStore:
    return SpeakerStore(tmp_path, FakeEmbedder(), Settings())


# --------------------------------------------------------------------------------------------
def test_pcm_conversion_resamples_and_mixes():
    stereo_48k = np.stack([np.full(48000, 16384, "<i2"), np.zeros(48000, "<i2")], axis=1).tobytes()
    out = pcm_to_float(stereo_48k, 48000, 2, 2)
    assert len(out) == RATE
    assert out.dtype == np.float32
    assert abs(out.mean() - 0.25) < 1e-3
    assert len(pcm_to_float(b"\x01\x02\x03", RATE, 2, 1)) == 1  # páratlan bájtszám nem omlik össze


def test_wav_roundtrip(tmp_path):
    audio = tone(300, 1.0)
    float_to_wav(audio, tmp_path / "a.wav")
    assert np.allclose(read_wav(tmp_path / "a.wav"), audio, atol=1e-4)


def test_settings_clamped_and_persisted(tmp_path):
    s = Settings()
    s.update({"threshold": 5, "margin": -1, "max_samples": "7", "auto_enroll": 0, "unknown_key": 1})
    assert (s.threshold, s.margin, s.max_samples, s.auto_enroll) == (0.95, 0.0, 7, False)
    s.save(tmp_path / "s.json")
    assert Settings.load(tmp_path / "s.json").max_samples == 7
    (tmp_path / "bad.json").write_text("{nem json")
    assert Settings.load(tmp_path / "bad.json").threshold == Settings().threshold


def test_identify_known_unknown_short_and_ambiguous(store):
    assert store.identify(tone(200))[0].reason == "nincs_profil"
    store.add_sample("Anna", tone(200, seed=1), "kezi")
    store.add_sample("Bence", tone(1800, seed=2), "kezi")
    assert store.identify(tone(200, noise=0.02, seed=3))[0].speaker == "Anna"
    assert store.identify(tone(1800, noise=0.02, seed=4))[0].speaker == "Bence"
    assert store.identify(tone(3500))[0].reason == "gyenge"
    assert store.identify(tone(200, 0.4))[0].reason == "rovid"
    store.settings.update({"margin": 0.5, "threshold": 0.1})
    mixed = (tone(200) + tone(1800)) / 2
    assert store.identify(mixed)[0].reason == "bizonytalan"


def test_profile_cap_drops_oldest_automatic_first(store):
    store.settings.update({"max_samples": 3})
    store.add_sample("Anna", tone(200, seed=1), "kezi")
    autos = [store.add_sample("Anna", tone(200, seed=i), "auto") for i in range(2, 5)]
    sources = [s["source"] for s in store.speakers["Anna"]["samples"]]
    assert sources == ["kezi", "auto", "auto"]
    assert not (store.samples_dir / f"{autos[0]['id']}.npy").exists()


def test_rename_delete_and_persistence(tmp_path):
    st = SpeakerStore(tmp_path, FakeEmbedder(), Settings())
    st.add_sample("Anna", tone(200), "kezi")
    st.update_speaker("Anna", new_name="Anna B", person="person.anna")
    reloaded = SpeakerStore(tmp_path, FakeEmbedder(), Settings())
    result = reloaded.identify(tone(200))[0]
    assert (result.speaker, result.person) == ("Anna B", "person.anna")
    with pytest.raises(ValueError):
        reloaded.create_speaker("   ")
    reloaded.delete_speaker("Anna B")
    assert not list((tmp_path / "samples").iterdir())


def test_recent_log_assign_and_trim(store):
    store.settings.update({"keep_recent": 2})
    audio = tone(200)
    vec, secs = store.embedder.embed(audio)
    ids = [store.log_recent(audio, vec, store.identify_vec(vec, secs), f"szöveg {i}", None) for i in range(3)]
    assert [r["transcript"] for r in store.recent()] == ["szöveg 2", "szöveg 1"]
    store.assign_recent(ids[2], "Anna")
    assert store.identify(tone(200))[0].speaker == "Anna"
    with pytest.raises(KeyError):
        store.assign_recent("../../speakers", "Anna")
    assert store.recent_path("../x") is None


def test_auto_enroll_only_when_confident(store):
    store.add_sample("Anna", tone(200), "kezi")
    audio = tone(200, noise=0.01, seed=9)
    result, vec = store.identify(audio)
    store.settings.update({"auto_enroll_score": 0.99})
    assert store.maybe_auto_enroll(audio, vec, result) is (result.score >= 0.99)
    store.settings.update({"auto_enroll": False})
    assert store.maybe_auto_enroll(audio, vec, result) is False


# --------------------------------------------------------------------------------------------
class FakeWhisper(AsyncEventHandler):
    async def handle_event(self, event):
        if Describe.is_type(event.type):
            a = Attribution(name="x", url="x")
            await self.write_event(Info(asr=[AsrProgram(
                name="faster-whisper", attribution=a, installed=True, description="", version="1",
                models=[AsrModel(name="small", attribution=a, installed=True, description="", version="1",
                                 languages=["hu"])])]).event())
            return True
        if AudioStop.is_type(event.type):
            await self.write_event(TranscriptStart(language="hu").event())
            await self.write_event(TranscriptStop().event())
            await self.write_event(Transcript(text="kapcsold fel a lámpát").event())
            return False
        return True


async def _run_proxy(store, published, port_up, port_proxy):
    upstream = AsyncServer.from_uri(f"tcp://127.0.0.1:{port_up}")
    up_task = asyncio.create_task(upstream.run(FakeWhisper))

    async def publish(result: Result, text: str, satellite):
        published.append((result.speaker, text))

    proxy = AsyncServer.from_uri(f"tcp://127.0.0.1:{port_proxy}")
    info = UpstreamInfo(f"tcp://127.0.0.1:{port_up}", "test")
    proxy_task = asyncio.create_task(proxy.run(partial(ProxyHandler, store=store, upstream=info, publish=publish)))
    await asyncio.sleep(0.2)
    return up_task, proxy_task


async def _say(port: int, audio: np.ndarray) -> str:
    async with AsyncClient.from_uri(f"tcp://127.0.0.1:{port}") as c:
        await c.write_event(Transcribe(language="hu").event())
        await c.write_event(AudioStart(rate=RATE, width=2, channels=1).event())
        data = pcm(audio)
        for i in range(0, len(data), 2048):
            await c.write_event(AudioChunk(rate=RATE, width=2, channels=1, audio=data[i : i + 2048]).event())
        await c.write_event(AudioStop().event())
        while True:
            event = await asyncio.wait_for(c.read_event(), 10)
            if Transcript.is_type(event.type):
                return Transcript.from_event(event).text


def test_proxy_end_to_end(store):
    store.add_sample("Anna", tone(200), "kezi")
    published: list = []

    async def scenario():
        tasks = await _run_proxy(store, published, 18300, 18360)
        try:
            async with AsyncClient.from_uri("tcp://127.0.0.1:18360") as c:
                await c.write_event(Describe().event())
                info = Info.from_event(await c.read_event())
            assert info.asr[0].name == "hangfelismero"
            assert info.asr[0].models[0].languages == ["hu"]
            assert await _say(18360, tone(200, noise=0.01)) == "kapcsold fel a lámpát"
            assert await _say(18360, tone(3500)) == "kapcsold fel a lámpát"
            store.settings.update({"block_unknown": True})
            assert await _say(18360, tone(3500)) == ""          # ismeretlen: elnyelve
            assert await _say(18360, tone(200)) == "kapcsold fel a lámpát"
            await asyncio.sleep(0.3)
        finally:
            for t in tasks:
                t.cancel()

    asyncio.run(scenario())
    assert published == [("Anna", "kapcsold fel a lámpát"), (None, "kapcsold fel a lámpát"),
                         (None, "kapcsold fel a lámpát"), ("Anna", "kapcsold fel a lámpát")]
    assert len(store.recent()) == 4
    assert json.loads((store.root / "speakers.json").read_text())["Anna"]["samples"]


# --------------------------------------------------------------------------------------------
# Home Assistant STT cél (ha://), pl. Nabu Casa felhő
async def _fake_ha(port: int, received: dict):
    from aiohttp import web

    async def describe(request: web.Request):
        assert request.headers.get("Authorization") == "Bearer teszt"
        return web.json_response({"languages": ["hu-HU", "en-US"], "formats": ["wav"]})

    async def stt(request: web.Request):
        if received.get("no_speech"):
            await request.read()
            return web.json_response({"result": "error", "text": None})
        received["engine"] = request.match_info["engine"]
        received["meta"] = request.headers.get("X-Speech-Content")
        received["bytes"] = len(await request.read())
        return web.json_response({"result": "success", "text": "indítsd el a netflixet"})

    app = web.Application()
    app.router.add_get("/api/stt/{engine}", describe)
    app.router.add_post("/api/stt/{engine}", stt)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def test_proxy_home_assistant_stt(store, monkeypatch):
    monkeypatch.setenv("HA_URL", "http://127.0.0.1:18400")
    monkeypatch.setenv("HA_TOKEN", "teszt")
    store.add_sample("Anna", tone(200), "kezi")
    published: list = []
    received: dict = {}

    async def scenario():
        runner = await _fake_ha(18400, received)

        async def publish(result: Result, text: str, satellite):
            published.append((result.speaker, text))

        proxy = AsyncServer.from_uri("tcp://127.0.0.1:18460")
        info = UpstreamInfo("ha://stt.home_assistant_cloud", "test")
        task = asyncio.create_task(proxy.run(partial(ProxyHandler, store=store, upstream=info, publish=publish)))
        await asyncio.sleep(0.2)
        try:
            async with AsyncClient.from_uri("tcp://127.0.0.1:18460") as c:
                await c.write_event(Describe().event())
                described = Info.from_event(await c.read_event())
            assert described.asr[0].models[0].languages == ["en", "en-US", "hu", "hu-HU"]
            assert await _say(18460, tone(200, noise=0.01)) == "indítsd el a netflixet"
            received["no_speech"] = True                      # a felhő nem ért semmit: üres szöveg, nem hiba
            assert await _say(18460, tone(200, noise=0.01)) == ""
        finally:
            task.cancel()
            await runner.cleanup()

    asyncio.run(scenario())
    assert published == [("Anna", "indítsd el a netflixet"), ("Anna", "")]
    assert received["engine"] == "stt.home_assistant_cloud"
    assert "language=hu-HU" in received["meta"] and "sample_rate=16000" in received["meta"]
    assert received["bytes"] == len(pcm(tone(200)))


def test_ha_target_parsing_and_language(monkeypatch):
    from hangfelismero import hastt

    monkeypatch.delenv("HA_URL", raising=False)
    monkeypatch.setenv("SUPERVISOR_TOKEN", "sv")
    target = hastt.parse_target("ha://stt.home_assistant_cloud?language=hu-HU")
    assert target.url == "http://supervisor/core/api/stt/stt.home_assistant_cloud"
    assert target.headers == {"Authorization": "Bearer sv"}
    assert target.language_for("hu") == "hu-HU"
    assert target.language_for("hu-HU") == "hu-HU"
    assert target.language_for("en") == "en"
    assert target.language_for(None) == "hu-HU"
    assert hastt.parse_target("ha://stt.x").language == "hu-HU"
    with pytest.raises(ValueError):
        hastt.parse_target("ha://")
