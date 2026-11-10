"""Belépési pont: Wyoming proxy + MQTT + betanító felület.

Add-onként a /data/options.json-ból olvas; fejlesztéshez parancssori kapcsolókkal is fut:
  python -m hangfelismero --upstream tcp://127.0.0.1:10300 --data ./data --model ... --no-mqtt --dev
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from functools import partial
from pathlib import Path

import aiohttp
from aiohttp import web
from wyoming.server import AsyncServer

from . import __version__
from .engine import Embedder, Settings, SpeakerStore
from .proxy import ProxyHandler, UpstreamInfo
from .web import build_app

_LOGGER = logging.getLogger("hangfelismero")

SUPERVISOR = "http://supervisor"


def _options(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


async def _supervisor(method: str, path: str, payload: dict | None = None) -> dict | None:
    token = os.environ.get("SUPERVISOR_TOKEN")
    if not token:
        return None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.request(method, SUPERVISOR + path, json=payload,
                                       headers={"Authorization": f"Bearer {token}"}) as resp:
                body = await resp.json(content_type=None)
                if resp.status >= 400:
                    _LOGGER.warning("Supervisor %s %s: %s %s", method, path, resp.status, body)
                    return None
                return body.get("data", body)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
        _LOGGER.warning("Supervisor %s %s nem sikerült: %s", method, path, err)
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(prog="hangfelismero")
    parser.add_argument("--options", default="/data/options.json")
    parser.add_argument("--data", default="/data")
    parser.add_argument("--upstream")
    parser.add_argument("--port", type=int)
    parser.add_argument("--web-port", type=int, default=8099)
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "/models/campplus.onnx"))
    parser.add_argument("--vad", default=os.environ.get("HF_VAD", "/models/silero_vad.onnx"))
    parser.add_argument("--no-mqtt", action="store_true")
    parser.add_argument("--dev", action="store_true", help="a betanító felület ingress nélkül is elérhető")
    args = parser.parse_args()

    opts = _options(Path(args.options))
    logging.basicConfig(level=getattr(logging, str(opts.get("log_level", "info")).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    upstream_uri = args.upstream or opts.get("upstream_uri") or "tcp://core-whisper:10300"
    port = args.port or int(opts.get("port") or 10360)

    settings_path = data / "settings.json"
    settings = Settings.load(settings_path)
    embedder = Embedder(args.model, args.vad, threads=int(opts.get("threads") or 2))
    store = SpeakerStore(data, embedder, settings)
    _LOGGER.info("Hangfelismerő %s indul: %d beszélő, beszédfelismerő: %s", __version__, len(store.speakers), upstream_uri)

    status: dict = {"upstream": upstream_uri, "upstream_ok": False, "mqtt": False, "port": port, "version": __version__}
    upstream = UpstreamInfo(upstream_uri, __version__)

    # ---- MQTT --------------------------------------------------------------------------
    mqtt_client = None
    if not args.no_mqtt and opts.get("mqtt_enabled", True):
        conf = {k: opts.get(f"mqtt_{k}") for k in ("host", "port", "username", "password") if opts.get(f"mqtt_{k}")}
        if not conf.get("host"):
            service = await _supervisor("GET", "/services/mqtt")
            if service:
                conf = {"host": service.get("host"), "port": service.get("port"),
                        "username": service.get("username"), "password": service.get("password")}
        if conf.get("host"):
            from .mqtt import Mqtt

            mqtt_client = Mqtt(conf["host"], int(conf.get("port") or 1883), conf.get("username"), conf.get("password"),
                               __version__)
            await mqtt_client.start()
        else:
            _LOGGER.warning("Nincs MQTT bróker (telepítsd a Mosquittót): a beszélő-szenzor nem fog megjelenni")

    async def publish(result, transcript, satellite) -> None:
        if mqtt_client is not None:
            await mqtt_client.publish_result(result, transcript, satellite)

    # ---- Wyoming ------------------------------------------------------------------------
    # start() és nem run(): a run() saját SIGTERM-kezelőt tesz fel, ami csak a Wyoming-szervert állítaná le,
    # az app többi része futva maradna. A leállítást lent mi vezéreljük. A start() a foglalt portot is
    # azonnal jelzi, nem csak egy háttérfeladatban.
    server = AsyncServer.from_uri(f"tcp://0.0.0.0:{port}")
    await server.start(partial(ProxyHandler, store=store, upstream=upstream, publish=publish))
    _LOGGER.info("Wyoming proxy figyel: tcp://0.0.0.0:%s", port)

    # Home Assistant automatikus felismerése (Beállítások -> Eszközök: „Wyoming Protocol” felfedezve)
    me = await _supervisor("GET", "/addons/self/info")
    if me and me.get("hostname"):
        await _supervisor("POST", "/discovery", {"service": "wyoming", "config": {"uri": f"tcp://{me['hostname']}:{port}"}})

    # ---- betanító felület ---------------------------------------------------------------
    app = build_app(store, settings_path, ingress_only=not args.dev, status=status)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", args.web_port).start()
    _LOGGER.info("Betanító felület: %s port", args.web_port)

    async def watch() -> None:
        while True:
            info = await upstream.get()
            status["upstream_ok"] = bool(info.asr and info.asr[0].models)
            status["mqtt"] = bool(mqtt_client and mqtt_client.client.is_connected())
            await asyncio.sleep(30 if status["upstream_ok"] else 10)

    watch_task = asyncio.create_task(watch(), name="watch")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    _LOGGER.info("Leállás")
    watch_task.cancel()
    try:
        await asyncio.wait_for(server.stop(), timeout=5)
    except asyncio.TimeoutError:
        _LOGGER.warning("A folyamatban lévő hangparancsok nem fejeződtek be időben")
    await runner.cleanup()
    if mqtt_client is not None:
        mqtt_client.stop()


if __name__ == "__main__":
    asyncio.run(main())
