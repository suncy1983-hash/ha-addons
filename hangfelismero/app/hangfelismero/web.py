"""Betanító felület (Home Assistant ingress mögött): beszélők, minták, legutóbbi hangok, beállítások."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
from pathlib import Path

from aiohttp import web

from .engine import RATE, SpeakerStore, decode_any

_LOGGER = logging.getLogger(__name__)

UI = Path(__file__).parent / "ui" / "index.html"
INGRESS_PEER = ipaddress.ip_address("172.30.32.2")  # a Supervisor ingress-proxyja
MAX_UPLOAD = 25 * 1024 * 1024


def build_app(store: SpeakerStore, settings_path: Path, *, ingress_only: bool, status: dict) -> web.Application:
    @web.middleware
    async def guard(request: web.Request, handler):
        if ingress_only:
            try:
                peer = ipaddress.ip_address((request.remote or "").split("%")[0])
            except ValueError:
                peer = None
            if peer != INGRESS_PEER:
                raise web.HTTPForbidden(text="Csak a Home Assistant felületéről érhető el")
        try:
            return await handler(request)
        except (KeyError, FileNotFoundError):
            return web.json_response({"error": "Nincs ilyen"}, status=404)
        except ValueError as err:
            return web.json_response({"error": str(err)}, status=400)

    app = web.Application(middlewares=[guard], client_max_size=MAX_UPLOAD)
    loop_run = lambda fn, *a: asyncio.get_running_loop().run_in_executor(None, fn, *a)  # noqa: E731

    async def index(request: web.Request) -> web.Response:
        return web.FileResponse(UI, headers={"Cache-Control": "no-store"})

    async def state(request: web.Request) -> web.Response:
        return web.json_response({
            "speakers": store.describe(),
            "recent": store.recent(),
            "settings": store.settings.__dict__,
            "status": status,
        })

    async def create_speaker(request: web.Request) -> web.Response:
        body = await request.json()
        store.create_speaker(body.get("name", ""), body.get("person"))
        return web.json_response({"ok": True})

    async def update_speaker(request: web.Request) -> web.Response:
        body = await request.json()
        store.update_speaker(request.match_info["name"], body.get("new_name"), body.get("person"))
        return web.json_response({"ok": True})

    async def delete_speaker(request: web.Request) -> web.Response:
        store.delete_speaker(request.match_info["name"])
        return web.json_response({"ok": True})

    async def delete_sample(request: web.Request) -> web.Response:
        store.delete_sample(request.match_info["sid"])
        return web.json_response({"ok": True})

    async def assign(request: web.Request) -> web.Response:
        body = await request.json()
        sample = await loop_run(store.assign_recent, request.match_info["rid"], body.get("name", ""))
        return web.json_response({"ok": True, "sample": sample})

    async def delete_recent(request: web.Request) -> web.Response:
        store.delete_recent(request.match_info["rid"])
        return web.json_response({"ok": True})

    async def _read_upload(request: web.Request) -> tuple[dict, list[bytes]]:
        fields: dict = {}
        files: list[bytes] = []
        reader = await request.multipart()
        async for part in reader:
            if part.filename:
                files.append(await part.read(decode=False))
            else:
                fields[part.name] = (await part.text()).strip()
        if not files:
            raise ValueError("Nem érkezett hangfájl")
        return fields, files

    async def upload(request: web.Request) -> web.Response:
        fields, files = await _read_upload(request)
        name = fields.get("name", "")
        added, errors = 0, []
        for data in files:
            try:
                audio = await loop_run(decode_any, data)
                await loop_run(store.add_sample, name, audio, "feltoltes")
                added += 1
            except ValueError as err:
                errors.append(str(err))
        return web.json_response({"ok": added > 0, "added": added, "errors": errors}, status=200 if added else 400)

    async def test(request: web.Request) -> web.Response:
        _, files = await _read_upload(request)
        audio = await loop_run(decode_any, files[0])
        result, _ = await loop_run(store.identify, audio)
        return web.json_response({**result.as_dict(), "duration": round(len(audio) / RATE, 1)})

    async def audio(request: web.Request) -> web.StreamResponse:
        kind, ident = request.match_info["kind"], request.match_info["ident"]
        path = store.recent_path(ident) if kind == "recent" else store.sample_path(ident) if kind == "sample" else None
        if path is None:
            raise KeyError(ident)
        return web.FileResponse(path, headers={"Content-Type": "audio/wav", "Cache-Control": "no-store"})

    async def save_settings(request: web.Request) -> web.Response:
        store.settings.update(await request.json())
        store.settings.save(settings_path)
        return web.json_response({"ok": True, "settings": store.settings.__dict__})

    app.add_routes([
        web.get("/", index),
        web.get("/api/state", state),
        web.post("/api/speakers", create_speaker),
        web.patch("/api/speakers/{name}", update_speaker),
        web.delete("/api/speakers/{name}", delete_speaker),
        web.delete("/api/samples/{sid}", delete_sample),
        web.post("/api/recent/{rid}/assign", assign),
        web.delete("/api/recent/{rid}", delete_recent),
        web.post("/api/upload", upload),
        web.post("/api/test", test),
        web.get("/api/audio/{kind}/{ident}", audio),
        web.post("/api/settings", save_settings),
    ])
    return app
