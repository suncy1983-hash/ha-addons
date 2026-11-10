"""MQTT: Home Assistant automatikusan felveszi a Hangfelismerő eszközt és a „Beszélő” szenzort.

Szenzorok (MQTT discovery):
  sensor.hangfelismero_beszelo      állapot: a beszélő neve, vagy „ismeretlen”
                                    attribútumok: speaker, person, score, nearest, reason, transcript,
                                                  satellite, time (ISO, ebből látszik, mennyire friss)
  sensor.hangfelismero_biztonsag    a legjobb egyezés hasonlósága, %
Esemény-topic: hangfelismero/esemeny (minden felismerésről egy JSON)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from .engine import Result

_LOGGER = logging.getLogger(__name__)

BASE = "hangfelismero"
UNKNOWN = "ismeretlen"
DEVICE = {
    "identifiers": ["hangfelismero"],
    "name": "Hangfelismerő",
    "manufacturer": "Saját fejlesztés",
    "model": "CAM++ beszélőazonosító",
}


class Mqtt:
    def __init__(self, host: str, port: int, username: str | None, password: str | None, version: str) -> None:
        self.version = version
        self._connected = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{BASE}-{os.getpid()}")
        if username:
            self.client.username_pw_set(username, password)
        self.client.will_set(f"{BASE}/status", "offline", qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(1, 30)
        self._host, self._port = host, port

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.client.connect_async(self._host, self._port, keepalive=60)
        self.client.loop_start()
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=15)
        except asyncio.TimeoutError:
            _LOGGER.warning("Az MQTT bróker (%s:%s) még nem érhető el; a háttérben újrapróbálom", self._host, self._port)

    def stop(self) -> None:
        self.client.publish(f"{BASE}/status", "offline", qos=1, retain=True)
        self.client.loop_stop()
        self.client.disconnect()

    # ----- visszahívások (paho szálon) ------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if reason_code.is_failure:
            _LOGGER.error("MQTT kapcsolódás elutasítva: %s", reason_code)
            return
        _LOGGER.info("MQTT kapcsolat rendben (%s:%s)", self._host, self._port)
        self._announce()
        client.subscribe("homeassistant/status")
        client.message_callback_add("homeassistant/status", self._on_ha_status)
        if self._loop:
            self._loop.call_soon_threadsafe(self._connected.set)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if self._loop:
            self._loop.call_soon_threadsafe(self._connected.clear)
        _LOGGER.warning("MQTT kapcsolat megszakadt (%s)", reason_code)

    def _on_ha_status(self, client, userdata, message) -> None:
        if message.payload == b"online":  # HA újraindult: újra bejelentjük magunkat
            self._announce()

    def _announce(self) -> None:
        device = {**DEVICE, "sw_version": self.version}
        availability = {"availability_topic": f"{BASE}/status"}
        configs = {
            f"homeassistant/sensor/{BASE}/beszelo/config": {
                "name": "Beszélő",
                "unique_id": f"{BASE}_beszelo",
                "default_entity_id": f"sensor.{BASE}_beszelo",
                "state_topic": f"{BASE}/beszelo/state",
                "json_attributes_topic": f"{BASE}/beszelo/attributes",
                "icon": "mdi:account-voice",
                "device": device,
                **availability,
            },
            f"homeassistant/sensor/{BASE}/biztonsag/config": {
                "name": "Biztonság",
                "unique_id": f"{BASE}_biztonsag",
                "default_entity_id": f"sensor.{BASE}_biztonsag",
                "state_topic": f"{BASE}/biztonsag/state",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "icon": "mdi:account-check",
                "entity_category": "diagnostic",
                "device": device,
                **availability,
            },
        }
        for topic, payload in configs.items():
            self.client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1, retain=True)
        self.client.publish(f"{BASE}/status", "online", qos=1, retain=True)

    # ----- eredmény -----------------------------------------------------------------------
    def payload(self, result: Result, transcript: str, satellite: str | None) -> dict:
        return {
            **result.as_dict(),
            "transcript": transcript,
            "satellite": satellite,
            "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }

    async def publish_result(self, result: Result, transcript: str, satellite: str | None) -> None:
        data = self.payload(result, transcript, satellite)
        attrs = {k: v for k, v in data.items() if k != "scores"}
        messages = [
            (f"{BASE}/beszelo/attributes", json.dumps(attrs, ensure_ascii=False)),
            (f"{BASE}/beszelo/state", result.speaker or UNKNOWN),
            (f"{BASE}/biztonsag/state", str(round(result.score * 100))),
            (f"{BASE}/esemeny", json.dumps(data, ensure_ascii=False)),
        ]
        infos = [self.client.publish(topic, body, qos=1, retain=topic != f"{BASE}/esemeny")
                 for topic, body in messages]

        def wait() -> None:
            for info in infos:
                try:
                    info.wait_for_publish(timeout=2)
                except (RuntimeError, ValueError):
                    pass  # nincs kapcsolat: a paho később elküldi

        await asyncio.get_running_loop().run_in_executor(None, wait)
