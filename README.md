# Sanya Home Assistant appjai

## Hangfelismerő

Felismeri, **ki beszél** a hangsegédhez, és ezt átadja Home Assistantnek és a Sentientnek. Minden helyben fut.

- Wyoming proxy a Whisper elé: a hangot élőben továbbítja, közben ~0,1 mp alatt azonosítja a beszélőt (CAM++)
- `sensor.hangfelismero_beszelo` MQTT-n, mielőtt a szöveg a beszélgető ügynökhöz érne
- Betanító panel az oldalsávban: legutóbbi hangok hozzárendelése, feltöltés, próba, beállítások
- Biztos találatokból magától tanul tovább; opcionálisan elnyeli az ismeretlen hangok parancsait

### Telepítés

[![Tároló hozzáadása](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2Fsuncy1983-hash%2Fha-addons)

Részletek: [hangfelismero/DOCS.md](hangfelismero/DOCS.md)

### Fejlesztés

```bash
pip install -r hangfelismero/requirements.txt pytest
PYTHONPATH=hangfelismero/app pytest -q tests
```
