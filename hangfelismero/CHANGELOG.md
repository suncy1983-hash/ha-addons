# Változások

## 0.2.0

- Új: a szövegfelismerést a Home Assistant saját STT szolgáltatása is végezheti, pl. a Nabu Casa felhő
  (`ha://stt.home_assistant_cloud`). A hang élőben megy fel, a beszélőazonosítás továbbra is helyben fut.
- Ez lett az alapértelmezett cím, mert a helyi Whisper magyarul gyakran félreérti a parancsokat.
  Meglévő telepítésnél a Beállításokban kell átírni a címet.

## 0.1.1

- Javítva: a NUC-on nem épült fel az image (a Supervisor az elavult build.yaml helyett Alpine alapra épített); a Debian alap most a Dockerfile-ban van

## 0.1.0

- Első változat: Wyoming STT proxy beszélőazonosítással (CAM++ + Silero VAD)
- Betanító felület az oldalsávban: beszélők, minták, legutóbbi hangok, feltöltés, próba
- MQTT szenzorok: `sensor.hangfelismero_beszelo`, `sensor.hangfelismero_biztonsag`
- Automatikus tanulás a biztos találatokból, opcionális szűrés az ismeretlen hangokra
