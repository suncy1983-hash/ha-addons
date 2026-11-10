# Hangfelismerő

Felismeri, **ki beszél** a hangsegédhez, és ezt átadja Home Assistantnek és a Sentientnek.
Minden a NUC-on fut, felhő nélkül.

## Hogyan működik

A hangpipeline-ban a beszédfelismerő (STT) helyére kerül, és a hangot élőben továbbküldi a Whispernek.
A felvétel végén, a Whisperrel párhuzamosan, egy CAM++ hangmodell (~0,1 mp) eldönti, kinek a hangja szólt.
Az eredményt **előbb** közli MQTT-n, és csak utána adja vissza a szöveget. Így mire a Sentient megkapja
a kérést, a `sensor.hangfelismero_beszelo` már a friss nevet mutatja.

## Beállítás

1. Indítsd el az appot. Home Assistant felajánlja a *Wyoming Protocol* integrációt
   (Beállítások → Eszközök és szolgáltatások → Felfedezve). Add hozzá.
2. **Beállítások → Hangsegédek** → a hangsegéded → *Beszéd-szöveg*: válaszd a **hangfelismero**-t.
   A nyelv maradjon magyar; a háttérben a Whisper dolgozik tovább.
3. Nyisd meg az oldalsávban a **Hangfelismerő** panelt, és add hozzá a háztartás tagjait.
4. Beszélj a hangsegédhez néhányszor, majd a *Legutóbbi hangok* listában jelöld meg, ki volt az.
   Beszélőnként 3–5 minta elég. Utána a biztos találatokból magától tanul tovább.
5. A Sentient beállításaiban kapcsold be a **Beszélő felismerése hangról** lehetőséget,
   és válaszd ki a `sensor.hangfelismero_beszelo` szenzort.

## Entitások

| Entitás | Jelentés |
|---|---|
| `sensor.hangfelismero_beszelo` | A beszélő neve, vagy `ismeretlen`. Attribútumok: `person`, `score`, `nearest`, `reason`, `transcript`, `time` |
| `sensor.hangfelismero_biztonsag` | A legjobb egyezés hasonlósága (%) |

Az `ismeretlen` okai (`reason`): `rovid` (túl rövid volt a beszéd), `gyenge` (senkire sem hasonlít),
`bizonytalan` (két ember túl közel áll egymáshoz), `nincs_profil` (még senki nincs betanítva).

## Tévé és idegen hangok

A *Ismeretlen hang parancsát ne hajtsa végre* kapcsolóval az ismeretlen hangok parancsai üres szöveget kapnak,
vagyis nem történik semmi. Csak akkor kapcsold be, ha már mindenki be van tanítva, a vendégek miatt is.

## Adatvédelem

A hangminták és a legutóbbi hangok csak az app saját `/data` mappájában vannak; ezek a Home Assistant mentésébe is bekerülnek.
A *Megőrzött legutóbbi hangok* 0-ra állításával semmi nem mentődik el automatikusan.
