"""Hangazonosítás: hangbeágyazás (CAM++), beszélőprofilok és a legutóbbi hangok naplója."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger(__name__)

RATE = 16000
MIN_SPEECH_SECONDS = 0.8  # ennél rövidebb beszédből nem lehet megbízhatóan azonosítani


# ----------------------------------------------------------------------------------------------
# Hang segédfüggvények
# ----------------------------------------------------------------------------------------------
def pcm_to_float(pcm: bytes, rate: int, width: int, channels: int) -> np.ndarray:
    """Nyers PCM -> 16 kHz-es mono float32 (-1..1)."""
    if width == 2:
        audio = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(pcm[: len(pcm) - len(pcm) % 4], dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Nem támogatott mintaszélesség: {width}")
    if channels > 1:
        audio = audio[: len(audio) - len(audio) % channels].reshape(-1, channels).mean(axis=1)
    if rate != RATE and len(audio):
        n = int(round(len(audio) * RATE / rate))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    return audio.astype(np.float32, copy=False)


def float_to_wav(audio: np.ndarray, path: Path) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return pcm_to_float(w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth(), w.getnchannels())


def decode_any(data: bytes) -> np.ndarray:
    """Bármilyen hangfájl (wav, mp3, m4a, ogg, flac, webm) -> 16 kHz mono float32, ffmpeg-gel."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"],
        input=data, capture_output=True, timeout=60, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError("A hangfájlt nem sikerült beolvasni: " + proc.stderr.decode(errors="replace")[:200])
    return pcm_to_float(proc.stdout, RATE, 2, 1)


# ----------------------------------------------------------------------------------------------
# Beágyazás
# ----------------------------------------------------------------------------------------------
class Embedder:
    """CAM++ hangbeágyazás + Silero VAD (csak a beszédes részekből számol)."""

    def __init__(self, model: str, vad_model: str | None, threads: int = 2) -> None:
        import sherpa_onnx

        self._sherpa = sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model, num_threads=threads, provider="cpu")
        if not cfg.validate():
            raise RuntimeError(f"Hibás beágyazó modell: {model}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._vad_model = vad_model if vad_model and os.path.exists(vad_model) else None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return self._extractor.dim

    def speech_only(self, audio: np.ndarray) -> np.ndarray:
        """A csendes szakaszok elhagyása; ha nincs VAD-modell, hangerő alapján vág."""
        if self._vad_model:
            cfg = self._sherpa.VadModelConfig()
            cfg.silero_vad.model = self._vad_model
            cfg.silero_vad.threshold = 0.5
            cfg.silero_vad.min_silence_duration = 0.25
            cfg.silero_vad.min_speech_duration = 0.2
            cfg.sample_rate = RATE
            vad = self._sherpa.VoiceActivityDetector(cfg, buffer_size_in_seconds=max(30, len(audio) / RATE + 5))
            window = cfg.silero_vad.window_size
            parts: list[np.ndarray] = []
            for i in range(0, len(audio), window):
                vad.accept_waveform(audio[i : i + window])
                while not vad.empty():
                    parts.append(np.asarray(vad.front.samples, dtype=np.float32))
                    vad.pop()
            vad.flush()
            while not vad.empty():
                parts.append(np.asarray(vad.front.samples, dtype=np.float32))
                vad.pop()
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        # tartalék: 30 ms-os keretek, a leghangosabbakhoz mérten
        frame = 480
        n = len(audio) // frame
        if n == 0:
            return audio
        frames = audio[: n * frame].reshape(n, frame)
        energy = np.sqrt((frames**2).mean(axis=1))
        keep = energy > max(0.01, np.percentile(energy, 80) * 0.15)
        return frames[keep].reshape(-1)

    def embed(self, audio: np.ndarray) -> tuple[np.ndarray | None, float]:
        """(egységnyi hosszúságú beágyazás vagy None, beszéd hossza másodpercben)."""
        speech = self.speech_only(audio)
        seconds = len(speech) / RATE
        if seconds < MIN_SPEECH_SECONDS:
            return None, seconds
        with self._lock:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate=RATE, waveform=speech)
            stream.input_finished()
            vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return (vec / norm if norm > 0 else None), seconds


# ----------------------------------------------------------------------------------------------
# Tárolás
# ----------------------------------------------------------------------------------------------
@dataclass
class Settings:
    threshold: float = 0.50       # legalább ekkora hasonlóság kell az azonosításhoz
    margin: float = 0.10          # ennyivel kell jobbnak lennie a második legjobbnál
    auto_enroll: bool = True      # biztos találatnál a minta bekerül a profilba
    auto_enroll_score: float = 0.70
    max_samples: int = 30         # beszélőnként
    keep_recent: int = 60         # ennyi legutóbbi hangot őriz meg a betanításhoz
    block_unknown: bool = False   # ismeretlen hangnál üres szöveget ad vissza (pl. tévé)

    @classmethod
    def load(cls, path: Path, defaults: dict | None = None) -> "Settings":
        data = dict(defaults or {})
        if path.exists():
            try:
                data.update(json.loads(path.read_text()))
            except (ValueError, OSError):
                _LOGGER.warning("A beállításfájl sérült, alapértékekkel indulok")
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def update(self, data: dict) -> None:
        for key, value in data.items():
            if key not in self.__dataclass_fields__:
                continue
            current = getattr(self, key)
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(self, key, value)
        self.threshold = min(max(self.threshold, 0.1), 0.95)
        self.margin = min(max(self.margin, 0.0), 0.5)
        self.auto_enroll_score = min(max(self.auto_enroll_score, self.threshold), 0.99)
        self.max_samples = min(max(self.max_samples, 3), 200)
        self.keep_recent = min(max(self.keep_recent, 0), 500)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2))


@dataclass
class Result:
    speaker: str | None          # None = ismeretlen
    score: float                 # a legjobb profil hasonlósága
    nearest: str | None          # a legközelebbi profil (akkor is, ha nem elég közel)
    second: float                # a második legjobb hasonlósága
    seconds: float               # beszéd hossza
    reason: str                  # "ok" | "rovid" | "nincs_profil" | "gyenge" | "bizonytalan"
    person: str | None = None
    scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker, "score": round(self.score, 3), "nearest": self.nearest,
            "second": round(self.second, 3), "seconds": round(self.seconds, 2), "reason": self.reason,
            "person": self.person, "scores": {k: round(v, 3) for k, v in self.scores.items()},
        }


class SpeakerStore:
    """Beszélőprofilok (speakers.json + mintánkénti .npy és .wav) és a legutóbbi hangok naplója."""

    def __init__(self, root: Path, embedder: Embedder, settings: Settings) -> None:
        self.root = root
        self.embedder = embedder
        self.settings = settings
        self.samples_dir = root / "samples"
        self.recent_dir = root / "recent"
        for d in (self.samples_dir, self.recent_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._index_path = root / "speakers.json"
        self._lock = threading.RLock()
        self.speakers: dict[str, dict] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._load()

    # ----- perzisztencia -----------------------------------------------------------------
    def _load(self) -> None:
        if self._index_path.exists():
            try:
                self.speakers = json.loads(self._index_path.read_text())
            except (ValueError, OSError):
                _LOGGER.error("A speakers.json sérült; biztonsági másolatot készítek és üresen indulok")
                shutil.copy(self._index_path, self._index_path.with_suffix(".json.bak"))
                self.speakers = {}
        for name in self.speakers:
            self._refresh(name)

    def _save(self) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.speakers, ensure_ascii=False, indent=2))
        tmp.replace(self._index_path)

    def _refresh(self, name: str) -> None:
        vecs = []
        for s in self.speakers.get(name, {}).get("samples", []):
            p = self.samples_dir / f"{s['id']}.npy"
            if p.exists():
                vecs.append(np.load(p))
        if vecs:
            c = np.mean(vecs, axis=0)
            self._centroids[name] = c / (np.linalg.norm(c) or 1.0)
        else:
            self._centroids.pop(name, None)

    def _remove_files(self, sample_id: str) -> None:
        for ext in (".npy", ".wav"):
            (self.samples_dir / f"{sample_id}{ext}").unlink(missing_ok=True)

    # ----- beszélők ----------------------------------------------------------------------
    def create_speaker(self, name: str, person: str | None = None) -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("Adj meg egy nevet")
        if len(name) > 40:
            raise ValueError("A név legfeljebb 40 karakter lehet")
        with self._lock:
            self.speakers.setdefault(name, {"person": None, "samples": []})
            if person is not None:
                self.speakers[name]["person"] = person.strip() or None
            self._save()

    def update_speaker(self, name: str, new_name: str | None = None, person: str | None = None) -> None:
        with self._lock:
            if name not in self.speakers:
                raise KeyError(name)
            if person is not None:
                self.speakers[name]["person"] = person.strip() or None
            if new_name and (new_name := new_name.strip()) != name:
                if new_name in self.speakers:
                    raise ValueError("Ilyen nevű beszélő már van")
                self.speakers[new_name] = self.speakers.pop(name)
                if name in self._centroids:
                    self._centroids[new_name] = self._centroids.pop(name)
            self._save()

    def delete_speaker(self, name: str) -> None:
        with self._lock:
            if name not in self.speakers:
                raise KeyError(name)
            for s in self.speakers.pop(name).get("samples", []):
                self._remove_files(s["id"])
            self._centroids.pop(name, None)
            self._save()

    def add_sample(self, name: str, audio: np.ndarray, source: str, vec: np.ndarray | None = None) -> dict:
        """Minta hozzáadása; ha betelt a profil, a legrégebbi (lehetőleg automatikus) minta megy ki."""
        if vec is None:
            vec, seconds = self.embedder.embed(audio)
            if vec is None:
                raise ValueError(f"Túl kevés beszéd a mintában ({seconds:.1f} mp, legalább {MIN_SPEECH_SECONDS} mp kell)")
        with self._lock:
            self.create_speaker(name)
            sid = uuid.uuid4().hex[:12]
            np.save(self.samples_dir / f"{sid}.npy", vec.astype(np.float32))
            float_to_wav(audio, self.samples_dir / f"{sid}.wav")
            sample = {"id": sid, "created": time.time(), "source": source, "seconds": round(len(audio) / RATE, 1)}
            samples = self.speakers[name.strip()]["samples"]
            samples.append(sample)
            while len(samples) > self.settings.max_samples:
                auto = [s for s in samples if s.get("source") == "auto"]
                victim = min(auto or samples, key=lambda s: s["created"])
                samples.remove(victim)
                self._remove_files(victim["id"])
            self._refresh(name.strip())
            self._save()
            return sample

    def delete_sample(self, sample_id: str) -> None:
        with self._lock:
            for name, sp in self.speakers.items():
                for s in sp["samples"]:
                    if s["id"] == sample_id:
                        sp["samples"].remove(s)
                        self._remove_files(sample_id)
                        self._refresh(name)
                        self._save()
                        return
        raise KeyError(sample_id)

    def sample_path(self, sample_id: str) -> Path | None:
        p = self.samples_dir / f"{sample_id}.wav"
        return p if _safe_id(sample_id) and p.exists() else None

    # ----- azonosítás --------------------------------------------------------------------
    def identify_vec(self, vec: np.ndarray | None, seconds: float) -> Result:
        with self._lock:
            centroids = dict(self._centroids)
        if vec is None:
            return Result(None, 0.0, None, 0.0, seconds, "rovid")
        if not centroids:
            return Result(None, 0.0, None, 0.0, seconds, "nincs_profil")
        scores = {name: float(np.dot(vec, c)) for name, c in centroids.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        s = self.settings
        if score < s.threshold:
            return Result(None, score, best, second, seconds, "gyenge", scores=scores)
        if len(ranked) > 1 and score - second < s.margin:
            return Result(None, score, best, second, seconds, "bizonytalan", scores=scores)
        person = self.speakers.get(best, {}).get("person")
        return Result(best, score, best, second, seconds, "ok", person=person, scores=scores)

    def identify(self, audio: np.ndarray) -> tuple[Result, np.ndarray | None]:
        vec, seconds = self.embedder.embed(audio)
        return self.identify_vec(vec, seconds), vec

    def maybe_auto_enroll(self, audio: np.ndarray, vec: np.ndarray | None, result: Result) -> bool:
        s = self.settings
        if not (s.auto_enroll and vec is not None and result.speaker and result.score >= s.auto_enroll_score):
            return False
        self.add_sample(result.speaker, audio, "auto", vec=vec)
        return True

    # ----- legutóbbi hangok --------------------------------------------------------------
    def log_recent(self, audio: np.ndarray, vec: np.ndarray | None, result: Result, transcript: str,
                   satellite: str | None) -> str | None:
        if self.settings.keep_recent <= 0:
            return None
        # az azonosító időrendben rendezhető (ezredmásodperc + sorszám), így a vágás mindig a legrégebbit viszi
        now = time.time()
        self._seq = (getattr(self, "_seq", 0) + 1) % 1000
        rid = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"{int(now * 1000) % 1000:03d}{self._seq:03d}-" \
            + uuid.uuid4().hex[:4]
        float_to_wav(audio, self.recent_dir / f"{rid}.wav")
        if vec is not None:
            np.save(self.recent_dir / f"{rid}.npy", vec.astype(np.float32))
        meta = {"id": rid, "time": time.time(), "transcript": transcript, "satellite": satellite, **result.as_dict()}
        (self.recent_dir / f"{rid}.json").write_text(json.dumps(meta, ensure_ascii=False))
        self._trim_recent()
        return rid

    def _trim_recent(self) -> None:
        metas = sorted(self.recent_dir.glob("*.json"))
        for old in metas[: max(0, len(metas) - self.settings.keep_recent)]:
            for ext in (".json", ".wav", ".npy"):
                old.with_suffix(ext).unlink(missing_ok=True)

    def recent(self) -> list[dict]:
        out = []
        for p in sorted(self.recent_dir.glob("*.json"), reverse=True):
            try:
                out.append(json.loads(p.read_text()))
            except (ValueError, OSError):
                continue
        return out

    def recent_path(self, rid: str) -> Path | None:
        p = self.recent_dir / f"{rid}.wav"
        return p if _safe_id(rid) and p.exists() else None

    def assign_recent(self, rid: str, name: str) -> dict:
        wav = self.recent_path(rid)
        if wav is None:
            raise KeyError(rid)
        audio = read_wav(wav)
        npy = self.recent_dir / f"{rid}.npy"
        vec = np.load(npy) if npy.exists() else None
        sample = self.add_sample(name, audio, "kezi", vec=vec)
        self.delete_recent(rid)
        return sample

    def delete_recent(self, rid: str) -> None:
        if not _safe_id(rid):
            raise KeyError(rid)
        for ext in (".json", ".wav", ".npy"):
            (self.recent_dir / f"{rid}{ext}").unlink(missing_ok=True)

    def describe(self) -> list[dict]:
        with self._lock:
            return [{"name": n, "person": sp.get("person"), "samples": sp["samples"]}
                    for n, sp in sorted(self.speakers.items())]


def _safe_id(value: str) -> bool:
    return bool(value) and len(value) <= 64 and all(c.isalnum() or c == "-" for c in value)
