# A Supervisor BUILD_FROM-ként az Alpine-alapú HA base image-et adná, de a sherpa-onnx-nak glibc kell,
# ezért a Debian-alapú Python image-et közvetlenül használjuk (a build.yaml elavult, a Supervisor figyelmen kívül hagyja).
FROM docker.io/library/python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Beszélőmodell: 3D-Speaker CAM++ (kínai+angol tanítás, nyelvtől függetlenül is jól különít), ~30 MB.
# Beszédszakasz-felismerés: Silero VAD.
ARG MODEL_BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download
RUN mkdir -p /models \
    && curl -sSfL -o /models/campplus.onnx \
       "${MODEL_BASE}/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx" \
    && curl -sSfL -o /models/silero_vad.onnx "${MODEL_BASE}/asr-models/silero_vad.onnx" \
    && python -c "import sherpa_onnx as s; c = s.SpeakerEmbeddingExtractorConfig(model='/models/campplus.onnx'); assert c.validate(); print('dim', s.SpeakerEmbeddingExtractor(c).dim)"

COPY app /app
WORKDIR /app
ENV PYTHONPATH=/app HF_MODEL=/models/campplus.onnx HF_VAD=/models/silero_vad.onnx

LABEL org.opencontainers.image.title="Hangfelismerő" \
      org.opencontainers.image.description="Beszélőfelismerő Wyoming proxy Home Assistanthez" \
      org.opencontainers.image.licenses="MIT"

CMD ["python", "-m", "hangfelismero"]
