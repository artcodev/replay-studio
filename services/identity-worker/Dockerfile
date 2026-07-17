FROM python:3.10-slim

ARG SOCCERNET_COMMIT=1c958345067218297d221e45e1a6405f975f83e0
ARG REID_DOWNLOAD_MODELS=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SOCCERNET_ROOT=/opt/sn-gamestate \
    SOCCERNET_COMMIT=${SOCCERNET_COMMIT} \
    PRTREID_WEIGHTS=/models/prtreid-soccernet-baseline.pth.tar \
    PRTREID_HRNET_WEIGHTS=/models/hrnetv2_w32_imagenet_pretrained.pth

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/SoccerNet/sn-gamestate.git "$SOCCERNET_ROOT" \
    && git -C "$SOCCERNET_ROOT" checkout "$SOCCERNET_COMMIT" \
    && rm -rf "$SOCCERNET_ROOT/.git"

COPY identity_worker_service ./identity_worker_service
COPY models /models
COPY scripts/fetch_models.py /tmp/fetch_identity_models.py
RUN if [ "$REID_DOWNLOAD_MODELS" = "1" ]; then python /tmp/fetch_identity_models.py; fi

EXPOSE 8091
CMD ["uvicorn", "identity_worker_service.main:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "1"]
