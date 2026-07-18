# syntax=docker/dockerfile:1
FROM python:3.10-slim

ARG SOCCERNET_COMMIT=1c958345067218297d221e45e1a6405f975f83e0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SOCCERNET_ROOT=/opt/sn-gamestate \
    PNLCALIB_KEYPOINT_WEIGHTS=/models/pnl_SV_kp \
    PNLCALIB_LINE_WEIGHTS=/models/pnl_SV_lines

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt
RUN git clone --filter=blob:none https://github.com/SoccerNet/sn-gamestate.git /opt/sn-gamestate \
    && git -C /opt/sn-gamestate checkout "$SOCCERNET_COMMIT" \
    && rm -rf /opt/sn-gamestate/.git

COPY models /models
RUN test -s /models/pnl_SV_kp \
    && test -s /models/pnl_SV_lines
COPY calibration_worker_service ./calibration_worker_service

EXPOSE 8090
CMD ["uvicorn", "calibration_worker_service.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
