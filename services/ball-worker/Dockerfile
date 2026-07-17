FROM python:3.10-slim

ARG TORCH_VERSION=2.2.2
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WASB_WEIGHTS=/models/wasb-soccer-best.pth.tar \
    WASB_HRNET_SOURCE=/opt/wasb-reference/hrnet.py \
    WASB_DEVICE=cpu

WORKDIR /service
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"

COPY ball_worker_service ./ball_worker_service

EXPOSE 8092
CMD ["uvicorn", "ball_worker_service.main:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]

