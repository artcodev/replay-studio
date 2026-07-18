FROM python:3.12-slim

ARG TORCH_VERSION=2.12.1
ARG TORCHVISION_VERSION=0.27.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
# Installing the CPU wheels first prevents PyPI's generic Linux torch wheel
# from pulling several gigabytes of CUDA runtime into a CPU-only API/runner.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      --index-url "${TORCH_INDEX_URL}"
RUN --mount=type=cache,target=/root/.cache/pip pip install .
RUN python -c "from ultralytics import YOLO; YOLO('yolo26m.pt')"
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic
COPY app ./app
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
