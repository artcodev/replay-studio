FROM python:3.10-slim

ARG JERSEY_OCR_PROVIDER=mmocr
ARG JERSEY_OCR_DOWNLOAD_MODELS=1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JERSEY_OCR_PROVIDER=${JERSEY_OCR_PROVIDER} \
    JERSEY_OCR_DEVICE=cpu \
    JERSEY_OCR_PRELOAD=1 \
    JERSEY_OCR_ALLOW_RUNTIME_DOWNLOADS=0 \
    EASYOCR_MODULE_PATH=/models/easyocr \
    TORCH_HOME=/models/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service
COPY requirements*.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && if [ "$JERSEY_OCR_PROVIDER" = "mmocr" ]; then \
         pip install -r requirements-mmocr.txt; \
       elif [ "$JERSEY_OCR_PROVIDER" = "easyocr" ]; then \
         pip install -r requirements-easyocr.txt; \
       else \
         echo "Unsupported JERSEY_OCR_PROVIDER=$JERSEY_OCR_PROVIDER" >&2; exit 2; \
       fi \
    && python -c "import torch; assert torch.__version__ == '1.13.1+cpu'; assert torch.version.cuda is None" \
    && if [ "$JERSEY_OCR_PROVIDER" = "mmocr" ]; then \
         python -c "import importlib.metadata as m, importlib.util, mmcv; names = {d.metadata['Name'].lower() for d in m.distributions()}; assert mmcv.__version__ == '2.0.1'; assert m.version('opencv-python') == '4.11.0.86'; assert 'opencv-python-headless' not in names; assert importlib.util.find_spec('easyocr') is None"; \
       else \
         python -c "import easyocr, importlib.metadata as m, importlib.util; names = {d.metadata['Name'].lower() for d in m.distributions()}; assert m.version('opencv-python-headless') == '4.11.0.86'; assert 'opencv-python' not in names; assert importlib.util.find_spec('mmocr') is None"; \
       fi

COPY jersey_ocr_worker_service ./jersey_ocr_worker_service
COPY scripts ./scripts
RUN mkdir -p /models/easyocr /models/torch \
    && if [ "$JERSEY_OCR_DOWNLOAD_MODELS" = "1" ]; then \
         JERSEY_OCR_ALLOW_RUNTIME_DOWNLOADS=1 python -m scripts.cache_models; \
       fi

EXPOSE 8093
CMD ["uvicorn", "jersey_ocr_worker_service.main:app", "--host", "0.0.0.0", "--port", "8093", "--workers", "1"]
