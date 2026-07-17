"""Fail the Docker build early if the selected OCR models cannot be provisioned."""

from jersey_ocr_worker_service.providers import provider_from_environment


provider = provider_from_environment()
provider.load()
print(provider.info())
