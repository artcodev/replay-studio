"""Shared fail-closed policy for HTTP and provider-normalization contracts."""

from pydantic import BaseModel, ConfigDict


class TransportContract(BaseModel):
    model_config = ConfigDict(extra="forbid")
