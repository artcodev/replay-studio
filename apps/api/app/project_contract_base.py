"""Strict serialization policy shared by project-owned transport contracts."""

from pydantic import BaseModel, ConfigDict


def camel_case(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(item[:1].upper() + item[1:] for item in tail)


class ProjectContract(BaseModel):
    """Base model for the project API; it is policy, not a DTO registry."""

    model_config = ConfigDict(
        alias_generator=camel_case,
        populate_by_name=True,
        extra="forbid",
        from_attributes=True,
    )
