from __future__ import annotations

"""Shared database transaction primitives for atomic write repositories."""

from sqlalchemy import text


def begin_write_transaction(session) -> None:
    if session.get_bind().dialect.name == "sqlite":
        session.execute(text("BEGIN IMMEDIATE"))
    else:
        session.begin()
