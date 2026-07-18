from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from .config import get_settings


DEFAULT_ALEMBIC_DATABASE_URL = "sqlite:///./replay-studio.db"


def alembic_config_value(value: str) -> str:
    """Escape ConfigParser interpolation while preserving the URL Alembic reads."""

    return value.replace("%", "%%")


def resolve_alembic_database_url(configured_url: str | None) -> str:
    """Respect an explicit Alembic URL, otherwise resolve application settings.

    ``alembic.ini`` carries the zero-config SQLite URL as a placeholder.  A
    caller that supplies any other URL (tests, operators, or an embedding
    process) owns that choice and must never be redirected to the application's
    cached settings.  The placeholder still resolves through ``Settings`` so a
    normal CLI invocation keeps supporting ``DATABASE_URL`` and ``.env``.
    """

    normalized = str(configured_url or "").strip()
    if normalized and normalized != DEFAULT_ALEMBIC_DATABASE_URL:
        return normalized
    return get_settings().database_url


def upgrade_database(revision: str = "head") -> None:
    """Upgrade the configured database from API startup or an admin command."""

    package_root = Path(__file__).resolve().parent.parent
    config = Config(str(package_root / "alembic.ini"))
    config.set_main_option("script_location", str(package_root / "alembic"))
    config.set_main_option(
        "sqlalchemy.url",
        alembic_config_value(get_settings().database_url),
    )
    command.upgrade(config, revision)
