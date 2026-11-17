"""Service settings loaded from environment variables / .env file.

Only secrets, paths and process-level topology live here (see .env.example).
Business tunables (limits, defaults) live in the DB-backed runtime config
(app/runtime_config.py) and are editable from the admin console.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    """Project root (directory containing the ``model`` package)."""
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KRONOS_",
        env_file=(".env", str(repo_root() / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- process ---------------------------------------------------------
    env: str = "prod"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ---- admin console credentials (from env / .env) ----------------------
    admin_username: str = "admin"
    admin_password: str = "changeme"
    jwt_secret: str = ""  # empty -> generated per startup (sessions lost on restart)
    jwt_expire_hours: int = 12

    # ---- paths ------------------------------------------------------------
    models_dir: Path = repo_root() / "models"
    instance_dir: Path = repo_root() / "kronos_service" / "instance"
    log_dir: Path = repo_root() / "kronos_service" / "logs"

    # ---- inference process topology (restart required) --------------------
    device: str = "auto"  # auto | cuda | cuda:0 | cpu | mps
    max_concurrent_inferences: int = Field(1, ge=1, le=16)
    max_queue_size: int = Field(200, ge=1, le=5000)
    sync_timeout_seconds: int = Field(120, ge=1, le=3600)
    task_retention_hours: int = Field(24, ge=1, le=24 * 30)
    log_retention_days: int = Field(30, ge=1, le=365)

    # ---- misc --------------------------------------------------------------
    cors_origins: str = ""  # comma separated; empty = same-origin only
    max_request_body_mb: int = 20

    @property
    def db_path(self) -> Path:
        return self.instance_dir / "kronos.db"

    @property
    def hf_cache_dir(self) -> Path:
        return self.models_dir / "hf_cache"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def model_post_init(self, __context) -> None:
        # HF_HOME must be set before huggingface_hub is first imported so that
        # hub downloads land on the host-mounted models volume.
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = str(self.hf_cache_dir)
        if not self.jwt_secret:
            self.jwt_secret = secrets.token_urlsafe(48)

    def warnings(self) -> list[str]:
        out = []
        if self.admin_password == "changeme":
            out.append(
                "KRONOS_ADMIN_PASSWORD is left at the default 'changeme' - "
                "set a strong password in .env before exposing the service."
            )
        return out


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Used by tests to force re-reading the environment."""
    global _settings
    _settings = None
