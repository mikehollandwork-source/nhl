"""Settings loader. Values come from config/settings.yaml; paths and secrets from the environment."""

from __future__ import annotations

import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def settings() -> dict:
    path = Path(os.environ.get("NHL_SETTINGS", REPO_ROOT / "config" / "settings.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def data_dir() -> Path:
    """Where Parquet tables and model files live (the checked-out `data` branch in CI)."""
    p = Path(os.environ.get("NHL_DATA_DIR", REPO_ROOT / "data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir() -> Path:
    p = Path(os.environ.get("NHL_OUT_DIR", REPO_ROOT / "out"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def tz() -> ZoneInfo:
    return ZoneInfo(settings()["timezone"])


def today_local() -> date:
    override = os.environ.get("NHL_DATE")
    if override:
        return date.fromisoformat(override)
    return datetime.now(tz()).date()


def season_for(d: date) -> int:
    """NHL season id (e.g. 20262027) for a calendar date. Seasons start in late September/October."""
    start = d.year if d.month >= 8 else d.year - 1
    return start * 10000 + start + 1


def env(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None
