"""Run-once markers, so a job that posts once a day (or week) doesn't post twice when it is started
by two triggers (the external timer and GitHub's own, often late, schedule).

A marker is written only after a real post (not in dry-run mode) and lives in the data branch.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from nhlmodel.config import data_dir

log = logging.getLogger(__name__)


def _path(key: str):
    return data_dir() / "runs" / f"{key}.json"


def already_done(key: str, force: bool = False) -> bool:
    if force:
        return False
    if _path(key).exists():
        log.info("%s already posted; skipping (use --force to post again)", key)
        return True
    return False


def mark_done(key: str) -> None:
    if os.environ.get("NHL_DRY_RUN") == "1":
        return
    p = _path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"done": datetime.now(UTC).isoformat(timespec="seconds")}))
