"""Send posts to every configured channel: Discord (main) and Telegram (optional fallback).

With neither configured (or NHL_DRY_RUN=1), messages are printed once. A failure on one channel is
logged and does not stop the other or the job.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nhlmodel.publish import discord as DC
from nhlmodel.publish import telegram as TG

log = logging.getLogger(__name__)


def _targets():
    t = []
    if DC.configured():
        t.append(("discord", DC))
    if TG._creds():
        t.append(("telegram", TG))
    return t


def send_message(text: str) -> None:
    targets = _targets()
    if not targets:
        DC.send_message(text)   # prints the Discord version
        return
    for name, mod in targets:
        try:
            mod.send_message(text)
        except Exception:
            log.exception("%s send_message failed", name)


def send_document(path: Path, caption: str = "") -> None:
    targets = _targets()
    if not targets:
        DC.send_document(path, caption)
        return
    for name, mod in targets:
        try:
            mod.send_document(path, caption)
        except Exception:
            log.exception("%s send_document failed", name)
