"""Telegram Bot API publisher.

Needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Without them (or with NHL_DRY_RUN=1) messages
are printed to stdout instead, so every pipeline can be run locally.
Telegram limits: 4096 characters per message, 50 MB per document.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

from nhlmodel.config import env

log = logging.getLogger(__name__)
API = "https://api.telegram.org/bot{token}/{method}"
LIMIT = 4000


def _creds() -> tuple[str, str] | None:
    if os.environ.get("NHL_DRY_RUN") == "1":
        return None
    t, c = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    return (t, c) if t and c else None


def chunks(text: str, limit: int = LIMIT) -> list[str]:
    """Split on line boundaries so <pre> blocks and tags are not cut mid-line."""
    out, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            out.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        out.append(cur)
    return out


def _balance_pre(parts: list[str]) -> list[str]:
    """Close and reopen <pre> blocks that a split cut in half (Telegram rejects unbalanced HTML)."""
    fixed, open_pre = [], False
    for p in parts:
        if open_pre:
            p = "<pre>" + p
        opens, closes = p.count("<pre>"), p.count("</pre>")
        open_pre = opens > closes
        if open_pre:
            p = p + "</pre>"
        fixed.append(p)
    return fixed


def send_message(text: str) -> None:
    parts = _balance_pre(chunks(text))
    creds = _creds()
    if not creds:
        for p in parts:
            print(p)
        return
    token, chat = creds
    for p in parts:
        r = httpx.post(API.format(token=token, method="sendMessage"), timeout=30,
                       data={"chat_id": chat, "text": p, "parse_mode": "HTML", "disable_web_page_preview": "true"})
        if r.status_code != 200:
            log.error("telegram sendMessage failed: %s %s", r.status_code, r.text[:300])
            r.raise_for_status()


def send_document(path: Path, caption: str = "") -> None:
    creds = _creds()
    if not creds:
        print(f"[document] {path} {caption}")
        return
    token, chat = creds
    with open(path, "rb") as f:
        r = httpx.post(API.format(token=token, method="sendDocument"), timeout=60,
                       data={"chat_id": chat, "caption": caption[:1000]}, files={"document": (path.name, f)})
    if r.status_code != 200:
        log.error("telegram sendDocument failed: %s %s", r.status_code, r.text[:300])
        r.raise_for_status()
