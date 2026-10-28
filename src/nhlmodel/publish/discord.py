"""Discord webhook publisher.

Needs DISCORD_WEBHOOK_URL (Channel settings -> Integrations -> Webhooks -> New webhook -> Copy URL).
Without it (or with NHL_DRY_RUN=1) messages are printed instead.

Messages are written once in the Telegram HTML subset (<b>, <i>, <pre>, escaped text) and converted
here to Discord markdown. Discord limits: 2,000 characters per message, 10 MB per file on
non-boosted servers, about 5 requests per 2 seconds per webhook.
"""

from __future__ import annotations

import gzip
import html
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path

import httpx

from nhlmodel.config import env

log = logging.getLogger(__name__)
LIMIT = 1900
FILE_LIMIT = 10 * 1024 * 1024
FENCE = "```"
_MD = re.compile(r"([\\*_~|`>])")
_TOKEN = re.compile(r"(<pre>.*?</pre>)", re.S)


def _url() -> str | None:
    if os.environ.get("NHL_DRY_RUN") == "1":
        return None
    return env("DISCORD_WEBHOOK_URL") or None


def configured() -> bool:
    return _url() is not None


def _inline(seg: str) -> str:
    """Plain text with <b>/<i>: escape markdown in the text, then convert the tags."""
    parts = re.split(r"(</?[bi]>)", seg)
    out = []
    for p in parts:
        if p in ("<b>", "</b>"):
            out.append("**")
        elif p in ("<i>", "</i>"):
            out.append("*")
        else:
            text = html.unescape(re.sub(r"<[^>]+>", "", p))
            out.append(_MD.sub(r"\\\1", text))
    return "".join(out)


def to_markdown(text: str) -> str:
    out = []
    for seg in _TOKEN.split(text):
        if seg.startswith("<pre>") and seg.endswith("</pre>"):
            body = html.unescape(re.sub(r"<[^>]+>", "", seg[5:-6])).strip("\n").replace(FENCE, "'''")
            out.append(f"\n{FENCE}\n{body}\n{FENCE}\n")
        else:
            out.append(_inline(seg))
    md = "".join(out)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def chunks(md: str, limit: int = LIMIT) -> list[str]:
    """Split on line boundaries; a code block cut in two is closed and reopened."""
    parts, cur, in_code = [], "", False
    for line in md.split("\n"):
        # very long single lines are hard-wrapped
        while len(line) > limit - 10:
            parts_line, line = line[: limit - 10], line[limit - 10:]
            if cur:
                parts.append(cur + (f"\n{FENCE}" if in_code else ""))
                cur = FENCE + "\n" if in_code else ""
            parts.append((FENCE + "\n" if in_code else "") + parts_line + (f"\n{FENCE}" if in_code else ""))
        extra = len(FENCE) + 1 if in_code else 0
        if cur and len(cur) + len(line) + 1 + extra > limit:
            parts.append(cur.rstrip("\n") + (f"\n{FENCE}" if in_code else ""))
            cur = FENCE + "\n" if in_code else ""
        cur += line + "\n"
        if line.strip() == FENCE:
            in_code = not in_code
    if cur.strip() and cur.strip() != FENCE:
        parts.append(cur.rstrip("\n"))
    return [p for p in parts if p.strip()]


def _post(url: str, **kw) -> httpx.Response:
    for _ in range(4):
        r = httpx.post(url, timeout=60, **kw)
        if r.status_code == 429:
            try:
                wait = float(r.json().get("retry_after", 1.0))
            except ValueError:
                wait = 1.0
            time.sleep(min(wait, 30.0))
            continue
        return r
    return r


def send_message(text: str) -> None:
    parts = chunks(to_markdown(text))
    url = _url()
    if not url:
        for p in parts:
            print(p)
        return
    for p in parts:
        r = _post(url, json={"content": p, "allowed_mentions": {"parse": []}})
        if r.status_code not in (200, 204):
            log.error("discord message failed: %s %s", r.status_code, r.text[:300])
            r.raise_for_status()


def send_document(path: Path, caption: str = "") -> None:
    url = _url()
    if not url:
        print(f"[document] {path} {caption}")
        return
    path = Path(path)
    if path.stat().st_size > FILE_LIMIT:
        gz = path.with_suffix(path.suffix + ".gz")
        with open(path, "rb") as src, gzip.open(gz, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = gz
        if path.stat().st_size > FILE_LIMIT:
            log.error("file %s is over Discord's 10 MB limit even gzipped; not sent", path)
            return
    payload = {"content": to_markdown(caption)[:1900], "allowed_mentions": {"parse": []}}
    with open(path, "rb") as f:
        r = _post(url, data={"payload_json": json.dumps(payload)}, files={"files[0]": (path.name, f)})
    if r.status_code not in (200, 204):
        log.error("discord file upload failed: %s %s", r.status_code, r.text[:300])
        r.raise_for_status()
