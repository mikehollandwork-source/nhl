"""Projected lines, special-teams units and injuries from DailyFaceoff (unofficial scrape).

Verified page structure (Oct 2026): /teams/{slug}/line-combinations embeds `__NEXT_DATA__` with
props.pageProps.combinations.players[], each with name, positionIdentifier, groupIdentifier
(f1-f4, d1-d3, g, pp1, pp2, pk1, pk2, ir), injuryStatus (ir/out/dtd/None), gameTimeDecision.
A player can appear in several groups (e.g. F2 and PP1).
"""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

from nhlmodel.ingest.http import get
from nhlmodel.ingest.nhl_api import name_key

log = logging.getLogger(__name__)
URL = "https://www.dailyfaceoff.com/teams/{slug}/line-combinations"
EVEN_GROUPS = {"f1", "f2", "f3", "f4", "d1", "d2", "d3"}


def team_slug(place: str, common: str) -> str:
    s = f"{place} {common}".lower().replace(".", "").replace("é", "e")
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def parse_page(html: str) -> tuple[pd.DataFrame, str | None]:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return pd.DataFrame(), None
    comb = json.loads(m.group(1))["props"]["pageProps"].get("combinations") or {}
    rows = [{"name": p.get("name"), "name_key": name_key(p.get("name") or ""),
             "group": p.get("groupIdentifier"), "slot": p.get("positionIdentifier"),
             "injury": p.get("injuryStatus"), "gtd": bool(p.get("gameTimeDecision"))}
            for p in comb.get("players", [])]
    return pd.DataFrame(rows), comb.get("updatedAt")


def fetch_team(slug: str) -> tuple[pd.DataFrame, str | None]:
    try:
        r = get(URL.format(slug=slug), retries=3)
        r.raise_for_status()
        return parse_page(r.text)
    except Exception as e:  # best effort
        log.warning("DailyFaceoff lines failed for %s: %s", slug, e)
        return pd.DataFrame(), None


def summarize(lines: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """One row per player with NHL id: even-strength group, PP unit, PK unit, injury, game-time decision.

    `dressed` is True for players in an F/D group who are not listed as injured.
    """
    if lines.empty:
        return pd.DataFrame()
    agg = []
    for key, g in lines.groupby("name_key"):
        groups = set(g.group.dropna())
        ev = sorted(groups & EVEN_GROUPS)
        injury = next((i for i in g.injury if isinstance(i, str) and i), None)
        agg.append({"name_key": key, "name": g.name.iloc[0], "ev_group": ev[0] if ev else None,
                    "pp_unit": 1 if "pp1" in groups else 2 if "pp2" in groups else 0,
                    "pk_unit": 1 if "pk1" in groups else 2 if "pk2" in groups else 0,
                    "is_goalie": "g" in groups, "injury": injury, "gtd": bool(g.gtd.any())})
    df = pd.DataFrame(agg)
    df["dressed"] = df.ev_group.notna() & ~df.injury.isin(["ir", "out"])
    if not roster.empty:
        df = df.merge(roster[["player_id", "name_key"]], on="name_key", how="left")
    else:
        df["player_id"] = None
    return df
