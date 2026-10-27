"""Download and parse finished games into the Parquet store.

python -m nhlmodel.pipelines.backfill --seasons 20212022 20222023 ...   # full seasons
python -m nhlmodel.pipelines.backfill --current                          # only new games this season
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from nhlmodel.config import season_for, settings, today_local
from nhlmodel.ingest import nhl_api
from nhlmodel.store import db

log = logging.getLogger("backfill")
TABLES = ("games", "shots", "player_games", "goalie_games", "states")
FINAL_STATE_ID = 7  # stats API gameStateId for final


KEYS = {"games": ["game_id"], "shots": ["game_id", "event_id"],
        "player_games": ["game_id", "player_id"], "goalie_games": ["game_id", "player_id"],
        "states": ["game_id", "period_bucket", "score_diff_home", "home_goalie", "away_goalie", "sk_diff_home"]}


def backfill_season(season: int, chunk: int = 200, per_day: bool = True) -> int:
    """Store every final game of a season that is not stored yet.

    per_day=True writes one small file per game date (used by the daily jobs) so files are
    written once and never rewritten, keeping the git history of the data branch small.
    per_day=False writes one file per season (initial bulk backfill of past seasons).
    """
    ids = nhl_api.season_game_ids(season)
    if ids.empty:
        log.info("season %s: no games", season)
        return 0
    stored = db.read("games", partitions=[str(season)])
    done = set(stored["game_id"]) if not stored.empty else set()
    todo = ids[(ids["game_state_id"] == FINAL_STATE_ID) & (~ids["game_id"].isin(done))]["game_id"].tolist()
    log.info("season %s: %d games listed, %d new final games", season, len(ids), len(todo))
    added = 0
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        fetched = nhl_api.fetch_games(batch)
        frames: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
        for gid, (pbp, box) in fetched.items():
            if not pbp or not box or pbp.get("gameState") not in ("OFF", "FINAL"):
                log.warning("skipping %s (missing or not final)", gid)
                continue
            try:
                parsed = nhl_api.parse_game(pbp, box)
            except Exception as e:  # keep going; one odd game shouldn't kill a backfill
                log.exception("parse failed for %s: %s", gid, e)
                continue
            for t in TABLES:
                if not parsed[t].empty:
                    frames[t].append(parsed[t])
        if not frames["games"]:
            continue
        date_of = pd.concat(frames["games"]).set_index("game_id")["game_date"]
        for t in TABLES:
            if not frames[t]:
                continue
            df = pd.concat(frames[t], ignore_index=True)
            if per_day:
                df["_d"] = df["game_id"].map(date_of)
                for d, part in df.groupby("_d"):
                    db.append(t, part.drop(columns="_d"), f"{season}_{d}", key=KEYS[t])
            else:
                db.append(t, df, str(season), key=KEYS[t])
        added += len(frames["games"])
        log.info("season %s: %d/%d", season, min(i + chunk, len(todo)), len(todo))
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int)
    ap.add_argument("--current", action="store_true", help="update the current season only")
    ap.add_argument("--all", action="store_true", help="train + test + current seasons from settings")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    h = settings()["history"]
    seasons = a.seasons or []
    if a.all:
        seasons = [*h["train_seasons"], h["test_season"], season_for(today_local())]
    if a.current:
        seasons = [season_for(today_local())]
    for s in seasons:
        n = backfill_season(s, per_day=a.current)
        log.info("season %s: added %d games", s, n)


if __name__ == "__main__":
    main()
