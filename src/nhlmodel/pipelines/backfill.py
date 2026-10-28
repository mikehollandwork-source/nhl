"""Download and parse finished games into the Parquet store.

python -m nhlmodel.pipelines.backfill --seasons 20212022 20222023 ...   # full seasons
python -m nhlmodel.pipelines.backfill --current                          # only new games this season
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from nhlmodel.config import season_for, settings, today_local
from nhlmodel.ingest import nhl_api, shifts
from nhlmodel.store import db

log = logging.getLogger("backfill")
TABLES = ("games", "shots", "player_games", "goalie_games", "states", "penalties", "stints", "player_toi",
          "officials", "scratches")
FINAL_STATE_ID = 7  # stats API gameStateId for final


KEYS = {"games": ["game_id"], "shots": ["game_id", "event_id"],
        "player_games": ["game_id", "player_id"], "goalie_games": ["game_id", "player_id"],
        "states": ["game_id", "period_bucket", "score_diff_home", "home_goalie", "away_goalie", "sk_diff_home"],
        "penalties": ["game_id", "event_id"], "stints": ["game_id", "period", "t0"],
        "player_toi": ["game_id", "player_id"], "officials": ["game_id"], "scratches": ["game_id", "player_id"]}


def parse_extras(pbp: dict, box: dict, rail: dict | None, shifts_raw: dict | None) -> dict[str, pd.DataFrame]:
    """Stints, ice time by strength, officials and scratches for one finished game."""
    gid, gtype = pbp["id"], int(pbp["gameType"])
    hid, aid = pbp["homeTeam"]["id"], pbp["awayTeam"]["id"]
    goalies = {p["playerId"] for s in ("homeTeam", "awayTeam")
               for p in box.get("playerByGameStats", {}).get(s, {}).get("goalies", [])}
    out: dict[str, pd.DataFrame] = {}
    st = shifts.parse_shifts(shifts_raw, gid, gtype, hid, goalies) if shifts_raw else pd.DataFrame()
    if not st.empty:
        out["stints"] = st
        toi = shifts.strength_toi(st, hid, aid)
        out["player_toi"] = toi.assign(game_id=gid)
    if rail:
        off, scr = shifts.parse_right_rail(rail, gid)
        out["officials"] = pd.DataFrame([off])
        out["scratches"] = pd.DataFrame(scr)
    return out


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
        for gid, got in fetched.items():
            pbp, box = got.get("pbp"), got.get("box")
            if not pbp or not box or pbp.get("gameState") not in ("OFF", "FINAL"):
                log.warning("skipping %s (missing or not final)", gid)
                continue
            try:
                parsed = nhl_api.parse_game(pbp, box)
                parsed.update(parse_extras(pbp, box, got.get("rail"), got.get("shifts")))
            except Exception as e:  # keep going; one odd game shouldn't kill a backfill
                log.exception("parse failed for %s: %s", gid, e)
                continue
            for t in TABLES:
                if t in parsed and not parsed[t].empty:
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


def refresh_extras(season: int, days: int = 21) -> int:
    """Shift charts are sometimes published days after a game. Re-fetch stints / ice time /
    officials for recent stored games that are still missing them."""
    from datetime import timedelta
    games = db.read("games", partitions=[str(season)])
    if games.empty:
        return 0
    stints = db.read("stints", partitions=[str(season)])
    have = set(stints.game_id) if not stints.empty else set()
    cutoff = (today_local() - timedelta(days=days)).isoformat()
    todo = games[(games.game_date >= cutoff) & ~games.game_id.isin(have)]
    if todo.empty:
        return 0
    fetched = nhl_api.fetch_games(todo.game_id.tolist())
    n = 0
    for gid, got in fetched.items():
        if not got.get("pbp") or not got.get("box") or not got.get("shifts"):
            continue
        extra = parse_extras(got["pbp"], got["box"], got.get("rail"), got.get("shifts"))
        if "stints" not in extra:
            continue
        d = games.loc[games.game_id == gid, "game_date"].iloc[0]
        for t, df in extra.items():
            if not df.empty:
                db.append(t, df, f"{season}_{d}", key=KEYS[t])
        n += 1
    log.info("season %s: refreshed shift data for %d of %d games", season, n, len(todo))
    return n


def backfill_edge(seasons: list[int]) -> int:
    """EDGE team summaries for the given seasons (small; one file, rewritten)."""
    from nhlmodel.ingest import edge
    g = db.read("games")
    teams = sorted(set(g.home_id) | set(g.away_id)) if not g.empty else []
    df = edge.fetch_seasons(teams, seasons)
    if df.empty:
        return 0
    old = db.read("edge_team")
    if not old.empty:
        old = old[~old.season.isin(df.season.unique())]
        df = pd.concat([old, df], ignore_index=True)
    db.write("edge_team", df, "all")
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int)
    ap.add_argument("--current", action="store_true", help="update the current season only")
    ap.add_argument("--all", action="store_true", help="train + test + current seasons from settings")
    ap.add_argument("--edge", action="store_true", help="also refresh NHL EDGE team summaries")
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
    if a.edge:
        log.info("EDGE rows: %d", backfill_edge(seasons or [season_for(today_local())]))


if __name__ == "__main__":
    main()
