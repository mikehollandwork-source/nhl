"""NHL public API (api-web.nhle.com) client and parsers.

Verified fields (Oct 2026):
- play-by-play `situationCode` is 4 digits: away goalie, away skaters, home skaters, home goalie.
- `homeTeamDefendingSide` ("left"/"right") tells which net each team attacks in that period.
- The roster endpoint does NOT include starting goalies, line combos or injuries.
"""

from __future__ import annotations

import math
import unicodedata
from datetime import date

import numpy as np
import pandas as pd

from nhlmodel.ingest.http import fetch_many, get_json

WEB = "https://api-web.nhle.com/v1"
STATS = "https://api.nhle.com/stats/rest/en"

SHOT_EVENTS = {"shot-on-goal": "sog", "goal": "goal", "missed-shot": "miss", "blocked-shot": "block"}
NET_X = 89.0
BLUE_LINE_X = 25.0


# ----------------------------------------------------------------------------- endpoints

def schedule_day(d: date) -> list[dict]:
    """Games scheduled on one calendar date (the endpoint returns a week; we keep the one day)."""
    data = get_json(f"{WEB}/schedule/{d.isoformat()}")
    for day in data.get("gameWeek", []):
        if day.get("date") == d.isoformat():
            return day.get("games", [])
    return []


def season_game_ids(season: int, game_types: tuple[int, ...] = (2, 3)) -> pd.DataFrame:
    """All games in a season from the stats REST API."""
    types = " or ".join(f"gameType={t}" for t in game_types)
    data = get_json(f"{STATS}/game", params={"cayenneExp": f"season={season} and ({types})"})
    df = pd.DataFrame(data["data"])
    if df.empty:
        return df
    return df.rename(columns={"id": "game_id", "visitingTeamId": "away_id", "homeTeamId": "home_id",
                              "gameDate": "game_date", "gameType": "game_type",
                              "gameStateId": "game_state_id"})


def roster(team_abbrev: str) -> pd.DataFrame:
    data = get_json(f"{WEB}/roster/{team_abbrev}/current")
    rows = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []):
            rows.append({
                "player_id": p["id"],
                "first": p["firstName"]["default"],
                "last": p["lastName"]["default"],
                "position": p.get("positionCode"),
                "team": team_abbrev,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["name"] = df["first"] + " " + df["last"]
        df["name_key"] = df["name"].map(name_key)
    return df


def name_key(name: str) -> str:
    """Normalize a person or team name for matching across sources (accents, punctuation, case)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return "".join(ch for ch in s.lower() if ch.isalnum())


def fetch_games(game_ids: list[int], concurrency: int = 8, extras: bool = True) -> dict[int, dict]:
    """Fetch play-by-play, boxscore and (with extras) right-rail + shift charts for each game."""
    from nhlmodel.ingest.shifts import RAIL_URL, SHIFTS_URL
    urls: dict[int, dict[str, str]] = {}
    for g in game_ids:
        urls[g] = {"pbp": f"{WEB}/gamecenter/{g}/play-by-play", "box": f"{WEB}/gamecenter/{g}/boxscore"}
        if extras:
            urls[g].update({"rail": RAIL_URL.format(gid=g), "shifts": SHIFTS_URL.format(gid=g)})
    got = fetch_many([u for d in urls.values() for u in d.values()], concurrency=concurrency)
    return {g: {k: got.get(u) for k, u in d.items()} for g, d in urls.items()}


# ----------------------------------------------------------------------------- parsing helpers

def _secs(mmss: str | None) -> int:
    if not mmss:
        return 0
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def period_length(period: int, game_type: int) -> int:
    if period <= 3:
        return 1200
    return 1200 if game_type == 3 else 300


def game_seconds(period: int, t_in_period: str, game_type: int) -> int:
    base = sum(period_length(p, game_type) for p in range(1, period))
    return base + _secs(t_in_period)


def parse_situation(code: str | None) -> tuple[int, int, int, int]:
    """-> (away_goalie, away_skaters, home_skaters, home_goalie). Defaults to 5v5 with goalies."""
    if not code or len(code) != 4 or not code.isdigit():
        return 1, 5, 5, 1
    return int(code[0]), int(code[1]), int(code[2]), int(code[3])


def shot_geometry(x: float, y: float, attack_sign: int) -> tuple[float, float]:
    """Distance (ft) and angle (deg) to the attacked net at x = attack_sign * 89.

    Angle is 0 straight on, 90 at the goal line, >90 from behind the net.
    """
    dx = NET_X - x * attack_sign
    dist = math.hypot(dx, y)
    ang = math.degrees(math.atan2(abs(y), dx))
    return dist, ang


def _attack_sign(is_home: bool, home_defending: str | None, x: float | None, zone: str | None) -> int:
    if home_defending in ("left", "right"):
        home_sign = -1 if home_defending == "right" else 1  # home defends right net -> attacks left (x<0)
        return home_sign if is_home else -home_sign
    # Fallback: offensive-zone event tells us the side directly.
    if x is not None and zone == "O":
        return 1 if x > 0 else -1
    return 1 if (x or 0) >= 0 else -1


# ----------------------------------------------------------------------------- game parsing

def parse_game(pbp: dict, box: dict | None) -> dict[str, pd.DataFrame]:
    """Turn one game's play-by-play + boxscore into table rows.

    Returns frames: games, shots, player_games, goalie_games, states, penalties.
    """
    gid = pbp["id"]
    gtype = int(pbp["gameType"])
    season = int(pbp["season"])
    gdate = pbp["gameDate"]
    home_id = pbp["homeTeam"]["id"]
    away_id = pbp["awayTeam"]["id"]

    plays = sorted(pbp.get("plays", []), key=lambda p: p.get("sortOrder", 0))
    player_team = {r["playerId"]: r["teamId"] for r in pbp.get("rosterSpots", [])}
    shots = []
    penalties: list[dict] = []
    states: dict[tuple, list[float]] = {}
    goals_by_team = {home_id: 0, away_id: 0}
    reg_goals = {home_id: 0, away_id: 0}
    pp_points: dict[int, int] = {}
    en_goals = {home_id: 0, away_id: 0}
    first_goal_team = None
    pulls: dict[int, tuple[int, int] | None] = {home_id: None, away_id: None}
    pending: dict[int, tuple[int, int] | None] = {home_id: None, away_id: None}

    prev_t = 0
    prev_sit = (1, 5, 5, 1)
    prev_event = None  # (t, team_id, type, x, y)

    for p in plays:
        pd_ = p.get("periodDescriptor", {})
        per = int(pd_.get("number", 1))
        ptype = pd_.get("periodType", "REG")
        if ptype == "SO":
            continue
        t = game_seconds(per, p.get("timeInPeriod"), gtype)
        sit = parse_situation(p.get("situationCode"))
        etype = p.get("typeDescKey")
        det = p.get("details", {}) or {}

        # accumulate time spent in the previous state
        if t > prev_t:
            key = _state_key(per, goals_by_team[home_id] - goals_by_team[away_id], prev_sit, t)
            acc = states.setdefault(key, [0.0, 0, 0, 0, 0])
            acc[0] += t - prev_t
        prev_t = max(prev_t, t)
        prev_sit = sit

        if etype in SHOT_EVENTS:
            # Verified: for all shot events (incl. blocked) the event owner is the shooting team.
            # Prefer the shooter's roster team when available.
            shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
            team = player_team.get(shooter, det.get("eventOwnerTeamId"))
            if team not in (home_id, away_id):
                continue
            is_home = team == home_id
            ag, ask, hsk, hg = sit
            sk_for, sk_against = (hsk, ask) if is_home else (ask, hsk)
            goalie_against_in = (ag if is_home else hg) == 1
            own_goalie_in = (hg if is_home else ag) == 1
            x, y = det.get("xCoord"), det.get("yCoord")
            sign = _attack_sign(is_home, p.get("homeTeamDefendingSide"), x, det.get("zoneCode"))
            dist = ang = np.nan
            if x is not None and y is not None:
                dist, ang = shot_geometry(float(x), float(y), sign)
            score_for = goals_by_team[team]
            score_against = goals_by_team[away_id if is_home else home_id]

            rebound = rush = False
            if prev_event is not None:
                pt, pteam, ptype_ev, px, _py = prev_event
                dt = t - pt
                if pteam == team and ptype_ev in ("shot-on-goal",) and 0 <= dt <= 3:
                    rebound = True
                if 0 <= dt <= 4 and px is not None and px * sign < BLUE_LINE_X and ptype_ev not in SHOT_EVENTS:
                    rush = True

            ev = SHOT_EVENTS[etype]
            row = {
                "game_id": gid, "season": season, "game_type": gtype, "game_date": gdate,
                "event_id": p.get("eventId"), "period": per, "t": t, "team_id": team, "is_home": is_home,
                "shooter_id": det.get("shootingPlayerId") or det.get("scoringPlayerId"),
                "goalie_id": det.get("goalieInNetId"),
                "event": ev, "shot_type": det.get("shotType") or "unknown",
                "x": x, "y": y, "distance": dist, "angle": ang,
                "sk_for": sk_for, "sk_against": sk_against,
                "empty_net": not goalie_against_in, "own_goalie_pulled": not own_goalie_in,
                "score_diff": score_for - score_against,
                "rebound": rebound, "rush": rush,
                "assist1_id": det.get("assist1PlayerId"), "assist2_id": det.get("assist2PlayerId"),
                "blocker_id": det.get("blockingPlayerId"),
                "is_goal": ev == "goal",
            }
            shots.append(row)

            # state counters: [secs, home_sog, away_sog, home_goals, away_goals]
            key = _state_key(per, goals_by_team[home_id] - goals_by_team[away_id], sit, t)
            acc = states.setdefault(key, [0.0, 0, 0, 0, 0])
            if ev in ("sog", "goal"):
                acc[1 if is_home else 2] += 1
            if ev == "goal":
                acc[3 if is_home else 4] += 1
                goals_by_team[team] += 1
                if per <= 3:
                    reg_goals[team] += 1
                if first_goal_team is None:
                    first_goal_team = team
                if not goalie_against_in:
                    en_goals[team] += 1
                if sk_for > sk_against:  # power-play goal -> power-play points
                    for pid in (row["shooter_id"], row["assist1_id"], row["assist2_id"]):
                        if pid:
                            pp_points[pid] = pp_points.get(pid, 0) + 1

        if etype == "penalty":
            # Verified: the event owner is the team that committed the penalty.
            ag, ask, hsk, hg = sit
            penalties.append({
                "game_id": gid, "season": season, "game_type": gtype, "game_date": gdate,
                "event_id": p.get("eventId"), "period": per, "t": t,
                "team_id": det.get("eventOwnerTeamId"), "committed_by": det.get("committedByPlayerId"),
                "drawn_by": det.get("drawnByPlayerId"), "served_by": det.get("servedByPlayerId"),
                "type_code": det.get("typeCode"), "desc": det.get("descKey"), "minutes": det.get("duration"),
                "home_skaters": hsk, "away_skaters": ask,
            })

        # Goalie pulls in regulation P3 while trailing. A pull counts only when it is still in effect
        # at the next event (filters out brief delayed-penalty pulls) or that event is a goal.
        if per == 3 and gtype in (2, 3):
            ag, _ask, _hsk, hg = sit
            for team, g_in in ((home_id, hg), (away_id, ag)):
                opp = away_id if team == home_id else home_id
                deficit = goals_by_team[opp] - goals_by_team[team]
                if pulls[team] is not None:
                    continue
                cand = pending[team]
                if cand is not None:
                    if g_in == 0 or etype == "goal":
                        pulls[team] = cand
                    pending[team] = None
                elif g_in == 0 and deficit > 0 and etype != "goal":
                    pending[team] = (3 * 1200 - t, deficit)

        prev_event = (t, det.get("eventOwnerTeamId"), etype, det.get("xCoord"), det.get("yCoord"))

    last_period = (pbp.get("gameOutcome") or {}).get("lastPeriodType") or (box or {}).get(
        "gameOutcome", {}).get("lastPeriodType")
    home_score = pbp["homeTeam"].get("score")
    away_score = pbp["awayTeam"].get("score")
    games = pd.DataFrame([{
        "game_id": gid, "season": season, "game_type": gtype, "game_date": gdate,
        "start_utc": pbp.get("startTimeUTC"),
        "home_id": home_id, "away_id": away_id,
        "home_abbrev": pbp["homeTeam"].get("abbrev"), "away_abbrev": pbp["awayTeam"].get("abbrev"),
        "home_score": home_score, "away_score": away_score,
        "home_reg_goals": reg_goals[home_id], "away_reg_goals": reg_goals[away_id],
        "home_en_goals": en_goals[home_id], "away_en_goals": en_goals[away_id],
        "home_sog": pbp["homeTeam"].get("sog"), "away_sog": pbp["awayTeam"].get("sog"),
        "last_period": last_period,
        "first_goal_team": first_goal_team,
        "home_pull_remaining": pulls[home_id][0] if pulls[home_id] else None,
        "home_pull_deficit": pulls[home_id][1] if pulls[home_id] else None,
        "away_pull_remaining": pulls[away_id][0] if pulls[away_id] else None,
        "away_pull_deficit": pulls[away_id][1] if pulls[away_id] else None,
        "game_state": pbp.get("gameState"),
        "venue_utc_offset": _offset_hours(pbp.get("venueUTCOffset")),
    }])

    st_rows = [{"game_id": gid, "season": season, "game_type": gtype, "period_bucket": k[0],
                "score_diff_home": k[1], "home_goalie": k[2], "away_goalie": k[3], "sk_diff_home": k[4],
                "secs": v[0], "home_sog": v[1], "away_sog": v[2], "home_goals": v[3], "away_goals": v[4]}
               for k, v in states.items()]

    pg, gg = _parse_box(box, gid, season, gtype, gdate, pp_points) if box else (pd.DataFrame(), pd.DataFrame())
    return {"games": games, "shots": pd.DataFrame(shots), "player_games": pg, "goalie_games": gg,
            "states": pd.DataFrame(st_rows), "penalties": pd.DataFrame(penalties)}


def _offset_hours(s: str | None) -> float | None:
    """'-04:00' -> -4.0"""
    if not s:
        return None
    sign = -1 if s.startswith("-") else 1
    h, m = s.lstrip("+-").split(":")
    return sign * (int(h) + int(m) / 60)


def _state_key(period: int, score_diff_home: int, sit: tuple[int, int, int, int], t: int) -> tuple:
    ag, ask, hsk, hg = sit
    if period <= 2:
        pb = "early"
    elif period == 3:
        pb = "p3_late" if t >= 3 * 1200 - 180 else "p3"
    else:
        pb = "ot"
    return (pb, int(max(-3, min(3, score_diff_home))), hg, ag, int(max(-2, min(2, hsk - ask))))


def _parse_box(box: dict, gid: int, season: int, gtype: int, gdate: str,
               pp_points: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    skaters, goalies = [], []
    stats = box.get("playerByGameStats", {})
    for side in ("homeTeam", "awayTeam"):
        team_id = box[side]["id"]
        opp_id = box["awayTeam" if side == "homeTeam" else "homeTeam"]["id"]
        block = stats.get(side, {})
        for grp in ("forwards", "defense"):
            for p in block.get(grp, []):
                skaters.append({
                    "game_id": gid, "season": season, "game_type": gtype, "game_date": gdate,
                    "team_id": team_id, "opp_id": opp_id, "is_home": side == "homeTeam",
                    "player_id": p["playerId"], "name": p.get("name", {}).get("default"),
                    "position": p.get("position"), "toi": _secs(p.get("toi")) / 60.0,
                    "goals": p.get("goals", 0), "assists": p.get("assists", 0), "points": p.get("points", 0),
                    "sog": p.get("sog", 0), "blocks": p.get("blockedShots", 0), "hits": p.get("hits", 0),
                    "ppg": p.get("powerPlayGoals", 0), "ppp": pp_points.get(p["playerId"], 0),
                    "shifts": p.get("shifts", 0),
                })
        for p in block.get("goalies", []):
            goalies.append({
                "game_id": gid, "season": season, "game_type": gtype, "game_date": gdate,
                "team_id": team_id, "opp_id": opp_id, "is_home": side == "homeTeam",
                "player_id": p["playerId"], "name": p.get("name", {}).get("default"),
                "starter": bool(p.get("starter", False)), "toi": _secs(p.get("toi")) / 60.0,
                "shots_against": p.get("shotsAgainst", 0), "saves": p.get("saves", 0),
                "goals_against": p.get("goalsAgainst", 0),
            })
    return pd.DataFrame(skaters), pd.DataFrame(goalies)
