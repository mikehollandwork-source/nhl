"""Shift charts -> on-ice "stints", plus officials / scratches / coaches from the right-rail feed.

A stint is a stretch of game time during which the same players are on the ice for both teams.
Stints give:
  - who was on the ice for every shot (for player impact ratings, features/rapm.py);
  - each player's ice time at even strength / power play / shorthanded.

Verified (Oct 2026): stats API `shiftcharts` rows with typeCode 517 are shifts; 505 rows are goal
markers and are skipped. Times are mm:ss within the period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nhlmodel.ingest.nhl_api import game_seconds

SHIFTS_URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gid}"
RAIL_URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/right-rail"
MAX_SK = 6  # columns h1..h6 / a1..a6 (6 skaters when a goalie is pulled)


def parse_shifts(raw: dict, game_id: int, game_type: int, home_id: int, goalie_ids: set[int]) -> pd.DataFrame:
    """Shift rows -> stints (one row per interval with an unchanged on-ice group)."""
    rows = [r for r in (raw or {}).get("data", []) if r.get("typeCode") == 517 and r.get("startTime") and r.get("endTime")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame({
        "player_id": [r["playerId"] for r in rows],
        "team_id": [r["teamId"] for r in rows],
        "period": [int(r["period"]) for r in rows],
    })
    df["t0"] = [game_seconds(int(r["period"]), r["startTime"], game_type) for r in rows]
    df["t1"] = [game_seconds(int(r["period"]), r["endTime"], game_type) for r in rows]
    df = df[df.t1 > df.t0]
    if df.empty:
        return pd.DataFrame()
    df["is_goalie"] = df.player_id.isin(goalie_ids)
    df["is_home"] = df.team_id == home_id

    out = []
    for per, g in df.groupby("period"):
        b = np.unique(np.r_[g.t0.to_numpy(), g.t1.to_numpy()])
        if len(b) < 2:
            continue
        lo, hi = b[:-1], b[1:]
        on = (g.t0.to_numpy()[None, :] <= lo[:, None]) & (g.t1.to_numpy()[None, :] >= hi[:, None])
        pid = g.player_id.to_numpy()
        home = g.is_home.to_numpy()
        goalie = g.is_goalie.to_numpy()
        for i in range(len(lo)):
            m = on[i]
            hs = sorted(set(pid[m & home & ~goalie]))[:MAX_SK]
            as_ = sorted(set(pid[m & ~home & ~goalie]))[:MAX_SK]
            rec = {"period": int(per), "t0": int(lo[i]), "t1": int(hi[i]),
                   "home_goalie_in": bool((m & home & goalie).any()), "away_goalie_in": bool((m & ~home & goalie).any()),
                   "n_home": len(hs), "n_away": len(as_)}
            for k in range(MAX_SK):
                rec[f"h{k + 1}"] = hs[k] if k < len(hs) else 0
                rec[f"a{k + 1}"] = as_[k] if k < len(as_) else 0
            out.append(rec)
    st = pd.DataFrame(out)
    if st.empty:
        return st
    # merge consecutive intervals with identical groups (shift-record splits)
    key_cols = ["period", "home_goalie_in", "away_goalie_in"] + [f"h{k + 1}" for k in range(MAX_SK)] + [f"a{k + 1}" for k in range(MAX_SK)]
    same = (st[key_cols] == st[key_cols].shift()).all(axis=1) & (st.t0 == st.t1.shift())
    grp = (~same).cumsum()
    agg = {c: "first" for c in st.columns}
    agg["t1"] = "last"
    st = st.groupby(grp).agg(agg).reset_index(drop=True)
    st["secs"] = st.t1 - st.t0
    st.insert(0, "game_id", game_id)
    return st


def strength_toi(stints: pd.DataFrame, home_id: int, away_id: int) -> pd.DataFrame:
    """Per player: seconds at 5v5, other even strength, power play, shorthanded (goalies in)."""
    if stints.empty:
        return pd.DataFrame()
    s = stints[stints.home_goalie_in & stints.away_goalie_in]
    recs = []
    for side, team, opp_n, own_n in (("h", home_id, "n_away", "n_home"), ("a", away_id, "n_home", "n_away")):
        diff = s[own_n] - s[opp_n]
        state = np.where((s.n_home == 5) & (s.n_away == 5), "s5v5",
                         np.where(diff == 0, "ev_other", np.where(diff > 0, "pp", "sh")))
        for k in range(MAX_SK):
            col = s[f"{side}{k + 1}"]
            m = col > 0
            recs.append(pd.DataFrame({"player_id": col[m].to_numpy(), "team_id": team,
                                      "state": state[m.to_numpy()], "secs": s.secs[m].to_numpy()}))
    r = pd.concat(recs, ignore_index=True)
    out = r.pivot_table(index=["player_id", "team_id"], columns="state", values="secs", aggfunc="sum", fill_value=0)
    for c in ("s5v5", "ev_other", "pp", "sh"):
        if c not in out:
            out[c] = 0
    out = out.reset_index().rename(columns={"s5v5": "toi_5v5_s", "ev_other": "toi_evo_s", "pp": "toi_pp_s",
                                            "sh": "toi_sh_s"})
    out.columns.name = None
    return out


def parse_right_rail(raw: dict, game_id: int) -> tuple[dict, list[dict]]:
    """-> (officials row, scratch rows). Empty lists before warmups (verified)."""
    gi = (raw or {}).get("gameInfo") or {}
    refs = [r.get("fullName", {}).get("default") for r in gi.get("referees", [])]
    lines = [r.get("fullName", {}).get("default") for r in gi.get("linesmen", [])]
    off = {"game_id": game_id,
           "ref1": refs[0] if len(refs) > 0 else None, "ref2": refs[1] if len(refs) > 1 else None,
           "lines1": lines[0] if len(lines) > 0 else None, "lines2": lines[1] if len(lines) > 1 else None,
           "home_coach": (gi.get("homeTeam") or {}).get("headCoach", {}).get("default"),
           "away_coach": (gi.get("awayTeam") or {}).get("headCoach", {}).get("default")}
    scratches = []
    for side in ("homeTeam", "awayTeam"):
        for p in (gi.get(side) or {}).get("scratches", []) or []:
            scratches.append({"game_id": game_id, "side": side[:4], "player_id": p.get("id"),
                              "name": f"{p.get('firstName', {}).get('default', '')} {p.get('lastName', {}).get('default', '')}".strip()})
    return off, scratches
