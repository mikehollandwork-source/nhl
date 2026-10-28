"""Near-puck-drop update.

python -m nhlmodel.pipelines.warmup

Runs every 15 minutes during game windows. For each game in warmups (NHL game state PRE),
it reads what only becomes public then (verified): referees and scratches from the right-rail
feed, plus DailyFaceoff's confirmed starters. It re-runs the game from the morning cache
(data/daily_cache/<date>.json, no history reload) and posts a "final" update only when something
material changed: the moneyline moved >= warmup.ml_change, the starting goalie changed, or a
player from the projected lineup was scratched. No odds calls (free-plan credits are saved for
the scheduled snapshots). Each game is processed once.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd

from nhlmodel.config import data_dir, season_for, settings, today_local
from nhlmodel.features import lineup as LU
from nhlmodel.features import officials as OF
from nhlmodel.ingest import dailyfaceoff as DF
from nhlmodel.ingest import nhl_api
from nhlmodel.ingest.http import get_json
from nhlmodel.ingest.shifts import RAIL_URL, parse_right_rail
from nhlmodel.publish import telegram as TG
from nhlmodel.sim import markets as M
from nhlmodel.sim.engine import simulate
from nhlmodel.sim.model import GameModelV2
from nhlmodel.store import db

log = logging.getLogger("warmup")
LEAD_MINUTES = 35


def due_games(d: date, now: datetime) -> list[dict]:
    out = []
    for g in nhl_api.schedule_day(d):
        start = datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00"))
        mins = (start - now).total_seconds() / 60
        if g.get("gameState") == "PRE" or (g.get("gameState") == "FUT" and 0 <= mins <= LEAD_MINUTES):
            out.append(g)
    return out


def lineup_deltas(gc: dict, side: str, scratched: set[int], cache: dict, rapm: pd.DataFrame) -> tuple[dict, list[int]]:
    """Recompute a team's lineup deltas after scratches. Each scratched player is replaced by a
    replacement-level skater with the same ice time (call-ups are usually below average)."""
    team = gc[f"{side}_id"]
    lu = pd.DataFrame(gc[f"{side}_lineup"])
    removed = [int(p) for p in lu.player_id if int(p) in scratched]
    info = cache.get("players", {})
    rows = []
    for rec in lu.itertuples(index=False):
        f = info.get(str(int(rec.player_id)), {})
        base = {"proj_toi": f.get("proj_toi", 800.0), "proj_pp_min": f.get("proj_pp_min", 0.0),
                "pp_xg60": f.get("pp_xg60", 0.5), "ev_xg60": f.get("ev_xg60", 0.5), "finishing": f.get("finishing", 1.0)}
        if int(rec.player_id) in scratched:
            rows.append({**base, "player_id": -1, "p_play": 1.0, "proj_pp_min": 0.0, "finishing": 1.0,
                         "pp_xg60": base["pp_xg60"] * 0.5})
        else:
            rows.append({**base, "player_id": int(rec.player_id), "p_play": rec.p_play})
    df = pd.DataFrame(rows)
    now = LU.impact_of(df, rapm, cache.get("repl"))
    typ = cache.get("typical", {}).get(str(team), now)
    out = {f"d_{c}": now[c] - typ.get(c, now[c]) for c in LU.ALL}
    now0 = LU.impact_of(df, rapm, None)
    typ0 = cache.get("typical0", {}).get(str(team), now0)
    out.update({f"d0_{c}": now0[c] - typ0.get(c, now0[c]) for c in LU.IMPACTS})
    return out, removed


def process(g: dict, gc: dict, cache: dict, gm: GameModelV2, rapm: pd.DataFrame, df_goalies: pd.DataFrame,
            names: dict[int, str]) -> tuple[str | None, dict]:
    gid = int(g["id"])
    rail = get_json(RAIL_URL.format(gid=gid))
    off, scr = parse_right_rail(rail, gid)
    scratched = {int(s["player_id"]) for s in scr if s.get("player_id")}
    hr, ar = pd.Series(gc["home_row"]), pd.Series(gc["away_row"])
    changes = []

    rf = OF.crew_factor(cache.get("ref_factors", {}), off.get("ref1"), off.get("ref2"))
    hr["ref_factor"] = ar["ref_factor"] = rf

    # confirmed goalies
    goalies = {"home": gc["home_goalie_id"], "away": gc["away_goalie_id"]}
    for side, tid in (("home", gc["home_id"]), ("away", gc["away_id"])):
        abbrev = gc[side]
        row = df_goalies[(df_goalies.team == abbrev) & (df_goalies.status == "confirmed")] if not df_goalies.empty else df_goalies
        if row is None or row.empty:
            continue
        opts = gc["goalie_options"].get(str(tid), {})
        match = [int(k) for k, v in opts.items() if nhl_api.name_key(v.get("name", "")) == row.goalie_key.iloc[0]]
        if match and match[0] != goalies[side]:
            goalies[side] = match[0]
            feats = {k: v for k, v in opts[str(match[0])].items() if k != "name"}
            target = ar if side == "home" else hr   # a team's goalie is the OTHER row's "opp goalie"
            for k, v in feats.items():
                target[k] = v
            changes.append(f"{abbrev} goalie now {opts[str(match[0])]['name']} (confirmed)")

    removed_all = []
    for side, row, other in (("home", hr, ar), ("away", ar, hr)):
        deltas, removed = lineup_deltas(gc, side, scratched, cache, rapm)
        if removed:
            for k, v in deltas.items():
                row[k] = v
            for c in ("d_def_xg", "d_def_sog", "d0_def_xg", "d0_def_sog"):
                other[f"o_{c}"] = deltas[c]
            removed_all += [f"{gc[side]} {names.get(p, p)}" for p in removed]
    if removed_all:
        changes.append("Scratched from projected lineup: " + ", ".join(removed_all[:6]))

    gi = gm.game_input(hr, ar, gc["game_type"])
    sim = simulate(gi, gm.constants, n=20000, step=settings()["sim"]["step_seconds"],
                   rng=np.random.default_rng(settings()["sim"]["seed"] + gid % 1_000_000))
    s = M.summary(sim)
    dp = s["p_home"] - gc["p_home"]
    total = s["exp_goals_home"] + s["exp_goals_away"]
    material = abs(dp) >= settings()["warmup"]["ml_change"] or bool(changes)
    refs = " / ".join(x for x in (off.get("ref1"), off.get("ref2")) if x) or "not posted"
    msg = None
    if material:
        msg = (f"🔒 <b>Final: {gc['away']} @ {gc['home']}</b>\n"
               + "".join(f"• {c}\n" for c in changes)
               + f"Referees: {refs} (crew penalty factor {rf:.2f})\n"
               f"{gc['home']} win {100 * gc['p_home']:.0f}% → <b>{100 * s['p_home']:.0f}%</b> · "
               f"total {gc['exp_total']:.2f} → <b>{total:.2f}</b>")
    rec = {"game_date": cache["date"], "game_id": gid, "home": gc["home"], "away": gc["away"],
           "home_goalie_id": goalies["home"], "away_goalie_id": goalies["away"], "ref1": off.get("ref1"),
           "ref2": off.get("ref2"), "ref_factor": rf, "p_home": s["p_home"], "exp_total": total,
           "material": material, "run_type": "warmup", "ts": datetime.now(UTC).isoformat(timespec="seconds")}
    return msg, rec


def run() -> None:
    d = today_local()
    path = data_dir() / "daily_cache" / f"{d.isoformat()}.json"
    if not path.exists():
        log.info("no morning cache for %s; nothing to do", d)
        return
    now = datetime.now(UTC)
    games = due_games(d, now)
    done = db.read("warmups", partitions=[d.isoformat()])
    done_ids = set(done.game_id) if not done.empty else set()
    games = [g for g in games if int(g["id"]) not in done_ids]
    if not games:
        log.info("no games in warmups")
        return
    cache = json.loads(path.read_text())
    mdir = data_dir() / "models"
    gm = GameModelV2.load(mdir / "game_model_v2.json")
    rapm = pd.read_parquet(mdir / "rapm.parquet")
    df_goalies = DF.fetch(d)
    pg = db.read("player_games", partitions=[str(season_for(d))])
    names = pg.drop_duplicates("player_id").set_index("player_id").name.to_dict() if not pg.empty else {}
    msgs, recs = [], []
    for g in games:
        gc = cache["games"].get(str(g["id"]))
        if gc is None:
            continue
        try:
            msg, rec = process(g, gc, cache, gm, rapm, df_goalies, names)
        except Exception:
            log.exception("warmup failed for %s", g["id"])
            continue
        if rec["ref1"] is None and g.get("gameState") == "FUT":
            continue  # officials not posted yet; try again next run
        recs.append(rec)
        if msg:
            msgs.append(msg)
    if recs:
        db.append("warmups", pd.DataFrame(recs), d.isoformat(), key=["game_id"])
    if msgs:
        TG.send_message("\n\n".join(msgs))
    log.info("processed %d games, %d updates posted", len(recs), len(msgs))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run()


if __name__ == "__main__":
    main()
