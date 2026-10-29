"""Project a day's slate with the phase-2 model.

Uses DailyFaceoff lines/injuries for tonight's lineups (falls back to each team's last lineup),
special-teams-aware simulation, and props allocated by strength state (EV / PP / SH).
Also writes a small cache (data/daily_cache/<date>.json) that the warmup job uses to re-run a game
quickly once referees, scratches and confirmed goalies are posted.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import data_dir, settings
from nhlmodel.features import assemble as A
from nhlmodel.features import history as H
from nhlmodel.features import lineup as LU
from nhlmodel.features import live as LV
from nhlmodel.features import players as PL
from nhlmodel.features.xg import XGModel
from nhlmodel.ingest import dailyfaceoff as DF
from nhlmodel.ingest import lineups as LN
from nhlmodel.ingest import nhl_api
from nhlmodel.sim import markets as M
from nhlmodel.sim import props as P
from nhlmodel.sim.engine import simulate
from nhlmodel.sim.model import GameModelV2

log = logging.getLogger("slate_v2")


def available() -> bool:
    return settings().get("model", {}).get("version", 1) == 2 and \
        (data_dir() / "models" / "game_model_v2.json").exists() and (data_dir() / "models" / "rapm.parquet").exists()


def load() -> tuple[XGModel, GameModelV2, pd.DataFrame]:
    mdir = data_dir() / "models"
    return XGModel.load(mdir), GameModelV2.load(mdir / "game_model_v2.json"), pd.read_parquet(mdir / "rapm.parquet")


def team_lineup(h: H.History, team_id: int, abbrev: str, place: str, common: str, roster: pd.DataFrame,
                fallback) -> tuple[pd.DataFrame, list[str], str, set[int]]:
    """-> (players with player_id, name, pp_unit, p_play), notes for the card, source label,
    ids listed out / injured reserve."""
    lines, _upd = LN.fetch_team(LN.team_slug(place, common))
    summ = LN.summarize(lines, roster) if not lines.empty else pd.DataFrame()
    notes = []
    injured: set[int] = set()
    if not summ.empty:
        injured = {int(x) for x in summ[summ.injury.isin(["ir", "out"]) & summ.player_id.notna()].player_id}
        out = summ[summ.injury.isin(["ir", "out"]) & summ.player_id.notna()]
        dtd = summ[(summ.injury == "dtd") | summ.gtd]
        if len(out):
            notes.append("Out: " + ", ".join(out.name.head(6)))
        if len(dtd):
            notes.append("Game-time: " + ", ".join(dtd.name.head(4)))
        pp1 = summ[summ.pp_unit == 1].name.tolist()
        if pp1:
            notes.append("PP1: " + ", ".join(n.split()[-1] for n in pp1))
        sk = summ[summ.dressed & ~summ.is_goalie & summ.player_id.notna()].copy()
        if len(sk) >= 15:
            sk["player_id"] = sk.player_id.astype(int)
            sk["p_play"] = np.where(sk.gtd | (sk.injury == "dtd"), LV.gtd_play_prob(), 1.0)
            return sk[["player_id", "name", "pp_unit", "pk_unit", "p_play"]].reset_index(drop=True), notes, "DailyFaceoff", injured
    lu = fallback(h, team_id, roster)
    lu = lu[~lu.player_id.isin(injured)].assign(pp_unit=-1, pk_unit=-1, p_play=1.0)
    return lu, notes, "last game", injured


def healthy_goalie(goalie: tuple, injured: set[int], h: H.History, team_id: int, d: date, roster: pd.DataFrame,
                   b2b: bool) -> tuple:
    """If the projected starter is listed out / on IR, fall back to our guess among healthy goalies."""
    gid, _name, _status = goalie
    if gid is None or gid not in injured:
        return goalie
    ids = set(roster[roster.position == "G"].player_id) - injured if not roster.empty else None
    alt = DF.guess_starter(h.goalie_games, team_id, d, b2b, ids)
    if alt is None:
        return goalie
    names = roster.set_index("player_id").name.to_dict() if not roster.empty else {}
    log.warning("starter %s listed injured; using %s", gid, alt)
    return alt, names.get(alt, str(alt)), "guessed"


def prop_weights(h: H.History, events: pd.DataFrame, ex: A.Extras, d: date, lu: pd.DataFrame,
                 pp_units: dict) -> pd.DataFrame:
    r = PL.player_state_rates(h.player_games, events, ex.player_toi, d, lu.player_id.tolist())
    r = r.merge(lu[["player_id", "name", "pp_unit"]], on="player_id", how="left")
    # tonight's power-play role from DailyFaceoff overrides history when known
    known = r.pp_unit >= 0
    r.loc[known & (r.pp_unit == 1), "min_pp"] = np.maximum(r.min_pp, pp_units.get("pp1_min", 3.0))
    r.loc[known & (r.pp_unit == 2), "min_pp"] = np.clip(r.min_pp, 0.5 * pp_units.get("pp2_min", 1.5),
                                                       pp_units.get("pp2_min", 1.5) * 1.5)
    r.loc[known & (r.pp_unit == 0), "min_pp"] = r.min_pp * 0.25
    return PL.state_weights(r)


def project(d: date, games: list[dict], lineup_fallback, resolve_goalie, rest_flags) -> tuple[list, GameModelV2, dict]:
    """Returns (list of GameRun-like dicts, model, cache)."""
    xg, gm, rapm = load()
    h = H.load(xg)
    ex = A.load_extras()
    ctx = LV.build_context(h, ex, rapm, d)
    df_goalies = DF.fetch(d)
    events = PL.player_state_events(h.shots)
    a_state = PL.league_assist_rates_by_state(h.shots[pd.to_datetime(h.shots.game_date).dt.date >= d - timedelta(days=365)])
    pp_units = gm.meta.get("pp_units", {})
    lines_cfg = settings()["props"]["lines"]
    n_sims = settings()["sim"]["n_sims"]
    rosters: dict[str, pd.DataFrame] = {}
    runs, cache = [], {"date": d.isoformat(), "model_version": gm.version, "games": {}}
    for g in games:
        home, away = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        hid, aid = g["homeTeam"]["id"], g["awayTeam"]["id"]
        for t in (home, away):
            if t not in rosters:
                try:
                    rosters[t] = nhl_api.roster(t)
                except Exception as e:  # keep going without the roster
                    log.warning("roster %s failed: %s", t, e)
                    rosters[t] = pd.DataFrame()
        hl, hnotes, hsrc, hinj = team_lineup(h, hid, home, g["homeTeam"].get("placeName", {}).get("default", ""),
                                       g["homeTeam"].get("commonName", {}).get("default", ""), rosters[home], lineup_fallback)
        al, anotes, asrc, ainj = team_lineup(h, aid, away, g["awayTeam"].get("placeName", {}).get("default", ""),
                                       g["awayTeam"].get("commonName", {}).get("default", ""), rosters[away], lineup_fallback)
        offset = nhl_api._offset_hours(g.get("venueUTCOffset"))
        hrest, arest = rest_flags(h.team_games, hid, d), rest_flags(h.team_games, aid, d)
        hg = healthy_goalie(resolve_goalie(home, hid, df_goalies, rosters[home], h, d), hinj, h, hid, d,
                            rosters[home], bool(hrest["b2b"]))
        ag = healthy_goalie(resolve_goalie(away, aid, df_goalies, rosters[away], h, d), ainj, h, aid, d,
                            rosters[away], bool(arest["b2b"]))
        hr = LV.team_row(ctx, hid, aid, True, d, hrest, arest, ag[0], hl, al, offset)
        ar = LV.team_row(ctx, aid, hid, False, d, arest, hrest, hg[0], al, hl, offset)
        gi = gm.game_input(hr, ar, int(g["gameType"]))
        rng = np.random.default_rng(settings()["sim"]["seed"] + int(g["id"]) % 1_000_000)
        sim = simulate(gi, gm.constants, n=n_sims, step=settings()["sim"]["step_seconds"], rng=rng)

        props = None
        tables = []
        for side, lu, abbrev, goalie in (("home", hl, home, hg), ("away", al, away, ag)):
            if lu.empty:
                continue
            w = prop_weights(h, events, ex, d, lu, pp_units).assign(team=abbrev)
            tid = hid if side == "home" else aid
            tp = P.TeamPropsV2(w, a_state, PL.team_block_ratio(h.player_games, h.games, tid, d),
                               float(settings()["props"].get("player_var", 0.0)))
            stats = P.team_player_sims_v2(sim, side, tp, rng)
            tables.append(P.prop_table(stats, w, sim.first_goal == (1 if side == "home" else -1), lines_cfg))
            opp = "a" if side == "home" else "h"
            saves = getattr(sim, f"sog_{opp}") - getattr(sim, f"goals_{opp}")
            tables.append(P.goalie_table(saves, goalie[0], goalie[1], abbrev, lines_cfg["saves"]))
        if tables:
            props = pd.concat(tables, ignore_index=True).assign(game_id=int(g["id"]), home=home, away=away)
            props = P.recalibrate(props, settings()["props"].get("recal"))

        notes = [f"{home} " + n for n in hnotes] + [f"{away} " + n for n in anotes]
        if hsrc != "DailyFaceoff" or asrc != "DailyFaceoff":
            notes.append(f"Lineup source: {home} {hsrc}, {away} {asrc}")
        runs.append({"game_id": int(g["id"]), "game_type": int(g["gameType"]), "start_utc": g["startTimeUTC"],
                     "home": home, "away": away, "home_id": hid, "away_id": aid,
                     "home_goalie_id": hg[0], "away_goalie_id": ag[0], "home_goalie": hg[1], "away_goalie": ag[1],
                     "home_goalie_status": hg[2], "away_goalie_status": ag[2], "sim": sim,
                     "summary": M.summary(sim), "props": props, "notes": notes})
        goalie_opts = {}
        for team, tid in ((home, hid), (away, aid)):
            ros = rosters[team]
            gids = ros[ros.position == "G"].player_id.tolist() if not ros.empty else []
            names = ros.set_index("player_id").name.to_dict() if not ros.empty else {}
            goalie_opts[str(tid)] = {str(x): {**LV.goalie_features(ctx, int(x), d), "name": names.get(x, str(x))}
                                     for x in gids}
        cache["games"][str(g["id"])] = {
            "home": home, "away": away, "home_id": hid, "away_id": aid, "game_type": int(g["gameType"]),
            "home_row": _jsonable(hr), "away_row": _jsonable(ar),
            "home_goalie_id": hg[0], "away_goalie_id": ag[0],
            "home_lineup": hl[["player_id", "p_play"]].to_dict("records"),
            "away_lineup": al[["player_id", "p_play"]].to_dict("records"),
            "goalie_options": goalie_opts, "p_home": float(sim.home_win.mean()),
            "exp_total": float((sim.final_h + sim.final_a).mean()),
        }
        log.info("%s @ %s: home %.3f total %.2f", away, home, sim.home_win.mean(), (sim.final_h + sim.final_a).mean())
    # what the warmup job needs to recompute lineup deltas without reloading history
    players = {int(p["player_id"]) for gc in cache["games"].values() for side in ("home_lineup", "away_lineup")
               for p in gc[side]}
    inputs = LV.player_inputs(ctx, pd.DataFrame({"player_id": sorted(players)}))
    cache["players"] = {str(int(r.player_id)): {k: float(getattr(r, k)) for k in
                        ("proj_toi", "proj_pp_min", "pp_xg60", "ev_xg60", "finishing")}
                        for r in inputs.itertuples(index=False)}
    cache["repl"] = ctx.repl
    cache["typical"] = {str(t): LU.typical_impact(ctx.lineup_hist, t)
                        for gc in cache["games"].values() for t in (gc["home_id"], gc["away_id"])} \
        if not ctx.lineup_hist.empty else {}
    cache["typical0"] = {str(t): LU.typical_impact(ctx.lineup_hist0, t)
                         for gc in cache["games"].values() for t in (gc["home_id"], gc["away_id"])} \
        if not ctx.lineup_hist0.empty else {}
    cache["ref_factors"] = ctx.ref_latest
    return runs, gm, cache


def _jsonable(s: pd.Series) -> dict:
    out = {}
    for k, v in s.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, date):
            v = v.isoformat()
        out[k] = v
    return out


def save_cache(cache: dict) -> None:
    p = data_dir() / "daily_cache"
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{cache['date']}.json").write_text(json.dumps(cache, default=float))
