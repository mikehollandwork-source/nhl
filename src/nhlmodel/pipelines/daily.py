"""Daily projections.

python -m nhlmodel.pipelines.daily --run morning   # full slate, opening lines, everything posted
python -m nhlmodel.pipelines.daily --run pregame   # goalie re-check + line snapshot; posts only changes

Set NHL_DATE=YYYY-MM-DD to run for another date, NHL_DRY_RUN=1 to print instead of posting.
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import data_dir, out_dir, season_for, settings, today_local, tz
from nhlmodel.features import history as H
from nhlmodel.features import players as PL
from nhlmodel.features import team as T
from nhlmodel.features.xg import XGModel
from nhlmodel.ingest import dailyfaceoff as DF
from nhlmodel.ingest import exchanges, kalshi, nhl_api
from nhlmodel.market.devig import cents_to_american, prob_to_american
from nhlmodel.pipelines import once, slate_v2
from nhlmodel.pipelines.backfill import backfill_season
from nhlmodel.publish import format as F
from nhlmodel.publish import notify as PUB
from nhlmodel.sim import markets as M
from nhlmodel.sim import props as P
from nhlmodel.sim.engine import SimResult, simulate
from nhlmodel.sim.params import GameModel
from nhlmodel.store import db
from nhlmodel.tracking import disagree as DG

log = logging.getLogger("daily")


@dataclass
class GameRun:
    game_id: int
    game_type: int
    start_utc: str
    home: str
    away: str
    home_id: int
    away_id: int
    home_goalie_id: int | None
    away_goalie_id: int | None
    home_goalie: str
    away_goalie: str
    home_goalie_status: str
    away_goalie_status: str
    sim: SimResult | None = None
    summary: dict | None = None
    props: pd.DataFrame | None = None
    notes: list | None = None


# ----------------------------------------------------------------------------- inputs

def load_models() -> tuple[XGModel, GameModel]:
    mdir = data_dir() / "models"
    if not (mdir / "game_model.json").exists():
        raise SystemExit("No trained model found. Run: python -m nhlmodel.pipelines.train --mode production")
    return XGModel.load(mdir), GameModel.load(mdir / "game_model.json")


def rest_flags(tg: pd.DataFrame, team_id: int, d: date) -> dict:
    played = set(tg.loc[tg.team_id == team_id, "game_date"])
    b2b = (d - timedelta(days=1)) in played
    n3 = sum((d - timedelta(days=k)) in played for k in (1, 2, 3))
    return {"b2b": int(b2b), "three_in_four": int(n3 >= 2)}


def team_row(h: H.History, team_id: int, opp_id: int, is_home: bool, d: date, lg: dict,
             opp_goalie_factor: float) -> pd.Series:
    tr = h.team_ratings
    me = tr[(tr.team_id == team_id) & (tr.game_id == -1)]
    op = tr[(tr.team_id == opp_id) & (tr.game_id == -1)]
    r = {}
    for m in T.RATED:
        r[f"r_{m}"] = float(me[f"r_{m}"].iloc[0]) if len(me) else lg[f"lg_{m}"]
        r[f"o_{m}"] = float(op[f"r_{m}"].iloc[0]) if len(op) else lg[f"lg_{m}"]
    mine, theirs = rest_flags(h.team_games, team_id, d), rest_flags(h.team_games, opp_id, d)
    return pd.Series({**r, **lg, "is_home": int(is_home), "b2b": mine["b2b"],
                      "three_in_four": mine["three_in_four"], "opp_b2b": theirs["b2b"],
                      "opp_three_in_four": theirs["three_in_four"], "opp_goalie_factor": opp_goalie_factor})


def resolve_goalie(team: str, team_id: int, df_goalies: pd.DataFrame, roster: pd.DataFrame,
                   h: H.History, d: date) -> tuple[int | None, str, str]:
    """(goalie_id, name, status) using DailyFaceoff first, our guess second."""
    goalies = roster[roster.position == "G"] if not roster.empty else roster
    if not df_goalies.empty:
        row = df_goalies[df_goalies.team == team]
        if len(row):
            key = row.goalie_key.iloc[0]
            hit = goalies[goalies.name_key == key] if not goalies.empty else goalies
            if len(hit):
                return int(hit.player_id.iloc[0]), str(hit.name.iloc[0]), str(row.status.iloc[0])
            log.warning("%s: DailyFaceoff goalie %s not on roster; guessing", team, row.goalie_name.iloc[0])
    b2b = bool(rest_flags(h.team_games, team_id, d)["b2b"])
    ids = set(goalies.player_id) if not goalies.empty else None
    gid = DF.guess_starter(h.goalie_games, team_id, d, b2b, ids)
    name = "unknown"
    if gid is not None and not goalies.empty and (goalies.player_id == gid).any():
        name = str(goalies.loc[goalies.player_id == gid, "name"].iloc[0])
    elif gid is not None:
        nm = h.goalie_games.loc[h.goalie_games.player_id == gid, "name"]
        name = str(nm.iloc[-1]) if len(nm) else str(gid)
    return gid, name, "guessed"


def lineup(h: H.History, team_id: int, roster: pd.DataFrame) -> pd.DataFrame:
    """Skaters who dressed in the team's most recent game and are still on the roster."""
    pg = h.player_games[h.player_games.team_id == team_id]
    if pg.empty:
        return pd.DataFrame(columns=["player_id", "name"])
    last_game = pg.sort_values("game_date").game_id.iloc[-1]
    ids = pg.loc[pg.game_id == last_game, ["player_id", "name"]]
    if not roster.empty:
        ids = ids[ids.player_id.isin(roster.player_id)]
        full = roster.set_index("player_id")["name"]
        ids = ids.assign(name=ids.player_id.map(full).fillna(ids.name))
    return ids.reset_index(drop=True)


# ----------------------------------------------------------------------------- core

def project_slate(d: date, run_type: str) -> tuple[list[GameRun], GameModel, H.History]:
    games = [g for g in nhl_api.schedule_day(d) if g.get("gameType") in (2, 3)]
    if run_type == "pregame":
        now = datetime.now(UTC)
        games = [g for g in games if datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00")) > now]
    if not games:
        return [], None, None
    if slate_v2.available():
        dicts, gm2, cache = slate_v2.project(d, games, lineup, resolve_goalie, rest_flags)
        slate_v2.save_cache(cache)
        return [GameRun(**x) for x in dicts], gm2, None
    xg, gm = load_models()
    h = H.load(xg)
    lg = T.latest_league_average(h.team_games, d)
    df_goalies = DF.fetch(d)
    ab_rosters: dict[str, pd.DataFrame] = {}
    lines_cfg = settings()["props"]["lines"]
    a_rates = PL.league_assist_rates(h.shots[pd.to_datetime(h.shots.game_date).dt.date >= d - timedelta(days=365)])
    ixg = PL.player_ixg(h.shots)
    cfg_sim = settings()["sim"]
    runs = []
    for g in games:
        home, away = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        hid, aid = g["homeTeam"]["id"], g["awayTeam"]["id"]
        for t in (home, away):
            if t not in ab_rosters:
                try:
                    ab_rosters[t] = nhl_api.roster(t)
                except Exception as e:
                    log.warning("roster %s failed: %s", t, e)
                    ab_rosters[t] = pd.DataFrame()
        hg_id, hg_name, hg_st = resolve_goalie(home, hid, df_goalies, ab_rosters[home], h, d)
        ag_id, ag_name, ag_st = resolve_goalie(away, aid, df_goalies, ab_rosters[away], h, d)
        hgf, _ = T.latest_goalie_factor(h.goalie_ratings, hg_id)
        agf, _ = T.latest_goalie_factor(h.goalie_ratings, ag_id)
        hr = team_row(h, hid, aid, True, d, lg, opp_goalie_factor=agf)
        ar = team_row(h, aid, hid, False, d, lg, opp_goalie_factor=hgf)
        gi = gm.game_input(hr, ar, int(g["gameType"]))
        rng = np.random.default_rng(cfg_sim["seed"] + int(g["id"]) % 1_000_000)
        sim = simulate(gi, gm.constants, n=cfg_sim["n_sims"], step=cfg_sim["step_seconds"], rng=rng)
        run = GameRun(int(g["id"]), int(g["gameType"]), g["startTimeUTC"], home, away, hid, aid,
                      hg_id, ag_id, hg_name, ag_name, hg_st, ag_st, sim=sim, summary=M.summary(sim))

        # props
        tps = {}
        for side, tid, abbrev in (("home", hid, home), ("away", aid, away)):
            lu = lineup(h, tid, ab_rosters[abbrev])
            if lu.empty:
                continue
            rates = PL.player_rates(h.player_games, ixg, d, lu.player_id.tolist())
            rates = rates.merge(lu, on="player_id", how="left").assign(team=abbrev)
            tps[side] = P.TeamProps(
                players=rates, p_a1=a_rates["p_a1"], p_a2_given_a1=a_rates["p_a2_given_a1"],
                pp_goal_frac=PL.team_pp_goal_frac(h.shots, tid, d, a_rates["pp_goal_frac"]),
                pp_pts_per_goal=a_rates["pp_pts_per_goal"],
                block_ratio=PL.team_block_ratio(h.player_games, h.games, tid, d))
        if len(tps) == 2:
            hs, as_, sv_h, sv_a = P.simulate_props(sim, tps["home"], tps["away"], rng)
            run.props = pd.concat([
                P.prop_table(hs, tps["home"].players, sim.first_goal == 1, lines_cfg),
                P.prop_table(as_, tps["away"].players, sim.first_goal == -1, lines_cfg),
                P.goalie_table(sv_h, hg_id, hg_name, home, lines_cfg["saves"]),
                P.goalie_table(sv_a, ag_id, ag_name, away, lines_cfg["saves"]),
            ], ignore_index=True).assign(game_id=run.game_id, home=home, away=away)
        runs.append(run)
        log.info("%s @ %s: %s", away, home, {k: round(v, 3) for k, v in run.summary.items()})
    return runs, gm, h


def projection_rows(runs: list[GameRun], d: date, run_id: str, run_type: str, version: str) -> pd.DataFrame:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for r in runs:
        base = {"run_id": run_id, "ts": ts, "run_type": run_type, "game_date": d.isoformat(), "game_id": r.game_id,
                "home": r.home, "away": r.away, "model_version": version,
                "home_goalie": r.home_goalie, "away_goalie": r.away_goalie,
                "home_goalie_status": r.home_goalie_status, "away_goalie_status": r.away_goalie_status}
        items = [("h2h", "home", None), ("h2h", "away", None)]
        items += [("regulation", s, None) for s in ("home", "draw", "away")]
        items += [("spreads", s, ln) for s in ("home", "away") for ln in (-1.5, 1.5)]
        items += [("totals", s, ln) for s in ("over", "under") for ln in (5.5, 6.0, 6.5)]
        for market, sel, line in items:
            p = M.market_prob(r.sim, market, sel, line)
            rows.append({**base, "market": market, "selection": sel, "line": line, "model_prob": p,
                         "fair_odds": prob_to_american(p)})
        for k, v in r.summary.items():
            rows.append({**base, "market": "summary", "selection": k, "line": None, "model_prob": v,
                         "fair_odds": None})
    return pd.DataFrame(rows)


def fetch_and_compare(runs: list[GameRun], d: date, snapshot_type: str, run_id: str, run_type: str,
                      version: str) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """Pull game lines from Polymarket and Kalshi (no keys; prices are buy prices as American odds),
    store the snapshot, and compare with the model. Returns (disagreements, note, consensus)."""
    none = pd.DataFrame()
    slate = {(r.home, r.away): r for r in runs}
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    upcoming = [r for r in runs if not r.start_utc or pd.Timestamp(r.start_utc) > pd.Timestamp(ts)]
    try:
        odds = exchanges.game_odds(d, [(r.home, r.away, r.start_utc) for r in upcoming], team_names(), snapshot_type)
    except Exception as e:
        log.exception("exchange odds failed")
        return none, f"Lines: fetch failed ({type(e).__name__})", none
    if odds.empty:
        return none, "Lines: no Polymarket or Kalshi lines yet", none
    db.append("odds_snapshots", odds.assign(game_date=d.isoformat(), run_id=run_id), d.isoformat())

    fair = DG.per_book_fair(odds)
    cons = DG.consensus(fair)
    if cons.empty:
        return none, "Lines: no complete two-way markets", none
    db.append("consensus", cons.assign(ts=ts, snapshot_type=snapshot_type, game_date=d.isoformat()), d.isoformat())

    def model_prob(home, away, market, sel, line):
        r = slate.get((home, away))
        if r is None or market not in ("h2h", "spreads", "totals"):
            return None
        return M.market_prob(r.sim, market, sel, line)

    dis = DG.compare(cons, model_prob)
    if dis.empty:
        return dis, "Lines: no comparable lines", cons
    status = {(r.home, r.away): (r.home_goalie_status, r.away_goalie_status) for r in runs}
    dis = dis.assign(run_id=run_id, run_type=run_type, ts=ts, game_date=d.isoformat(), model_version=version,
                     goalie_status=[_combined_status(*status[(h, a)]) for h, a in zip(dis.home, dis.away, strict=True)])
    db.append("disagreements", dis, d.isoformat())
    books = " + ".join(sorted({"polymarket": "Polymarket", "kalshi": "Kalshi"}.get(b, b) for b in odds.book.unique()))
    return dis, f"Lines: {cons.event_id.nunique()} games from {books} (buy prices)", cons


def _combined_status(h: str, a: str) -> str:
    order = ["guessed", "unconfirmed", "likely", "confirmed"]
    return min(h, a, key=order.index)


_TEAMS: pd.DataFrame | None = None


def team_names() -> pd.DataFrame:
    """abbrev + nickname for all teams (from the standings endpoint)."""
    global _TEAMS
    if _TEAMS is None:
        data = nhl_api.get_json(f"{nhl_api.WEB}/standings/now")
        _TEAMS = pd.DataFrame([{"abbrev": t["teamAbbrev"]["default"], "common": t["teamCommonName"]["default"]}
                               for t in data["standings"]])
    return _TEAMS


def write_csv(runs: list[GameRun], proj: pd.DataFrame, d: date, run_type: str, cons: pd.DataFrame | None = None,
              kl: pd.DataFrame | None = None):
    """Every projection, plus the best exchange buy price (American odds) where a line exists."""
    path = out_dir() / f"nhl_projections_{d.isoformat()}_{run_type}.csv"
    props = pd.concat([r.props for r in runs if r.props is not None], ignore_index=True) if runs else pd.DataFrame()
    game = proj[proj.market != "summary"][["game_id", "away", "home", "market", "selection", "line",
                                           "model_prob", "fair_odds"]].rename(columns={"model_prob": "p"})
    game = _attach(game, cons)
    if not props.empty:
        pr = props.rename(columns={"p_over": "p"}).assign(selection="over")
        pr["fair_odds"] = pr["p"].map(lambda p: prob_to_american(p) if 0 < p < 1 else None)
        if kl is not None and not kl.empty:
            pr = pr.merge(kl[["player_id", "market", "line", "kalshi_price", "kalshi_prob"]],
                          on=["player_id", "market", "line"], how="left")
        cols = ["game_id", "away", "home", "team", "player", "market", "selection", "line", "p", "fair_odds", "mean"]
        cols += [c for c in ("kalshi_price", "kalshi_prob") if c in pr]
        game = pd.concat([game, pr[cols]], ignore_index=True)
    game.to_csv(path, index=False)
    return path, props


def _attach(game: pd.DataFrame, cons: pd.DataFrame | None) -> pd.DataFrame:
    """market_price (best buy price, American), market_prob (implied by it), market_source."""
    if cons is None or cons.empty:
        return game
    x = cons[["home", "away", "market", "selection", "line", "best_price", "best_dec", "best_book"]].copy()
    x["_line"] = x.line.astype(float).fillna(0.0).round(2)
    x = x.assign(market_prob=1 / x.best_dec).rename(columns={"best_price": "market_price", "best_book": "market_source"})
    x = x.drop(columns=["line", "best_dec"])
    g = game.assign(_line=game.line.astype(float).fillna(0.0).round(2))
    g = g.merge(x, on=["home", "away", "market", "selection", "_line"], how="left")
    return g.drop(columns="_line")


def posted_path(d: date):
    return data_dir() / "daily_cache" / f"{d.isoformat()}_posted.json"


def posted_games(d: date) -> set[int] | None:
    """Game ids posted this morning (had an exchange line); None if unknown (post everything)."""
    p = posted_path(d)
    if not p.exists():
        return None
    return set(json.loads(p.read_text()).get("game_ids", []))


def kalshi_props(props: pd.DataFrame, d: date) -> pd.DataFrame:
    """Kalshi prop lines merged with the model's probability for the same player / market / line.
    kalshi_prob is the BUY price (ask); kalshi_price is that price as American odds; edge = model - ask."""
    if props.empty or not settings().get("lines", {}).get("kalshi", True):
        return pd.DataFrame()
    players = props[props.market != "saves"].drop_duplicates("player_id")[["player_id", "player", "team"]]
    try:
        kl = kalshi.prop_lines(d, players, settings().get("lines", {}).get("max_spread", 0.10))
    except Exception:
        log.exception("kalshi fetch failed")
        return pd.DataFrame()
    if kl.empty:
        return kl
    m = kl.merge(props[["player_id", "market", "line", "p_over", "mean", "game_id", "home", "away"]],
                 on=["player_id", "market", "line"], how="inner")
    m = m.rename(columns={"p_over": "model_prob", "prob": "kalshi_mid"})
    m["kalshi_prob"] = m.ask
    m["kalshi_price"] = m.ask.map(cents_to_american)
    m["edge"] = m.model_prob - m.kalshi_prob
    return m


def select_props(kl: pd.DataFrame, skip_players: set | None = None) -> pd.DataFrame:
    """Mark the props to post (column `posted`): model above the Kalshi buy price, price at least
    lines.min_prop_price (no cheap longshots), at most one line per player (the biggest edge), and the
    props.top_n biggest edges per market (what the message shows). Players in skip_players (already
    posted today) are skipped."""
    if kl is None or kl.empty:
        return kl
    cfg = settings().get("lines", {})
    ok = (kl.edge > 0) & (kl.kalshi_prob >= cfg.get("min_prop_price", 0.12))
    if skip_players:
        ok &= ~kl.player_id.isin(skip_players)
    cand = kl[ok].sort_values("edge", ascending=False)
    if cfg.get("one_per_player", True):
        cand = cand.drop_duplicates("player_id", keep="first")
    cand = cand.groupby("market", group_keys=False).head(settings()["props"].get("top_n", 15))
    return kl.assign(posted=kl.index.isin(cand.index))


SRC = {"polymarket": "PM", "kalshi": "KS"}


def _key(market: str, sel: str, line) -> tuple:
    ln = 0.0 if line is None or (isinstance(line, float) and np.isnan(line)) else round(float(line), 2)
    return market, sel, ln


def line_rows(r: GameRun, cons: pd.DataFrame) -> list[dict]:
    """Model vs the best exchange buy price for the ML, the main puck line and the main total
    (the line closest to 50/50). Edge = model - probability implied by that price."""
    th = settings()["disagreement"]["thresholds"]
    g = cons[(cons.home == r.home) & (cons.away == r.away)] if not cons.empty else cons
    out = []
    for market, (a, b) in DG.PAIRS.items():
        m = g[g.market == market] if not g.empty else g
        if m.empty:
            continue
        if market == "h2h":
            picks = [(a, None), (b, None)]
        else:
            ma = m[m.selection == a]
            if ma.empty:
                continue
            if market == "spreads":
                ma = ma[np.isclose(ma.line.abs(), 1.5)] if np.isclose(ma.line.abs(), 1.5).any() else ma
            ln = float(ma.loc[(ma.book_prob - 0.5).abs().idxmin(), "line"])
            picks = [(a, ln), (b, -ln if market == "spreads" else ln)]
        for sel, ln in picks:
            x = m[m.selection == sel]
            if ln is not None:
                x = x[np.isclose(x.line.astype(float), ln)]
            if x.empty:
                continue
            x = x.iloc[0]
            mp = M.market_prob(r.sim, market, sel, ln)
            pp = 1.0 / float(x.best_dec)
            name = {"home": r.home, "away": r.away, "over": "Over", "under": "Under"}[sel]
            label = (f"ML {name}" if market == "h2h" else f"{name} {ln:+g}" if market == "spreads"
                     else f"{name} {ln:g}")
            out.append({"label": label, "key": _key(market, sel, ln), "model": mp, "price": float(x.best_price),
                        "src": SRC.get(x.best_book, "?"),
                        "edge": mp - pp, "flag": mp - pp >= th.get(market, 0.04)})
    return out


# ----------------------------------------------------------------------------- runs

def morning(d: date, force: bool = False) -> None:
    if once.already_done(f"morning_{d.isoformat()}", force):
        return
    backfill_season(season_for(d))  # make sure last night's games are in
    runs, gm, h = project_slate(d, "morning")
    if not runs:
        log.info("no games on %s", d)
        return
    run_id = uuid.uuid4().hex[:10]
    proj = projection_rows(runs, d, run_id, "morning", gm.version)
    db.append("projections", proj, d.isoformat())
    db.write("slates", _slate_rows(runs, d, "morning", run_id), d.isoformat())
    dis, note, cons = fetch_and_compare(runs, d, "open", run_id, "morning", gm.version)
    ts = datetime.now(UTC).isoformat(timespec="seconds")

    ml_pairs = set(zip(cons[cons.market == "h2h"].home, cons[cons.market == "h2h"].away, strict=True)) \
        if not cons.empty else set()
    posted = [r for r in sorted(runs, key=lambda r: r.start_utc) if (r.home, r.away) in ml_pairs]
    skipped = [f"{r.away}@{r.home}" for r in runs if (r.home, r.away) not in ml_pairs]
    posted_path(d).parent.mkdir(parents=True, exist_ok=True)
    posted_path(d).write_text(json.dumps({"game_ids": [r.game_id for r in posted]}))
    main_keys = set()                  # the lines shown on the cards (ML, main puck line, main total)
    cards = []
    for r in posted:
        rows = line_rows(r, cons)
        main_keys |= {(r.home, r.away, x["key"]) for x in rows}
        cards.append(F.game_card({**r.__dict__, "book_lines": [F.lines_block(rows)]}))
    if posted:
        if skipped:
            note += f"\nNot posted (no Polymarket/Kalshi line yet): {', '.join(skipped)}"
        PUB.send_message(F.morning_message(d.strftime("%a %b %-d"), gm.version, cards, note))
    else:
        PUB.send_message(f"🏒 <b>NHL · {d.strftime('%a %b %-d')}</b>\nNo Polymarket or Kalshi lines yet, so no game "
                         f"projections posted.\n{note}")
    if not dis.empty:
        # all lines are stored for tracking; the message lists only the main lines shown on the cards
        mine = dis[[(x.home, x.away, _key(x.market, x.selection, x.line)) in main_keys for x in dis.itertuples()]]
        if not mine.empty:
            PUB.send_message(F.disagreements_message(mine))

    tables = [r.props for r in runs if r.props is not None]
    props = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    kl = select_props(kalshi_props(props, d)) if not props.empty else pd.DataFrame()
    csv_path, props = write_csv(runs, proj, d, "morning", cons, kl)
    if not props.empty:
        db.append("prop_projections", props.assign(run_id=run_id, game_date=d.isoformat(),
                                                  model_version=gm.version), d.isoformat())
    if not kl.empty:
        db.append("prop_lines", kl.assign(ts=ts, run_id=run_id, game_date=d.isoformat(), model_version=gm.version),
                  d.isoformat())
    top_n = settings()["props"].get("top_n", settings()["props"].get("telegram_top_n", 15))
    shown = kl[kl.posted] if not kl.empty else kl
    PUB.send_message(F.market_props_message(shown, top_n, settings().get("lines", {}).get("prop_flag_edge", 0.04)))
    PUB.send_document(csv_path, caption=f"All projections {d.isoformat()} (games + every player prop line, "
                                        f"with the Polymarket/Kalshi buy price where there is one)")
    once.mark_done(f"morning_{d.isoformat()}")


def pregame(d: date) -> None:
    prev = db.read("slates", partitions=[d.isoformat()])
    runs, gm, h = project_slate(d, "pregame")
    if not runs:
        log.info("no upcoming games on %s", d)
        return
    run_id = uuid.uuid4().hex[:10]
    dis, note, _ = fetch_and_compare(runs, d, "pregame", run_id, "pregame", gm.version)
    shown = posted_games(d)

    changes = []
    for r in runs:
        old = prev[prev.game_id == r.game_id] if not prev.empty else prev
        changed = []
        if len(old):
            o = old.iloc[-1]
            for side in ("home", "away"):
                if _id(o[f"{side}_goalie_id"]) != _id(getattr(r, f"{side}_goalie_id")):
                    changed.append(f"{getattr(r, side)} goalie now {getattr(r, f'{side}_goalie')}")
        new_flags = []
        if not dis.empty:
            mine = dis[(dis.home == r.home) & (dis.away == r.away) & dis.flagged]
            old_dis = db.read("disagreements", partitions=[d.isoformat()])
            if not old_dis.empty:
                was = old_dis[(old_dis.run_type == "morning") & old_dis.flagged]
                was_keys = set(zip(was.home, was.away, was.market, was.selection, was.line.fillna(0), strict=True))
                mine = mine[[(x.home, x.away, x.market, x.selection, 0 if x.line is None or x.line != x.line else x.line)
                             not in was_keys for x in mine.itertuples()]]
            new_flags = [F.book_line(x) for x in mine.itertuples()]
        if (changed or new_flags) and (shown is None or r.game_id in shown):
            header = "; ".join(changed) if changed else "new disagreement(s)"
            changes.append(f"⚠️ {header}\n" + F.game_card({**r.__dict__, "book_lines": new_flags}))
    db.append("slates", _slate_rows(runs, d, "pregame", run_id), d.isoformat())
    if any(_goalie_changed(prev, r) for r in runs):
        proj = projection_rows([r for r in runs if _goalie_changed(prev, r)], d, run_id, "pregame", gm.version)
        db.append("projections", proj, d.isoformat())
    if changes:
        PUB.send_message(F.changes_message(changes) + f"\n\n{note}")
    else:
        log.info("pregame: no changes (%s)", note)
    new_props(runs, d, run_id, gm.version)


def new_props(runs: list[GameRun], d: date, run_id: str, version: str) -> None:
    """Kalshi adds prop markets through the afternoon: post the ones that opened after the morning run
    (model above the buy price only), and store them for grading."""
    tables = [r.props for r in runs if r.props is not None]
    if not tables:
        return
    kl = kalshi_props(pd.concat(tables, ignore_index=True), d)
    if kl.empty:
        return
    old = db.read("prop_lines", partitions=[d.isoformat()])
    done = set()
    if not old.empty:
        seen = set(zip(old.player_id, old.market, old.line, strict=True))
        kl = kl[[(p, m, ln) not in seen for p, m, ln in zip(kl.player_id, kl.market, kl.line, strict=True)]]
        if "posted" in old:
            done = set(old[old.posted.fillna(False).astype(bool)].player_id)
    if kl.empty:
        return
    kl = select_props(kl, done)
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    db.append("prop_lines", kl.assign(ts=ts, run_id=run_id, game_date=d.isoformat(), model_version=version),
              d.isoformat())
    flag = settings().get("lines", {}).get("prop_flag_edge", 0.04)
    if kl.posted.any():
        top_n = settings()["props"].get("top_n", 15)
        PUB.send_message(F.market_props_message(kl[kl.posted], top_n, flag).replace("Player props", "New player props", 1))


def _slate_rows(runs: list[GameRun], d: date, run_type: str, run_id: str) -> pd.DataFrame:
    return pd.DataFrame([{"game_date": d.isoformat(), "game_id": r.game_id, "home": r.home, "away": r.away,
                          "home_goalie_id": r.home_goalie_id, "away_goalie_id": r.away_goalie_id,
                          "home_goalie": r.home_goalie, "away_goalie": r.away_goalie,
                          "home_goalie_status": r.home_goalie_status, "away_goalie_status": r.away_goalie_status,
                          "run_type": run_type, "run_id": run_id,
                          "ts": datetime.now(UTC).isoformat(timespec="seconds")} for r in runs])


def _id(x) -> int | None:
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)
    except (TypeError, ValueError):
        return None


def _goalie_changed(prev: pd.DataFrame, r: GameRun) -> bool:
    if prev.empty:
        return True
    o = prev[prev.game_id == r.game_id]
    if o.empty:
        return True
    o = o.iloc[-1]
    return _id(o.home_goalie_id) != _id(r.home_goalie_id) or _id(o.away_goalie_id) != _id(r.away_goalie_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["morning", "pregame"], required=True)
    ap.add_argument("--force", action="store_true", help="post again even if today's morning post already went out")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    d = today_local()
    log.info("slate date %s (%s)", d, tz())
    morning(d, a.force) if a.run == "morning" else pregame(d)


if __name__ == "__main__":
    main()
