"""Backtest betting rules built from what the model gets right and wrong.

python -m nhlmodel.pipelines.strategy_backtest [--sims 2000] [--refresh-prices] [--refresh-preds]

Data: pre-game exchange prices (ingest/history_prices.py: Polymarket 2025-26 and 2026-27, Kalshi
2026-27) and the model's out-of-sample predictions for the same games (2025-26 from a model fitted on
earlier seasons; 2026-27 from a model fitted through 2025-26). Predictions use the real starting goalie
and lineup, which were usually but not always known before the game: a mild optimism.

One bet candidate per game per market (moneyline; the totals line closest to 50/50; the +/-1.5 puck
line), 1 unit at the buy price (ask). Rules, fixed before looking at results:
  follow     bet the side where the model is >= X above the price (what the daily post flags)
  fade       bet against the model when it disagrees by >= X
  corrected  recalibrate the model on EARLIER games only (fixes level and stretch), then follow
  blend      learn on EARLIER games how much weight the model deserves next to the market (a negative
             weight means fade), then bet when the blend beats the price by >= X
Thresholds X = 3, 5, 8 points. Choose on the selection season, test once on the test season.
A rule passes if the test ROI's 90% interval (bootstrap by game day) is above 0 with >= MIN_BETS bets.
Writes models/strategy_backtest.json.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nhlmodel.config import data_dir, season_for, settings, today_local
from nhlmodel.features import assemble as A
from nhlmodel.features import history as H
from nhlmodel.ingest import history_prices as HP
from nhlmodel.market.tells import bh
from nhlmodel.pipelines.ablation import sim_preds
from nhlmodel.pipelines.train import train_xg
from nhlmodel.sim import model as M2
from nhlmodel.store import db
from nhlmodel.tracking import grade as GR

log = logging.getLogger("strategy_backtest")
THRESHOLDS = (0.03, 0.05, 0.08)
RULES = ("follow", "fade", "corrected", "blend")
MIN_BETS = 150
MIN_HISTORY = 150          # earlier candidates needed before a walk-forward fit is used
TOTAL_COLS = {4.5: "p_o45", 5.5: "p_o55", 6.5: "p_o65", 7.5: "p_o75", 8.5: "p_o85"}


# ----------------------------------------------------------------------------- model predictions

def predictions(n_sims: int) -> pd.DataFrame:
    """Out-of-sample model probabilities for the selection and test seasons."""
    hc = settings()["history"]
    train_s, sel_s = hc["train_seasons"], hc["test_season"]
    cur = season_for(today_local())
    step = settings()["sim"]["step_seconds"]
    xg, _ = train_xg(train_s[:-1], train_s[-1], "ablation")
    seasons = [*train_s, sel_s, cur]
    h = H.load(xg, seasons=seasons)
    ex = A.load_extras(seasons)
    rapm, _ = A.rapm_by_season(h, ex, [*train_s[1:], sel_s, cur])
    rows = A.assemble(h, ex, rapm)
    rows = rows[rows.valid.fillna(False) & (rows.lgs_ev_sogf > 0) & (rows.s_ev_sogf > 0) & (rows.o_ev_soga > 0)]
    rep = data_dir() / "models" / "ablation_report.json"
    groups = json.loads(rep.read_text())["final"]["groups"] if rep.exists() else ["lineup_repl"]
    out = []
    for fit, ev_s in ((train_s, sel_s), ([*train_s, sel_s], cur)):
        fr, fg = rows[rows.season.isin(fit)], h.games[h.games.season.isin(fit)]
        gm = M2.fit(fr, groups, h.states[h.states.season.isin(fit)], fg,
                    ex.penalties[ex.penalties.season.isin(fit)], "+".join(groups))
        M2.calibrate(gm, fr, fg)
        ev = rows[(rows.season == ev_s) & (rows.game_type == 2)]
        inputs = {gid: gm.game_input(g[g.is_home == 1].iloc[0], g[g.is_home == 0].iloc[0], 2)
                  for gid, g in ev.groupby("game_id") if len(g) == 2}
        p = sim_preds(inputs, gm.constants, n_sims, step).assign(season=ev_s)
        log.info("season %s: %d games predicted (fit on %s)", ev_s, len(p), fit)
        out.append(p)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- bet candidates

def candidates(preds: pd.DataFrame, hist: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per game per market: side A (home / over / home's puck line), its best ask, the best ask
    of the other side B, the model's probability of A, and whether A won."""
    g = games.drop_duplicates("game_id", keep="last").set_index("game_id")
    pr = preds.set_index("game_id")
    h = hist.copy()
    h["lk"] = h.line.astype(float).fillna(0.0).round(1)
    best = h.groupby(["game_id", "market", "selection", "lk"], as_index=False).ask.min()
    rows = []
    for gid, x in best.groupby("game_id"):
        if gid not in pr.index or gid not in g.index:
            continue
        p, gm = pr.loc[gid], g.loc[gid]
        hs, as_ = int(gm.home_score), int(gm.away_score)
        base = {"game_id": gid, "season": int(p.season), "game_date": str(gm.game_date)[:10]}

        def ask(market, sel, lk, x=x):
            r = x[(x.market == market) & (x.selection == sel) & np.isclose(x.lk, lk)]
            return float(r.ask.iloc[0]) if len(r) else None

        a, b = ask("h2h", "home", 0.0), ask("h2h", "away", 0.0)
        if a and b:
            rows.append({**base, "market": "h2h", "line": 0.0, "ask_a": a, "ask_b": b, "m_a": float(p.p_home),
                         "y_a": int(hs > as_)})
        best_t = None
        for ln, col in TOTAL_COLS.items():
            a, b = ask("totals", "over", ln), ask("totals", "under", ln)
            if a and b and col in p and (best_t is None or abs(a / (a + b) - 0.5) < best_t[0]):
                best_t = (abs(a / (a + b) - 0.5), ln, a, b, float(p[col]))
        if best_t:
            _, ln, a, b, m = best_t
            rows.append({**base, "market": "totals", "line": ln, "ask_a": a, "ask_b": b, "m_a": m,
                         "y_a": int(hs + as_ > ln)})
        fav_home = float(p.p_home) >= 0.5          # the puck line where the model's favourite gives 1.5
        hl = -1.5 if fav_home else 1.5
        a, b = ask("spreads", "home", hl), ask("spreads", "away", -hl)
        if a and b:
            m = float(p.p_home_m15) if fav_home else 1 - float(p.p_away_m15)
            rows.append({**base, "market": "spreads", "line": hl, "ask_a": a, "ask_b": b, "m_a": m,
                         "y_a": int(hs - as_ + hl > 0)})
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c["mkt_a"] = c.ask_a / (c.ask_a + c.ask_b)          # market's no-vig probability of A
    return c.sort_values(["game_date", "game_id"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- walk-forward fits

def _logit(p):
    p = np.clip(np.asarray(p, float), 0.01, 0.99)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _fit(X: np.ndarray, y: np.ndarray, offset: np.ndarray) -> np.ndarray:
    def nll(b):
        p = np.clip(_sig(offset + X @ b), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum() + 0.5 * (b ** 2).sum()   # light ridge
    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x


def walk_forward(c: pd.DataFrame) -> pd.DataFrame:
    """Add p_corr (model recalibrated on earlier games) and p_blend (market + learned model weight),
    each fitted only on candidates from earlier dates in the same market. NaN until MIN_HISTORY rows."""
    c = c.copy()
    c["p_corr"], c["p_blend"], c["w_blend"] = np.nan, np.nan, np.nan
    for _market, x in c.groupby("market"):
        lm, lk, y = _logit(x.m_a), _logit(x.mkt_a), x.y_a.to_numpy(float)
        dates = x.game_date.to_numpy()
        for d in np.unique(dates):
            past, now = dates < d, dates == d
            if past.sum() < MIN_HISTORY:
                continue
            Xc = np.c_[np.ones(past.sum()), lm[past]]
            bc = _fit(Xc, y[past], np.zeros(past.sum()))
            c.loc[x.index[now], "p_corr"] = _sig(bc[0] + bc[1] * lm[now])
            Xb = np.c_[np.ones(past.sum()), lm[past] - lk[past]]
            bb = _fit(Xb, y[past], lk[past])
            c.loc[x.index[now], "p_blend"] = _sig(lk[now] + bb[0] + bb[1] * (lm[now] - lk[now]))
            c.loc[x.index[now], "w_blend"] = bb[1]
    return c


# ----------------------------------------------------------------------------- rules and scoring

def place(c: pd.DataFrame, rule: str, x: float) -> pd.DataFrame:
    """Rows that get a bet under `rule` at threshold x, with side ('a'/'b'), price and units."""
    p = {"follow": c.m_a, "fade": c.m_a, "corrected": c.p_corr, "blend": c.p_blend}[rule]
    like_a, like_b = (p - c.ask_a) >= x, ((1 - p) - c.ask_b) >= x
    if rule == "fade":
        like_a, like_b = like_b, like_a
    side = np.where(like_a, "a", np.where(like_b, "b", ""))
    b = c[side != ""].assign(side=side[side != ""])
    price = np.where(b.side == "a", b.ask_a, b.ask_b)
    won = np.where(b.side == "a", b.y_a == 1, b.y_a == 0)
    return b.assign(price=price, won=won, units=np.where(won, (1 - price) / price, -1.0))


def score(b: pd.DataFrame, n_boot: int = 2000, seed: int = 4) -> dict:
    if b.empty:
        return {"n": 0}
    day = b.groupby("game_date").units.agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(day), (n_boot, len(day)))
    roi = day["sum"].to_numpy()[idx].sum(1) / np.maximum(day["size"].to_numpy()[idx].sum(1), 1)
    return {"n": int(len(b)), "win_rate": float(b.won.mean()), "avg_price": float(b.price.mean()),
            "units": float(b.units.sum()), "roi": float(b.units.mean()), "lo90": float(np.quantile(roi, 0.05)),
            "hi90": float(np.quantile(roi, 0.95)), "p_not_positive": float((roi <= 0).mean())}


def variants(c: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every rule x threshold x market (totals also split into over-only and under-only bets)."""
    out = {}
    for rule in RULES:
        for x in THRESHOLDS:
            b = place(c, rule, x)
            for market in ("h2h", "totals", "spreads"):
                bm = b[b.market == market]
                out[f"{rule} {int(100 * x)} {market}"] = bm
                if market == "totals":
                    out[f"{rule} {int(100 * x)} totals-over"] = bm[bm.side == "a"]
                    out[f"{rule} {int(100 * x)} totals-under"] = bm[bm.side == "b"]
    return out


def evaluate(c: pd.DataFrame, sel_season: int, test_season: int) -> dict:
    c = walk_forward(c)
    sel = {k: score(v[v.season == sel_season]) for k, v in variants(c).items()}
    test = {k: score(v[v.season == test_season]) for k, v in variants(c).items()}
    names = [k for k, v in sel.items() if v["n"] >= MIN_BETS]
    keep = bh([sel[k]["p_not_positive"] for k in names], 0.10) if names else []
    chosen = [k for k, ok in zip(names, keep, strict=True) if ok and sel[k]["lo90"] > 0]
    result = {}
    for k in chosen:
        t = test[k]
        result[k] = ("pass" if t["n"] >= MIN_BETS and t.get("lo90", -1) > 0 else
                     "promising (too few test bets)" if t["n"] and t.get("lo90", -1) > 0 else "fail")
    w = c.dropna(subset=["w_blend"]).groupby("market").w_blend.agg(["first", "last"]).round(3).to_dict("index")
    return {"selection_season": sel_season, "test_season": test_season,
            "candidates": {int(s): {m: int(n) for m, n in g.market.value_counts().items()} for s, g in c.groupby("season")},
            "selection": sel, "test": test, "chosen_on_selection": chosen, "test_verdict": result,
            "blend_weight_on_model": w}


def props_record() -> dict:
    """Posted Kalshi props since tracking began (the only prop price history), first half vs second half."""
    lines = db.read("prop_lines")
    if lines.empty:
        return {}
    pg = db.read("player_games", partitions=[str(season_for(today_local()))])
    g = GR.grade_props(lines, pg)
    if g.empty:
        return {}
    g = g[g.flagged]
    days = sorted(g.game_date.astype(str).unique())
    half = days[len(days) // 2] if len(days) > 1 else days[0]
    out = {}
    for name, x in (("all", g), ("first_half", g[g.game_date.astype(str) < half]),
                    ("second_half", g[g.game_date.astype(str) >= half])):
        b = x.assign(price=x.kalshi_prob)
        out[name] = score(b)
        for mk, xm in b.groupby("market"):
            out[f"{name} {mk}"] = score(xm)
    return out


def run(n_sims: int, refresh_prices: bool = False, refresh_preds: bool = False) -> dict:
    hc = settings()["history"]
    sel_s, cur = hc["test_season"], season_for(today_local())
    if refresh_prices or not db.exists("market_history"):
        log.info("price history: %s", HP.backfill([sel_s, cur], cur))
    path = data_dir() / "models" / "strategy_preds.parquet"
    games = db.read("games", partitions=[str(sel_s), str(cur)])
    games = games[games.game_type == 2].drop_duplicates("game_id", keep="last")
    preds = pd.read_parquet(path) if path.exists() else None
    if preds is None or refresh_preds:
        preds = predictions(n_sims)
        preds.to_parquet(path, index=False)
    c = candidates(preds, db.read("market_history"), games)
    rep = evaluate(c, sel_s, cur)
    rep["props_posted"] = props_record()
    rep["min_bets"] = MIN_BETS
    (data_dir() / "models" / "strategy_backtest.json").write_text(json.dumps(rep, indent=1, default=float))
    c.to_parquet(data_dir() / "models" / "strategy_candidates.parquet", index=False)
    log.info("chosen on %s: %s; verdicts on %s: %s", sel_s, rep["chosen_on_selection"], cur, rep["test_verdict"])
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--refresh-prices", action="store_true", help="download price history again")
    ap.add_argument("--refresh-preds", action="store_true", help="re-run the model predictions")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(a.sims, a.refresh_prices, a.refresh_preds)


if __name__ == "__main__":
    main()
