"""Nightly system report: what the system is good at, where it is weak, and what is trending.

Everything is computed from stored rows (morning projections, graded disagreements vs the market,
posted Kalshi props, final results). A finding is only called a strength or a weakness when the
sample is big enough AND a paired bootstrap 90% interval excludes zero; otherwise it is "leaning"
(direction only) or "too early". This keeps early-season reports from over-claiming.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhlmodel.tracking import grade as GR

MIN_GAMES = 100          # games before a game-level finding can be a strength / weakness
MIN_PROPS = 300          # graded prop lines before a prop finding can be a strength / weakness
MIN_TEAM_GAMES = 5
MIN_BUCKET = 25
STAR_PPG = 0.85
RECENT_DAYS = 7


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 7) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.quantile(m, 0.05)), float(np.quantile(m, 0.95))


def verdict(mean: float, lo: float, hi: float, n: int, min_n: int, good_if_negative: bool = True) -> str:
    """strength / weakness / leaning / too early, for a quantity where lower is better (log-loss gap)
    unless good_if_negative is False."""
    if n < 10 or np.isnan(mean):
        return "too early"
    good = mean < 0 if good_if_negative else mean > 0
    solid = n >= min_n and not np.isnan(lo) and (hi < 0 or lo > 0)
    if solid:
        return "strength" if good else "weakness"
    return "leaning"


def _f(label: str, kind: str, text: str, n: int, value: float | None = None) -> dict:
    return {"label": label, "verdict": kind, "text": text, "n": int(n), "value": value}


# ----------------------------------------------------------------------------- inputs

def morning_projections(proj: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """One row per final game: model home-win prob, projected goals, actual result."""
    if proj.empty or res.empty:
        return pd.DataFrame()
    p = proj[(proj.run_type == "morning") & (proj.market == "summary")]
    p = p.sort_values("ts").drop_duplicates(["game_date", "home", "away", "selection"], keep="last")
    w = p.pivot_table(index=["game_date", "home", "away"], columns="selection", values="model_prob").reset_index()
    need = {"p_home", "exp_goals_home", "exp_goals_away"}
    if not need.issubset(w.columns):
        return pd.DataFrame()
    m = w.merge(res, on=["game_date", "home", "away"])
    m["home_won"] = (m.home_score > m.away_score).astype(int)
    m["exp_total"] = m.exp_goals_home + m.exp_goals_away
    m["total"] = m.home_score + m.away_score
    return m


# ----------------------------------------------------------------------------- findings

def game_price_findings(graded: pd.DataFrame, day: date) -> list[dict]:
    out = []
    if graded is None or graded.empty:
        return out
    g = graded[graded.outcome.notna()]
    names = {"h2h": "Moneyline", "totals": "Totals", "spreads": "Puck line"}
    for market, label in names.items():
        x = g[g.market == market]
        if market == "h2h":
            x = x[x.selection == "home"]                      # one row per game
        elif market == "totals":
            x = x[x.selection == "over"]
        else:
            x = x[x.selection == "home"]
        if x.empty:
            continue
        per = (x.model_logloss - x.book_logloss).groupby([x.game_date, x.home, x.away]).mean()
        n = len(per)
        lo, hi = boot_ci(per.to_numpy())
        v = verdict(per.mean(), lo, hi, n, MIN_GAMES)
        better = "better" if per.mean() < 0 else "worse"
        out.append(_f(f"{label} prices", v, f"{label}: model {better} than the market's no-vig price "
                      f"(log-loss gap {per.mean():+.4f} per game, {n} games)", n, float(per.mean())))
    return out


def scoring_bias(m: pd.DataFrame) -> list[dict]:
    if m.empty:
        return []
    err = (m.exp_total - m.total).to_numpy()
    lo, hi = boot_ci(err)
    n = len(err)
    mean = float(err.mean())
    kind = "too early" if n < 10 else ("weakness" if n >= 30 and abs(mean) >= 0.25 and (hi < 0 or lo > 0)
                                       else ("leaning" if abs(mean) >= 0.15 else "strength" if n >= 30 else "leaning"))
    if kind == "strength":
        text = f"Total goals: no scoring bias (projected minus actual {mean:+.2f} per game, {n} games)"
    else:
        side = "low" if mean < 0 else "high"
        text = f"Total goals: projections run {side} by {abs(mean):.2f} goals per game ({n} games)"
    return [_f("Scoring bias", kind, text, n, mean)]


def favourite_calibration(m: pd.DataFrame) -> list[dict]:
    """Win rate by model probability bucket (from the side the model favours)."""
    if m.empty:
        return []
    p_fav = np.maximum(m.p_home, 1 - m.p_home)
    fav_won = np.where(m.p_home >= 0.5, m.home_won, 1 - m.home_won)
    b = pd.cut(p_fav, [0.5, 0.55, 0.6, 0.65, 1.0], include_lowest=True,
               labels=["50-55%", "55-60%", "60-65%", "65%+"])
    d = pd.DataFrame({"b": b, "p": p_fav, "y": fav_won})
    out = []
    worst = None
    for k, g in d.groupby("b", observed=True):
        if len(g) < MIN_BUCKET:
            continue
        gap = float(g.y.mean() - g.p.mean())
        if worst is None or abs(gap) > abs(worst[1]):
            worst = (k, gap, len(g), float(g.p.mean()), float(g.y.mean()))
    if worst:
        k, gap, n, pr, ac = worst
        lo, hi = boot_ci((d[d.b == k].y - d[d.b == k].p).to_numpy())
        solid = n >= MIN_BUCKET * 2 and (hi < 0 or lo > 0)
        kind = ("weakness" if solid and abs(gap) >= 0.05 else "leaning" if abs(gap) >= 0.04
                else "strength" if n >= MIN_BUCKET * 2 else "leaning")
        if kind == "strength":
            text = f"Favourites are well calibrated (largest bucket gap {100 * gap:+.0f} pts)"
        else:
            how = "too timid (they win more than we say)" if gap > 0 else "too bold (they win less than we say)"
            text = f"Favourites at {k}: we say {100 * pr:.0f}%, they win {100 * ac:.0f}% — {how} ({n} games)"
        out.append(_f("Favourite calibration", kind, text, n, gap))
    return out


def flagged_findings(graded: pd.DataFrame) -> list[dict]:
    if graded is None or graded.empty:
        return []
    g = graded[graded.outcome.notna() & graded.flagged]
    if g.empty:
        return []
    out = []
    names = {"h2h": "moneyline", "spreads": "puck line", "totals": "totals"}
    for market, x in g.groupby("market"):
        n = len(x)
        lo, hi = boot_ci(x.units.to_numpy())
        roi = float(x.units.mean())
        kind = verdict(roi, lo, hi, n, MIN_GAMES, good_if_negative=False)
        out.append(_f(f"Flagged {names.get(market, market)}", kind,
                      f"Flagged {names.get(market, market)} picks: {int(x.outcome.sum())}-{int(n - x.outcome.sum())}, "
                      f"{x.units.sum():+.1f}u ({100 * roi:+.0f}% ROI, {n} picks)", n, roi))
    return out


def prop_findings(gp: pd.DataFrame) -> list[dict]:
    """gp: GR.grade_props output (+ star flag)."""
    if gp is None or gp.empty:
        return []
    out = []
    for market, x in gp.groupby("market"):
        per = (x.model_ll - x.kalshi_ll).groupby(x.game_id).mean()
        n = len(x)
        lo, hi = boot_ci(per.to_numpy())
        kind = verdict(per.mean(), lo, hi, n, MIN_PROPS)
        better = "better" if per.mean() < 0 else "worse"
        out.append(_f(f"{market} props", kind, f"{market.capitalize()} props: model {better} than Kalshi "
                      f"(log-loss gap {per.mean():+.4f}, {n} lines)", n, float(per.mean())))
    f = gp[gp.flagged]
    if len(f):
        exp_m, exp_k = float(f.model_prob.sum()), float(f.kalshi_prob.sum())
        lo, hi = boot_ci(f.units.to_numpy())
        kind = verdict(float(f.units.mean()), lo, hi, len(f), MIN_PROPS // 3, good_if_negative=False)
        out.append(_f("Posted props", kind, f"Posted props: {int(f.won.sum())} of {len(f)} won "
                      f"(model expected {exp_m:.1f}, Kalshi {exp_k:.1f}), {f.units.sum():+.1f}u", len(f),
                      float(f.units.mean())))
    if "star" in gp:
        pts = gp[(gp.market == "points") & (gp.line == 0.5)]
        for flag, name in ((True, "Stars"), (False, "Depth players")):
            t = pts[pts.star == flag]
            if len(t) < 20:
                continue
            gap = float(t.model_prob.mean() - t.outcome.mean())
            lo, hi = boot_ci((t.model_prob - t.outcome).to_numpy())
            solid = len(t) >= 100 and (hi < 0 or lo > 0)
            kind = ("weakness" if solid and abs(gap) >= 0.03 else "leaning" if abs(gap) >= 0.03
                    else "strength" if len(t) >= 100 else "leaning")
            if kind == "strength":
                text = f"{name}' point props are well calibrated ({len(t)} lines)"
            else:
                side = "under" if gap < 0 else "over"
                text = (f"{name}' point props: we {side}-predict by {abs(100 * gap):.0f} pts "
                        f"({100 * t.model_prob.mean():.0f}% vs {100 * t.outcome.mean():.0f}% actual, {len(t)} lines)")
            out.append(_f(f"{name} props", kind, text, len(t), gap))
    return out


def team_findings(m: pd.DataFrame) -> list[dict]:
    if m.empty:
        return []
    rows = pd.concat([
        pd.DataFrame({"team": m.home, "err": m.exp_goals_home - m.home_score}),
        pd.DataFrame({"team": m.away, "err": m.exp_goals_away - m.away_score})])
    t = rows.groupby("team").err.agg(["mean", "size"])
    t = t[t["size"] >= MIN_TEAM_GAMES]
    if t.empty:
        return []
    out = []
    for team, r in t.reindex(t["mean"].abs().sort_values(ascending=False).index).head(3).iterrows():
        if abs(r["mean"]) < 0.4:
            continue
        side = "under" if r["mean"] < 0 else "over"
        out.append(_f(f"Team {team}", "trend", f"{team}: we {side}-project their scoring by {abs(r['mean']):.1f} "
                      f"goals per game ({int(r['size'])} games)", int(r["size"]), float(r["mean"])))
    return out


def trend_findings(graded: pd.DataFrame, gp: pd.DataFrame, day: date) -> list[dict]:
    out = []
    cut = (day - timedelta(days=RECENT_DAYS - 1)).isoformat()
    if graded is not None and not graded.empty:
        g = graded[graded.outcome.notna() & (graded.market == "h2h") & (graded.selection == "home")]
        gap = (g.model_logloss - g.book_logloss)
        recent, older = gap[g.game_date >= cut], gap[g.game_date < cut]
        if len(recent) >= 20 and len(older) >= 20:
            d = float(recent.mean() - older.mean())
            lo, hi = boot_ci(recent.to_numpy() - older.mean())
            word = "steady" if not (hi < 0 or lo > 0) else ("improving" if d < 0 else "slipping")
            out.append(_f("ML trend", "trend", f"Moneyline vs market, last {RECENT_DAYS} days: {word} "
                          f"(gap {recent.mean():+.4f} vs {older.mean():+.4f} before, {len(recent)} games)", len(recent), d))
    if gp is not None and not gp.empty:
        f = gp[gp.flagged]
        dates = f.game_date.astype(str)
        recent, older = f[dates >= cut], f[dates < cut]
        if len(recent) >= 15:
            word = "hot" if recent.units.sum() > 0 else "cold"
            extra = f" vs {older.units.mean() * 100:+.0f}% ROI before" if len(older) >= 15 else ""
            out.append(_f("Props trend", "trend", f"Posted props, last {RECENT_DAYS} days: {word}, "
                          f"{recent.units.sum():+.1f}u ({100 * recent.units.mean():+.0f}% ROI{extra})", len(recent),
                          float(recent.units.mean())))
    return out


def star_flags(gp: pd.DataFrame, player_games: pd.DataFrame) -> pd.DataFrame:
    """Mark players with STAR_PPG+ points per game over the 82 games before each prop's game."""
    if gp.empty or player_games.empty:
        return gp.assign(star=False)
    pg = player_games[["player_id", "game_date", "points"]].copy()
    pg["game_date"] = pd.to_datetime(pg.game_date)
    stars = {}
    for (pid, gd), _ in gp.groupby(["player_id", "game_date"]):
        h = pg[(pg.player_id == pid) & (pg.game_date < pd.Timestamp(gd))].sort_values("game_date").tail(82)
        stars[(pid, gd)] = bool(len(h) >= 10 and h.points.mean() >= STAR_PPG)
    return gp.assign(star=[stars[(p, d)] for p, d in zip(gp.player_id, gp.game_date, strict=True)])


def build(day: date, proj: pd.DataFrame, res: pd.DataFrame, graded: pd.DataFrame,
          prop_lines: pd.DataFrame, player_games: pd.DataFrame, flag_edge: float = 0.04) -> dict:
    m = morning_projections(proj, res)
    gp = GR.grade_props(prop_lines, player_games, flag_edge) if prop_lines is not None and not prop_lines.empty \
        else pd.DataFrame()
    if not gp.empty:
        gp = star_flags(gp, player_games)
    findings = (game_price_findings(graded, day) + scoring_bias(m) + favourite_calibration(m)
                + flagged_findings(graded) + prop_findings(gp))
    trends = trend_findings(graded, gp, day) + team_findings(m)
    n_games = int(len(m))
    n_props = int(len(gp))
    return {"date": day.isoformat(), "games": n_games, "prop_lines": n_props,
            "strengths": [f for f in findings if f["verdict"] == "strength"],
            "weaknesses": [f for f in findings if f["verdict"] == "weakness"],
            "leaning": [f for f in findings if f["verdict"] == "leaning"],
            "trends": trends}
