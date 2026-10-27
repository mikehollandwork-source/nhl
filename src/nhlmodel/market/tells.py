"""Reverse-engineering experiment: if books price the way the PDF describes, what gives them away?

Hypotheses (each tested on lines we collect, with a Benjamini-Hochberg correction across tests):
  H1 margin shape     - which de-vig method best predicts outcomes per book, and do underdogs
                        carry more of the margin than favorites (favorite-longshot loading)?
  H2 implied rates    - book-implied total goals vs our model, by goalie status.
  H3 consistency      - does a book's puck line agree with the rates implied by its own ML + total?
                        If the rate-implied price predicts outcomes better, the gap is a tell.
  H4 goalie news      - do lines move more when the starter changes between morning and pregame?
  H5 team shading     - do some teams consistently price shorter/longer than our model says?
Nothing is reported as a finding below `tells.min_games` graded games.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from nhlmodel.config import settings
from nhlmodel.market import devig as D
from nhlmodel.market import implied as IM
from nhlmodel.tracking.grade import settle


def bh(pvals: list[float], alpha: float) -> list[bool]:
    """Benjamini-Hochberg: which p-values are discoveries at false discovery rate alpha."""
    p = np.asarray(pvals, float)
    if len(p) == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.zeros(len(p), bool)
    below = np.where(ranked <= alpha)[0]
    if len(below):
        passed[order[: below.max() + 1]] = True
    return passed.tolist()


def closing_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """Each book's last snapshot before puck drop."""
    o = odds[pd.to_datetime(odds.ts, utc=True) < pd.to_datetime(odds.commence_time, utc=True)]
    last = o.groupby(["event_id", "book"]).ts.transform("max")
    return o[o.ts == last]


def _results(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    return df.merge(results, on=["game_date", "home", "away"], how="inner")


def _two_way(g: pd.DataFrame, a: str, b: str):
    ga, gb = g[g.selection == a], g[g.selection == b]
    if ga.empty or gb.empty:
        return None
    return ga.iloc[0], gb.iloc[0]


# ----------------------------------------------------------------------------- H1

def h1_margin_shape(close: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    h = _results(close[close.market == "h2h"], results)
    rows = []
    for (eid, book), g in h.groupby(["event_id", "book"]):
        pair = _two_way(g, "home", "away")
        if pair is None:
            continue
        rh, ra = pair
        prices = [rh.price, ra.price]
        q = D.implied(prices)
        y = int(rh.home_score > rh.away_score)
        rec = {"book": book, "event_id": eid, "overround": q.sum()}
        for m in D.METHODS:
            rec[f"ll_{m}"] = D.log_loss(D.devig(prices, m)[0], y)
        fav = int(np.argmax(q))
        fair = D.multiplicative(q)
        rec["fav_margin"] = q[fav] - fair[fav]
        rec["dog_margin"] = q[1 - fav] - fair[1 - fav]
        rows.append(rec)
    df = pd.DataFrame(rows)
    tests: list[tuple] = []
    if df.empty:
        return df, tests
    summary = df.groupby("book").agg(n=("event_id", "size"), overround=("overround", "mean"),
                                     **{f"ll_{m}": (f"ll_{m}", "mean") for m in D.METHODS}).reset_index()
    summary["best_method"] = summary[[f"ll_{m}" for m in D.METHODS]].idxmin(axis=1).str[3:]
    for book, g in df.groupby("book"):
        if len(g) >= 30:
            t = stats.ttest_rel(g.dog_margin, g.fav_margin)
            tests.append((f"H1 {book}: underdog carries more margin than favorite",
                          float(t.pvalue), float((g.dog_margin - g.fav_margin).mean())))
    return summary, tests


# ----------------------------------------------------------------------------- H2 + H5

def book_rates(cons: pd.DataFrame) -> pd.DataFrame:
    """Consensus-implied scoring rates per event from de-vigged ML and the most-quoted total."""
    ml = cons[(cons.market == "h2h") & (cons.selection == "home")][
        ["event_id", "game_date", "home", "away", "book_prob"]].rename(columns={"book_prob": "p_home_book"})
    tot = cons[(cons.market == "totals") & (cons.selection == "over")]
    main = tot.sort_values("n_books").groupby("event_id").tail(1)[["event_id", "line", "book_prob"]]
    main = main.rename(columns={"line": "tot_line", "book_prob": "p_over_book"})
    df = ml.drop_duplicates("event_id").merge(main, on="event_id")
    if df.empty:
        return df
    sol = [IM.solve_rates(r.p_home_book, r.tot_line, r.p_over_book) for r in df.itertuples()]
    df["book_lh"] = [s[0] for s in sol]
    df["book_la"] = [s[1] for s in sol]
    df["fit_err"] = [s[2] for s in sol]
    return df


def h2_h5_vs_model(cons_open: pd.DataFrame, projections: pd.DataFrame,
                   slates: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    tests: list[tuple] = []
    df = book_rates(cons_open)
    if df.empty or projections.empty:
        return df, tests
    p = projections[projections.run_type == "morning"]
    summ = p[p.market == "summary"].pivot_table(index=["game_date", "home", "away"], columns="selection",
                                                values="model_prob").reset_index()
    mlp = p[(p.market == "h2h") & (p.selection == "home")][["game_date", "home", "away", "model_prob"]]
    df = df.merge(summ[["game_date", "home", "away", "exp_goals_home", "exp_goals_away"]],
                  on=["game_date", "home", "away"])
    df = df.merge(mlp.rename(columns={"model_prob": "p_home_model"}), on=["game_date", "home", "away"])
    if not slates.empty:
        s = slates[slates.run_type == "morning"].drop_duplicates(["home", "away"])
        df = df.merge(s[["home", "away", "home_goalie_status", "away_goalie_status"]], on=["home", "away"], how="left")
    # book expected settlement total under its own rates (ties add one goal)
    df["book_total"] = [_exp_total(r.book_lh, r.book_la) for r in df.itertuples()]
    df["total_gap"] = df.book_total - (df.exp_goals_home + df.exp_goals_away)
    if len(df) >= 30:
        t = stats.ttest_1samp(df.total_gap, 0.0)
        tests.append(("H2 books' implied total differs from model overall", float(t.pvalue), float(df.total_gap.mean())))
    if "home_goalie_status" in df:
        for st, g in df.groupby(df.home_goalie_status.fillna("unknown")):
            if len(g) >= 30:
                t = stats.ttest_1samp(g.total_gap, 0.0)
                tests.append((f"H2 total gap when home goalie {st}", float(t.pvalue), float(g.total_gap.mean())))

    # H5: team-side moneyline gap (book minus model), positive = book rates the team higher
    side = pd.concat([
        pd.DataFrame({"team": df.home, "gap": df.p_home_book - df.p_home_model}),
        pd.DataFrame({"team": df.away, "gap": (1 - df.p_home_book) - (1 - df.p_home_model)}),
    ])
    for team, g in side.groupby("team"):
        if len(g) >= 20:
            t = stats.ttest_1samp(g.gap, 0.0)
            tests.append((f"H5 {team}: book vs model moneyline gap", float(t.pvalue), float(g.gap.mean())))
    return df, tests


def _exp_total(lh: float, la: float) -> float:
    d = IM.probs_from_rates(lh, la)
    return float((d["_m"] * d["_total"]).sum())


# ----------------------------------------------------------------------------- H3

def h3_consistency(close: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """Per book: rates from its own ML + total -> its implied puck line. Compare to its posted puck line."""
    rows = []
    for (eid, book), g in close.groupby(["event_id", "book"]):
        ml = _two_way(g[g.market == "h2h"], "home", "away")
        tot = g[(g.market == "totals")]
        pl = g[(g.market == "spreads") & (g.selection == "home")]
        if ml is None or tot.empty or pl.empty:
            continue
        over = tot[tot.selection == "over"].iloc[0]
        under = tot[(tot.selection == "under") & np.isclose(tot.line, over.line)]
        pla = g[(g.market == "spreads") & (g.selection == "away")]
        if under.empty or pla.empty:
            continue
        p_home = D.devig([ml[0].price, ml[1].price])[0]
        p_over = D.devig([over.price, under.iloc[0].price])[0]
        lh, la, err = IM.solve_rates(p_home, float(over.line), p_over)
        plh = pl.iloc[0]
        pla_m = pla[np.isclose(pla.line, -plh.line)]
        if pla_m.empty:
            continue
        posted = D.devig([plh.price, pla_m.iloc[0].price])[0]
        implied = IM.p_home_spread(lh, la, float(plh.line))
        rows.append({"event_id": eid, "book": book, "game_date": plh.game_date, "home": plh.home, "away": plh.away,
                     "pl_line": float(plh.line), "posted": posted, "rate_implied": implied,
                     "gap": implied - posted, "fit_err": err})
    df = pd.DataFrame(rows)
    tests: list[tuple] = []
    if df.empty:
        return df, tests
    df = _results(df, results)
    df["y"] = [settle("spreads", "home", r.pl_line, r.home_score, r.away_score) for r in df.itertuples()]
    df = df[df.y.notna()]
    df["ll_posted"] = [D.log_loss(p, int(y)) for p, y in zip(df.posted, df.y, strict=True)]
    df["ll_implied"] = [D.log_loss(p, int(y)) for p, y in zip(df.rate_implied, df.y, strict=True)]
    for book, g in df.groupby("book"):
        if len(g) >= 30:
            t = stats.ttest_rel(g.ll_posted, g.ll_implied)
            tests.append((f"H3 {book}: own-rate puck line beats posted puck line",
                          float(t.pvalue) / 2 if t.statistic > 0 else 1 - float(t.pvalue) / 2,
                          float((g.ll_posted - g.ll_implied).mean())))
    return df, tests


# ----------------------------------------------------------------------------- H4

def h4_goalie_moves(cons: pd.DataFrame, slates: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    tests: list[tuple] = []
    ml = cons[(cons.market == "h2h") & (cons.selection == "home")].copy()
    if ml.empty or slates.empty:
        return pd.DataFrame(), tests
    ml = ml.sort_values("ts")
    first = ml.groupby("event_id").head(1).set_index("event_id")
    last = ml[pd.to_datetime(ml.ts, utc=True) < pd.to_datetime(ml.commence_time, utc=True)].groupby("event_id").tail(1).set_index("event_id")
    mv = (last.book_prob - first.book_prob).rename("move").to_frame().join(first[["game_date", "home", "away"]])
    s = slates.sort_values("ts")
    m = s[s.run_type == "morning"].drop_duplicates(["game_date", "home", "away"], keep="last")
    p = s[s.run_type == "pregame"].drop_duplicates(["game_date", "home", "away"], keep="last")
    j = m.merge(p, on=["game_date", "home", "away"], suffixes=("_m", "_p"))
    j["changed"] = (j.home_goalie_id_m != j.home_goalie_id_p) | (j.away_goalie_id_m != j.away_goalie_id_p)
    mv = mv.reset_index().merge(j[["game_date", "home", "away", "changed"]], on=["game_date", "home", "away"])
    a, b = mv[mv.changed].move.abs(), mv[~mv.changed].move.abs()
    if len(a) >= 10 and len(b) >= 30:
        t = stats.mannwhitneyu(a, b, alternative="greater")
        tests.append(("H4 lines move more after a starter change", float(t.pvalue), float(a.mean() - b.mean())))
    return mv, tests


# ----------------------------------------------------------------------------- report

def report(odds: pd.DataFrame, cons: pd.DataFrame, projections: pd.DataFrame, slates: pd.DataFrame,
           results: pd.DataFrame) -> str:
    cfg = settings()["tells"]
    graded_games = _results(cons[["game_date", "home", "away"]].drop_duplicates(), results)
    n = len(graded_games)
    head = f"🔍 <b>Line reverse-engineering</b>\nGames with lines + results: {n}/{cfg['min_games']}"
    if n < cfg["min_games"]:
        return head + "\nStill collecting. Findings are withheld until the sample is large enough."
    close = closing_odds(odds)
    cons_open = cons[cons.snapshot_type == "open"] if "snapshot_type" in cons else cons
    s1, t1 = h1_margin_shape(close, results)
    _, t2 = h2_h5_vs_model(cons_open, projections, slates)
    _, t3 = h3_consistency(close, results)
    _, t4 = h4_goalie_moves(cons, slates)
    tests = t1 + t2 + t3 + t4
    found = bh([t[1] for t in tests], cfg["fdr_alpha"])
    lines = [head]
    if not s1.empty:
        lines.append("<b>Best de-vig method by book</b> (lowest log-loss)<pre>" +
                     s1[["book", "n", "overround", "best_method"]].to_string(index=False, float_format=lambda x: f"{x:.3f}") +
                     "</pre>")
    hits = [(t, f) for t, f in zip(tests, found, strict=True) if f]
    lines.append(f"Tests run: {len(tests)} · significant after FDR {cfg['fdr_alpha']:.0%}: {len(hits)}")
    for (name, p, eff), _ in hits:
        lines.append(f"• {name}: effect {eff:+.4f}, p={p:.4f}")
    if not hits:
        lines.append("No tell survives the multiple-testing correction yet.")
    return "\n".join(lines)
