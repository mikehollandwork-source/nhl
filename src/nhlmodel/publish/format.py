"""Telegram message formatting (HTML parse mode)."""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd

from nhlmodel.config import tz
from nhlmodel.market.devig import prob_to_american

GOALIE_ICON = {"confirmed": "✅", "likely": "🟡", "unconfirmed": "❔", "guessed": "🤖"}


def _odds(p: float) -> str:
    a = prob_to_american(p)
    return f"+{a:.0f}" if a > 0 else f"{a:.0f}"


def _pct(p: float) -> str:
    return f"{100 * p:.0f}%"


def _local_time(utc_iso: str | None) -> str:
    if not utc_iso:
        return ""
    t = datetime.fromisoformat(utc_iso.replace("Z", "+00:00")).astimezone(tz())
    return t.strftime("%-I:%M %p ET")


def game_card(g: dict) -> str:
    h, a = g["home"], g["away"]
    s = g["summary"]
    lines = [
        f"<b>{a} @ {h}</b>  {_local_time(g.get('start_utc'))}",
        f"G: {escape(g['away_goalie'])} {GOALIE_ICON.get(g['away_goalie_status'], '')} vs "
        f"{escape(g['home_goalie'])} {GOALIE_ICON.get(g['home_goalie_status'], '')}",
        f"Proj: {a} {s['exp_goals_away']:.2f} – {h} {s['exp_goals_home']:.2f}  | SOG {s['exp_sog_away']:.0f}–{s['exp_sog_home']:.0f}",
        f"ML: {a} {_pct(s['p_away'])} ({_odds(s['p_away'])}) · {h} {_pct(s['p_home'])} ({_odds(s['p_home'])})",
        f"Reg 3-way: {a} {_pct(s['p_reg_away'])} / tie {_pct(s['p_reg_draw'])} / {h} {_pct(s['p_reg_home'])}",
        f"PL -1.5: {a} {_pct(s['p_away_m15'])} · {h} {_pct(s['p_home_m15'])}",
        f"Total: o5.5 {_pct(s['p_over_5_5'])} · o6.5 {_pct(s['p_over_6_5'])}",
    ]
    for d in g.get("book_lines", []):
        lines.append(d)
    return "\n".join(lines)


def morning_message(date_str: str, version: str, cards: list[str], odds_note: str) -> str:
    head = f"🏒 <b>NHL projections · {date_str}</b>\nmodel {version} · {len(cards)} games\n{odds_note}"
    legend = "Goalies: ✅ confirmed 🟡 likely ❔ unconfirmed 🤖 our guess"
    return "\n\n".join([head, legend, *cards])


def book_line(r) -> str:
    sel = {"home": r.home, "away": r.away, "over": "Over", "under": "Under"}[r.selection]
    ln = "" if r.market == "h2h" else f" {r.line:+g}" if r.market == "spreads" else f" {r.line:g}"
    flag = " 🚩" if r.flagged else ""
    return (f"{_mk(r.market)} {sel}{ln}: model {_pct(r.model_prob)} vs book {_pct(r.book_prob)} "
            f"(edge {100 * r.edge:+.1f}) best {r.best_price:+.0f} @{r.best_book}{flag}")


def _mk(m: str) -> str:
    return {"h2h": "ML", "spreads": "PL", "totals": "Tot"}.get(m, m)


def disagreements_message(d: pd.DataFrame) -> str:
    f = d[d.flagged].sort_values("edge", ascending=False)
    if f.empty:
        return "🚩 <b>Disagreements</b>\nNone above threshold today."
    rows = ["🚩 <b>Disagreements</b> (model higher than book by ≥ threshold)", "<pre>"]
    rows.append(f"{'game':<10}{'bet':<14}{'model':>6}{'book':>6}{'edge':>6}{'best':>6}")
    for r in f.itertuples():
        sel = {"home": r.home, "away": r.away, "over": "O", "under": "U"}[r.selection]
        ln = "" if r.market == "h2h" else f"{r.line:+g}" if r.market == "spreads" else f"{r.line:g}"
        bet = f"{_mk(r.market)} {sel}{ln}"
        rows.append(f"{r.away + '@' + r.home:<10}{bet:<14}{_pct(r.model_prob):>6}{_pct(r.book_prob):>6}"
                    f"{100 * r.edge:>+6.1f}{r.best_price:>+6.0f}")
    rows.append("</pre>")
    rows.append("Tracking only: edges are small and most will lose. See nightly recap for the record.")
    return "\n".join(rows)


def props_message(props: pd.DataFrame, top_n: int) -> str:
    out = ["🎯 <b>Prop projections</b> (model only, no book lines on free plan)"]
    ag = props[props.market == "anytime_goal"].sort_values("p_over", ascending=False).head(top_n)
    out.append("<b>Anytime goal</b><pre>")
    out += [f"{escape(str(r.player))[:18]:<18} {r.team:<4}{_pct(r.p_over):>5} ({_odds(r.p_over)})" for r in ag.itertuples()]
    out.append("</pre><b>Shots on goal</b><pre>")
    sog = props[(props.market == "sog") & (props.line == 2.5)].sort_values("mean", ascending=False).head(top_n)
    out += [f"{escape(str(r.player))[:18]:<18} {r.team:<4}{r.mean:>4.1f} o2.5 {_pct(r.p_over):>4}" for r in sog.itertuples()]
    out.append("</pre><b>Points</b><pre>")
    pts = props[(props.market == "points") & (props.line == 0.5)].sort_values("p_over", ascending=False).head(top_n)
    out += [f"{escape(str(r.player))[:18]:<18} {r.team:<4}{r.mean:>4.2f} o0.5 {_pct(r.p_over):>4}" for r in pts.itertuples()]
    out.append("</pre><b>Goalie saves</b><pre>")
    sv = props[props.market == "saves"].groupby(["player", "team"], as_index=False)["mean"].first()
    out += [f"{escape(str(r.player))[:18]:<18} {r.team:<4}{r.mean:>5.1f}" for r in sv.sort_values("mean", ascending=False).itertuples()]
    out.append("</pre>Full list (every player, every line) in the CSV.")
    return "\n".join(out)


def changes_message(changes: list[str]) -> str:
    return "🔄 <b>Pregame update</b>\n" + "\n\n".join(changes)


def recap_message(date_str: str, day: dict, season: dict, by_market: pd.DataFrame, by_bucket: pd.DataFrame,
                  by_goalie: pd.DataFrame) -> str:
    def block(title, s):
        if not s:
            return f"<b>{title}</b>: nothing graded yet"
        wr = f"{100 * s['flagged_win_rate']:.0f}%" if s.get("flagged_win_rate") is not None else "–"
        clv = (f"{100 * s['flagged_clv_moved_our_way']:.0f}%" if s.get("flagged_clv_moved_our_way") is not None else "–")
        better = "model" if s["model_logloss"] < s["book_logloss"] else "book"
        return (f"<b>{title}</b>\n"
                f"All lines: model logloss {s['model_logloss']:.4f} vs book {s['book_logloss']:.4f} → {better} closer\n"
                f"Flagged: {s['flagged']} · won {wr} · {s['flagged_units']:+.2f}u · line moved our way {clv}")

    parts = [f"📊 <b>Results recap · {date_str}</b>", block("Last night", day), block("Season to date", season)]
    for title, df in (("By market", by_market), ("By edge size", by_bucket), ("By goalie status", by_goalie)):
        if df is not None and not df.empty:
            parts.append(f"<b>{title}</b> (flagged)<pre>" + df.to_string(index=False, float_format=lambda x: f"{x:.2f}") + "</pre>")
    return "\n\n".join(parts)
