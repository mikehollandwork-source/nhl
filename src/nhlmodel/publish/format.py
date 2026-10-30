"""Message formatting, written in a small HTML subset (<b>, <i>, <pre>) that publish/discord.py
converts to Discord markdown and Telegram accepts as-is."""

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
    for n in g.get("notes") or []:
        lines.append(f"<i>{escape(n)}</i>")
    for d in g.get("book_lines", []):
        lines.append(d)
    return "\n".join(lines)


def morning_message(date_str: str, version: str, cards: list[str], odds_note: str) -> str:
    head = f"🏒 <b>NHL projections · {date_str}</b>\nmodel {version} · {len(cards)} games\n{odds_note}"
    legend = ("Goalies: ✅ confirmed 🟡 likely ❔ unconfirmed 🤖 our guess\n"
              "Lines: model % and fair odds vs the best buy price on Polymarket (PM) or Kalshi (KS), as American "
              "odds; edge = model − price %, * = flagged")
    return "\n\n".join([head, legend, *cards])


def book_line(r) -> str:
    sel = {"home": r.home, "away": r.away, "over": "Over", "under": "Under"}[r.selection]
    ln = "" if r.market == "h2h" else f" {r.line:+g}" if r.market == "spreads" else f" {r.line:g}"
    flag = " 🚩" if r.flagged else ""
    src = {"polymarket": "Polymarket", "kalshi": "Kalshi"}.get(r.best_book, r.best_book)
    return (f"{_mk(r.market)} {sel}{ln}: model {_pct(r.model_prob)} ({_odds(r.model_prob)}) vs "
            f"{_am(r.best_price)} on {src} (edge {100 * r.edge:+.1f}){flag}")


def _mk(m: str) -> str:
    return {"h2h": "ML", "spreads": "PL", "totals": "Tot"}.get(m, m)


def disagreements_message(d: pd.DataFrame) -> str:
    f = d[d.flagged].sort_values("edge", ascending=False)
    if f.empty:
        return "🚩 <b>Disagreements</b>\nNone above threshold today."
    rows = ["🚩 <b>Disagreements</b> (model above the buy price by ≥ threshold)", "<pre>"]
    rows.append(f"{'game':<10}{'bet':<14}{'model':>6}{'price':>6}{'edge':>6}")
    for r in f.itertuples():
        sel = {"home": r.home, "away": r.away, "over": "O", "under": "U"}[r.selection]
        ln = "" if r.market == "h2h" else f"{r.line:+g}" if r.market == "spreads" else f"{r.line:g}"
        bet = f"{_mk(r.market)} {sel}{ln}"
        rows.append(f"{r.away + '@' + r.home:<10}{bet:<14}{_pct(r.model_prob):>6}{_am(r.best_price):>6}"
                    f"{100 * r.edge:>+6.1f}")
    rows.append("</pre>")
    rows.append("Tracking only: edges are small and most will lose. See nightly recap for the record.")
    return "\n".join(rows)


def _am(price: float) -> str:
    return f"+{price:.0f}" if price > 0 else f"{price:.0f}"


def lines_block(rows: list[dict]) -> str:
    """Model vs the best buy price on Polymarket (PM) / Kalshi (KS), as American odds.
    Fair = the model's probability as American odds; edge = model - price's implied %; * = flagged."""
    if not rows:
        return "<i>No Polymarket or Kalshi lines for this game.</i>"
    out = ["<pre>", f"{'':<12}{'Model':>6}{'Fair':>6}{'Price':>6}{'':>4}{'Edge':>6}"]
    for r in rows:
        flag = " *" if r.get("flag") else ""
        out.append(f"{r['label'][:12]:<12}{_pct(r['model']):>6}{_odds(r['model']):>6}{_am(r['price']):>6}"
                   f"{r['src']:>4}{100 * r['edge']:>+6.1f}{flag}")
    out.append("</pre>")
    return "\n".join(out)


def market_props_message(cmp: pd.DataFrame, top_n: int, flag_edge: float = 0.04) -> str:
    """Props that have a Kalshi line AND where the model is above Kalshi (over side only), biggest edge first.
    Props where the model is below the Kalshi price are not posted."""
    if cmp is None or cmp.empty:
        return "🎯 <b>Player props</b>\nNo Kalshi prop lines for tonight's games yet, so no props posted."
    cmp = cmp[cmp.edge > 0]
    if cmp.empty:
        return "🎯 <b>Player props</b>\nNo props where our projection is above Kalshi's price tonight."
    out = ["🎯 <b>Player props: model above Kalshi's price</b> (overs only; price = Kalshi buy price as "
           f"American odds; edge = model − price %, * = edge ≥ {100 * flag_edge:.0f})"]
    names = {"goals": "Goals", "points": "Points", "assists": "Assists"}
    for mk, title in names.items():
        g = cmp[cmp.market == mk]
        if g.empty:
            continue
        g = g.sort_values("edge", ascending=False).head(top_n)
        out.append(f"<b>{title}</b><pre>")
        out.append(f"{'player':<16}{'tm':<4}{'line':<5}{'model':>6}{'price':>6}{'edge':>6}")
        for r in g.itertuples():
            flag = " *" if r.edge >= flag_edge else ""
            out.append(f"{escape(str(r.player)[:15].ljust(16))}{r.team:<4}{'o' + format(r.line, 'g'):<5}"
                       f"{_pct(r.model_prob):>6}{_am(r.kalshi_price):>6}{100 * r.edge:>+6.1f}{flag}")
        out.append("</pre>")
    out.append("Shots, saves, blocks and PP points have no free line, so they are in the CSV only.")
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


def system_report_message(rep: dict) -> str:
    """Nightly system report: proven strengths, proven weaknesses, early leanings and trends."""
    def bullets(items, n):
        return [f"• {escape(i['text'])}" for i in items[:n]]

    out = [f"🧭 <b>System report</b> · through {rep['date']} · {rep['games']} games, {rep['prop_lines']} prop lines graded"]
    if rep["strengths"]:
        out += ["", "💪 <b>Strengths</b>", *bullets(rep["strengths"], 4)]
    if rep["weaknesses"]:
        out += ["", "⚠️ <b>Weaknesses</b>", *bullets(rep["weaknesses"], 4)]
    if not rep["strengths"] and not rep["weaknesses"]:
        out += ["", "No proven strengths or weaknesses yet. Verdicts need about 100 games "
                    "(or 300 prop lines) and a clear margin; until then these are leanings only."]
    if rep["leaning"]:
        out += ["", "🔎 <b>Leaning (not proven yet)</b>", *bullets(rep["leaning"], 8)]
    if rep["trends"]:
        out += ["", "📈 <b>Trends</b>", *bullets(rep["trends"], 5)]
    return "\n".join(out)
