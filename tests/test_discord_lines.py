import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nhlmodel.ingest import kalshi, polymarket
from nhlmodel.publish import discord as DC
from nhlmodel.publish import format as F
from nhlmodel.publish import notify
from nhlmodel.tracking import grade as GR

FIX = Path(__file__).parent / "fixtures"
D = date(2026, 10, 28)


# ---- Discord conversion ----

def test_to_markdown_tags_and_escaping():
    md = DC.to_markdown("<b>EDM @ CGY</b>\n<i>O&#x27;Reilly *gtd*</i>\n<pre>a_b  5%\nx*y</pre>end")
    assert "**EDM @ CGY**" in md
    assert "*O'Reilly \\*gtd\\**" in md
    assert "```\na_b  5%\nx*y\n```" in md          # no escaping inside code blocks


def test_chunks_limit_and_balanced_fences():
    text = "<b>t</b>\n<pre>" + "\n".join(f"row {i:04d} " + "x" * 60 for i in range(120)) + "</pre>\ntail"
    parts = DC.chunks(DC.to_markdown(text))
    assert len(parts) > 1
    assert all(len(p) <= 2000 for p in parts)
    assert all(p.count("```") % 2 == 0 for p in parts)
    assert parts[-1].endswith("tail")


def test_send_message_payload(monkeypatch):
    sent = []

    class R:
        status_code = 204
        text = ""

    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.test/hook")
    monkeypatch.delenv("NHL_DRY_RUN", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setattr(DC.httpx, "post", lambda url, **kw: sent.append((url, kw)) or R())
    notify.send_message("<b>hi</b> @everyone")
    assert len(sent) == 1
    url, kw = sent[0]
    assert url == "https://discord.test/hook"
    assert kw["json"]["allowed_mentions"] == {"parse": []}
    assert kw["json"]["content"].startswith("**hi**")


# ---- Polymarket ----

def _nick():
    return {"flames": "CGY", "oilers": "EDM"}


def test_polymarket_parse():
    ev = json.loads((FIX / "polymarket_cgy_edm.json").read_text())
    df = polymarket.parse_events(ev, D, [("EDM", "CGY")], _nick())
    ml = df[df.market == "h2h"].set_index("selection")
    assert ml.loc["home", "prob"] == pytest.approx(0.625)          # Oilers mid of 0.62 / 0.63
    assert ml.loc["home", "prob"] + ml.loc["away", "prob"] == pytest.approx(1.0)
    assert ml.loc["home", "bid"] == pytest.approx(0.62)            # flipped quotes for the second outcome
    sp = df[(df.market == "spreads") & (df.selection == "home") & (df.line == -1.5)]
    assert sp.prob.iloc[0] == pytest.approx(0.415)
    tot = df[(df.market == "totals") & (df.line == 6.5)].set_index("selection")
    assert tot.loc["over", "prob"] == pytest.approx(0.525)
    assert len(df[df.market == "spreads"]) == 12          # 6 spread markets x 2 sides
    # a market wider than max_spread is skipped
    wide = json.loads(json.dumps(ev))
    wide[0]["markets"][0].update({"bestBid": "0.40", "bestAsk": "0.60"})
    assert polymarket.parse_events(wide, D, [("EDM", "CGY")], _nick()).query("market == 'h2h'").empty
    assert polymarket.parse_events(ev, date(2026, 10, 29), [("EDM", "CGY")], _nick()).empty


# ---- Kalshi ----

def _players():
    return pd.DataFrame({"player_id": [1, 2, 3], "player": ["Ryan Strome", "Trent Frederic", "Alex Formenton"],
                         "team": ["EDM", "EDM", "EDM"]})


def test_kalshi_parse():
    mk = json.loads((FIX / "kalshi_pts_cgy_edm.json").read_text())
    df, miss = kalshi.parse_markets(mk, "points", D, _players())
    assert miss > 0                                    # players not in our toy roster
    s = df[(df.player_id == 1)].set_index("line")
    assert s.loc[0.5, "prob"] == pytest.approx(0.335)  # "Ryan Strome: 1+" -> over 0.5, mid of 0.32 / 0.35
    assert s.loc[1.5, "prob"] == pytest.approx(0.065)
    assert set(df.market) == {"points"}
    assert kalshi.date_code(D) == "26OCT28"


def test_kalshi_spread_filter():
    mk = [{"ticker": "KXNHLPTS-26OCT28CGYEDM-EDMRSTROME15-1", "yes_sub_title": "Ryan Strome: 1+",
           "yes_bid_dollars": "0.10", "yes_ask_dollars": "0.40"}]
    df, _ = kalshi.parse_markets(mk, "points", D, _players(), max_spread=0.10)
    assert df.empty


# ---- formatting, filter, grading ----

def test_lines_block_and_props_message():
    rows = [{"label": "ML EDM", "model": 0.55, "dk": 0.62, "dk_price": -188, "poly": 0.62, "edge": -0.07, "flag": False},
            {"label": "ML CGY", "model": 0.45, "dk": 0.38, "dk_price": 155, "poly": None, "edge": 0.07, "flag": True}]
    b = F.lines_block(rows)
    assert "-188" in b and "+155" in b and "–" in b and b.count("*") == 1
    cmp = pd.DataFrame({"player": ["Logan O'Connor"], "team": ["COL"], "market": ["goals"], "line": [0.5],
                        "model_prob": [0.21], "kalshi_prob": [0.08], "edge": [0.13]})
    md = DC.to_markdown(F.market_props_message(cmp, 15))
    assert "Logan O'Connor" in md
    assert "No Kalshi" in F.market_props_message(pd.DataFrame(), 15)
    under = cmp.assign(player="Leon Draisaitl", edge=-0.11)
    msg = F.market_props_message(pd.concat([cmp, under]), 15)
    assert "Draisaitl" not in msg                       # model below Kalshi -> not posted
    assert "No props" in F.market_props_message(under, 15)


def test_grade_props():
    lines = pd.DataFrame({"ts": ["t"] * 3, "game_date": ["2026-10-28"] * 3, "game_id": [9, 9, 9],
                          "player_id": [1, 1, 2], "market": ["points", "goals", "points"], "line": [0.5, 0.5, 0.5],
                          "model_prob": [0.7, 0.3, 0.5], "kalshi_prob": [0.5, 0.35, 0.5], "edge": [0.2, -0.05, 0.0]})
    pg = pd.DataFrame({"game_id": [9], "player_id": [1], "goals": [0], "assists": [1], "points": [1]})
    g = GR.grade_props(lines, pg)
    assert len(g) == 2                                 # player 2 did not play -> void
    pts = g[g.market == "points"].iloc[0]
    assert pts.outcome == 1.0 and pts.won and pts.units == pytest.approx(1.0)     # over at 0.50 -> +1u
    gl = g[g.market == "goals"].iloc[0]
    assert not gl.flagged and gl.units == 0.0          # model below Kalshi: never posted, never flagged
    s = GR.summarize_props(g)
    assert s["flagged"] == 1 and s["flagged_win_rate"] == 1.0
