"""Nightly: ingest finished games, grade logged disagreements, post the recap.

python -m nhlmodel.pipelines.nightly
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import out_dir, season_for, today_local
from nhlmodel.pipelines.backfill import backfill_season, refresh_extras
from nhlmodel.publish import format as F
from nhlmodel.publish import telegram as TG
from nhlmodel.store import db
from nhlmodel.tracking import grade as GR

log = logging.getLogger("nightly")


def results_frame(season: int) -> pd.DataFrame:
    g = db.read("games", partitions=[str(season)])
    if g.empty:
        return g
    return g.rename(columns={"home_abbrev": "home", "away_abbrev": "away"})[
        ["game_date", "home", "away", "home_score", "away_score"]]


def model_accuracy(season: int, day) -> str | None:
    """Moneyline accuracy of the morning projections (no odds needed)."""
    proj = db.read("projections")
    if proj.empty:
        return None
    p = proj[(proj.run_type == "morning") & (proj.market == "h2h") & (proj.selection == "home")]
    p = p.sort_values("ts").drop_duplicates(["game_date", "home", "away"], keep="last")
    res = results_frame(season)
    m = p.merge(res, on=["game_date", "home", "away"])
    if m.empty:
        return None
    y = (m.home_score > m.away_score).astype(int)
    pr = m.model_prob.clip(1e-6, 1 - 1e-6)
    m["ll"] = -(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    m["fav_won"] = ((pr >= 0.5) == (y == 1)).astype(int)
    last = m[m.game_date == day.isoformat()]
    lines = ["🎯 <b>Model accuracy</b> (moneyline, no odds needed)"]
    if len(last):
        lines.append(f"Last night: favourites won {int(last.fav_won.sum())}/{len(last)}")
    lines.append(f"Season: {len(m)} games · log-loss {m.ll.mean():.4f} (coin flip 0.6931) · "
                 f"favourites won {100 * m.fav_won.mean():.0f}%")
    return "\n".join(lines)


def run() -> None:
    today = today_local()
    yesterday = today - timedelta(days=1)
    season = season_for(yesterday)
    added = backfill_season(season)
    log.info("ingested %d new games", added)
    refresh_extras(season)

    acc = model_accuracy(season, yesterday)
    if acc:
        TG.send_message(acc)

    dis = db.read("disagreements")
    if dis.empty:
        log.info("no disagreements logged yet")
        return
    dis = dis[dis.run_type == "morning"]  # grade the opening-line comparison; pregame rows are context
    res = results_frame(season)
    cons = db.read("consensus")
    graded = GR.grade(dis, res, cons)
    if graded.empty:
        log.info("nothing to grade yet")
        return
    # grades are recomputed every night from the stored rows (not stored, to keep git history small)
    graded.to_csv(out_dir() / f"grades_{season}.csv", index=False)

    day = graded[graded.game_date == yesterday.isoformat()]
    graded = graded.assign(edge_bucket=GR.edge_bucket(graded.edge).astype(str))
    msg = F.recap_message(
        yesterday.strftime("%a %b %-d"), GR.summarize(day), GR.summarize(graded),
        GR.breakdown(graded, "market"), GR.breakdown(graded, "edge_bucket"), GR.breakdown(graded, "goalie_status"))
    TG.send_message(msg)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run()


if __name__ == "__main__":
    main()
