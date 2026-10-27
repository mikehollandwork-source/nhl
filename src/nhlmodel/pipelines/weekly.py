"""Weekly: retrain models on all data so far, then post the line reverse-engineering report.

python -m nhlmodel.pipelines.weekly
"""

from __future__ import annotations

import logging

from nhlmodel.config import season_for, today_local
from nhlmodel.market import tells
from nhlmodel.pipelines import train
from nhlmodel.pipelines.nightly import results_frame
from nhlmodel.publish import telegram as TG
from nhlmodel.store import db

log = logging.getLogger("weekly")


def run() -> None:
    res = train.run("production")
    v = res["version"]
    m = res["game_model"]
    TG.send_message(
        f"🛠 <b>Weekly retrain</b> · model {v}\n"
        f"xG log-loss {res['xg']['logloss_cal']:.4f} (no-skill {res['xg']['logloss_base']:.4f}), AUC {res['xg']['auc']:.3f}\n"
        f"Home ice: SOG {m['sog']['is_home']:+.3f}, goals/shot {m['goal']['is_home']:+.3f} (log scale)\n"
        f"Back-to-back: SOG {m['sog']['b2b']:+.3f}, goals/shot {m['goal']['b2b']:+.3f}")

    odds = db.read("odds_snapshots")
    cons = db.read("consensus")
    if odds.empty or cons.empty:
        TG.send_message("🔍 <b>Line reverse-engineering</b>\nNo lines collected yet.")
        return
    results = results_frame(season_for(today_local()))
    TG.send_message(tells.report(odds, cons, db.read("projections"), db.read("slates"), results))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run()


if __name__ == "__main__":
    main()
