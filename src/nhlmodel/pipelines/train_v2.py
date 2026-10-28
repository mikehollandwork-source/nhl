"""Train the phase-2 production model on all stored data.

python -m nhlmodel.pipelines.train_v2

Writes models/game_model_v2.json, models/rapm.parquet and models/pp_units.json.
Input groups come from the latest ablation report (models/ablation_report.json) unless
settings.model.groups_from_ablation is false.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nhlmodel.config import data_dir, settings
from nhlmodel.features import assemble as A
from nhlmodel.features import history as H
from nhlmodel.features import rapm as RP
from nhlmodel.features.xg import XGModel
from nhlmodel.sim import model as M2

log = logging.getLogger("train_v2")


def chosen_groups() -> tuple[list[str], float, float]:
    """(input groups, pace variance, dispersion) from the latest ablation, or from settings."""
    cfg = settings()["model"]
    rep = data_dir() / "models" / "ablation_report.json"
    if cfg.get("groups_from_ablation") and rep.exists():
        final = json.loads(rep.read_text()).get("final", {})
        return (list(final.get("groups", [])), float(final.get("pace_var", 0.0)),
                float(final.get("dispersion", cfg.get("dispersion", 1.0))))
    return list(cfg.get("groups", [])), float(cfg.get("pace_var", 0.0)), float(cfg.get("dispersion", 1.0))


def pp_unit_minutes(player_toi: pd.DataFrame) -> dict:
    """Typical PP minutes for a PP1 / PP2 skater: per team-game, rank skaters by PP time."""
    t = player_toi[player_toi.toi_pp_s > 0].copy()
    t["rank"] = t.groupby(["game_id", "team_id"]).toi_pp_s.rank(ascending=False, method="first")
    pp1 = t[t["rank"] <= 5].toi_pp_s.mean() / 60
    pp2 = t[(t["rank"] > 5) & (t["rank"] <= 10)].toi_pp_s.mean() / 60
    return {"pp1_min": float(pp1), "pp2_min": float(pp2)}


def run() -> dict:
    version = datetime.now(UTC).strftime("v2-%Y%m%d")
    mdir = data_dir() / "models"
    xg = XGModel.load(mdir)
    h = H.load(xg)
    ex = A.load_extras()
    seasons = sorted(int(s) for s in h.games.season.unique())
    rapm_hist, st = A.rapm_by_season(h, ex, seasons[1:])
    rows = A.assemble(h, ex, rapm_hist)
    ok = rows.valid.fillna(False) & (rows.lgs_ev_sogf > 0) & (rows.s_ev_sogf > 0) & (rows.o_ev_soga > 0)
    rows = rows[ok]
    groups, pace, disp = chosen_groups()
    gm = M2.fit(rows, groups, h.states, h.games, ex.penalties, version, pace_var=pace)
    M2.calibrate(gm, rows, h.games)
    gm.constants.dispersion = disp
    gm.constants.sog_var = float(settings()["props"].get("team_sog_var", 0.0))
    gm.meta["pp_units"] = pp_unit_minutes(ex.player_toi)
    gm.save(mdir / "game_model_v2.json")
    # production player impacts: all seasons, most recent weighted most
    prod = RP.fit_all(st, latest_season=int(st.season.max()))
    prod.to_parquet(mdir / "rapm.parquet", index=False)
    log.info("saved %s with groups %s (%d rows, %d players rated)", version, groups, len(rows), len(prod))
    return {"version": version, "groups": groups, "pace_var": pace, "dispersion": disp, "rows": int(len(rows)),
            "coefs": {k: dict(zip(["b0", *gm.feats[k]], np.round(v, 4).tolist(), strict=True)) for k, v in gm.coefs.items()}}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    print(json.dumps(run(), indent=1))


if __name__ == "__main__":
    main()
