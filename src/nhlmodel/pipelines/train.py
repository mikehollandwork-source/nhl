"""Train the xG model and game model, and run the out-of-sample backtest gate.

python -m nhlmodel.pipelines.train --mode backtest     # fit on train seasons, test on held-out season
python -m nhlmodel.pipelines.train --mode production   # fit on everything available (weekly job)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy.stats import poisson

from nhlmodel.config import data_dir, settings
from nhlmodel.features import history as H
from nhlmodel.features import xg as XG
from nhlmodel.sim.engine import simulate
from nhlmodel.sim.params import GameModel, fit_game_model
from nhlmodel.store import db

log = logging.getLogger("train")


def _version() -> str:
    return datetime.now(UTC).strftime("v%Y%m%d")


def train_xg(fit_seasons: list[int], calib_season: int, version: str) -> tuple[XG.XGModel, dict]:
    shots = db.read("shots", partitions=[str(s) for s in [*fit_seasons, calib_season]])
    model, metrics = XG.train(shots, fit_seasons, calib_season, version)
    log.info("xG metrics: %s", json.dumps(metrics, indent=1))
    return model, metrics


# ----------------------------------------------------------------------------- baselines

def _ml_from_poisson(lh: float, la: float, p_ot_home: float = 0.5, max_g: int = 15) -> float:
    ph = poisson.pmf(np.arange(max_g), lh)
    pa = poisson.pmf(np.arange(max_g), la)
    m = np.outer(ph, pa)
    win = np.tril(m, -1).sum()
    tie = np.trace(m)
    return float(win + tie * p_ot_home)


def backtest(h: H.History, gm: GameModel, test_season: int, home_rate: float, n_sims: int = 5000) -> dict:
    rows = H.training_rows(h)
    test = rows[(rows.season == test_season)].dropna(subset=["r_sogf", "o_soga", "lg_sogf"])
    games = h.games.set_index("game_id")
    rng = np.random.default_rng(settings()["sim"]["seed"])

    # simple Poisson baseline: goals for/against ratings only, home edge from training rows
    train_rows = rows[rows.season != test_season]
    hadv = float(train_rows.loc[train_rows.is_home == 1, "gf"].mean() / train_rows.loc[train_rows.is_home == 0, "gf"].mean())

    recs = []
    for gid, g in test.groupby("game_id"):
        if len(g) != 2 or gid not in games.index:
            continue
        hr = g[g.is_home == 1].iloc[0]
        ar = g[g.is_home == 0].iloc[0]
        gi = gm.game_input(hr, ar, int(games.loc[gid, "game_type"]))
        sim = simulate(gi, gm.constants, n=n_sims, step=settings()["sim"]["step_seconds"], rng=rng)
        actual = games.loc[gid]
        home_won = int(actual.home_score > actual.away_score)
        tot = actual.home_score + actual.away_score
        lh = hr.lg_gf * (hr.r_gf / hr.lg_gf) * (ar.r_ga / ar.lg_ga) * math.sqrt(hadv)
        la = ar.lg_gf * (ar.r_gf / ar.lg_gf) * (hr.r_ga / hr.lg_ga) / math.sqrt(hadv)
        recs.append({
            "game_id": gid, "home_won": home_won, "total": tot,
            "p_model": float(sim.home_win.mean()), "p_home_only": home_rate,
            "p_poisson": _ml_from_poisson(lh, la),
            "model_over55": float(((sim.final_h + sim.final_a) > 5.5).mean()),
            "model_over65": float(((sim.final_h + sim.final_a) > 6.5).mean()),
            "model_total": float((sim.final_h + sim.final_a).mean()),
            "model_sog": float((sim.sog_h + sim.sog_a).mean()),
            "actual_sog": float(actual.home_sog + actual.away_sog),
            "model_ot": float(sim.went_ot.mean()), "actual_ot": int(actual.last_period in ("OT", "SO")),
        })
    df = pd.DataFrame(recs)

    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def brier(p, y):
        return float(((p - y) ** 2).mean())

    y = df.home_won.to_numpy()
    over55 = (df.total > 5.5).astype(int).to_numpy()
    over65 = (df.total > 6.5).astype(int).to_numpy()
    report = {
        "test_season": test_season, "n_games": int(len(df)),
        "ml_logloss": {"model": ll(df.p_model, y), "poisson_baseline": ll(df.p_poisson, y),
                       "home_only_baseline": ll(df.p_home_only, y)},
        "ml_brier": {"model": brier(df.p_model, y), "poisson_baseline": brier(df.p_poisson, y),
                     "home_only_baseline": brier(df.p_home_only, y)},
        "totals": {"model_mean_total": float(df.model_total.mean()), "actual_mean_total": float(df.total.mean()),
                   "over55_logloss_model": ll(df.model_over55, over55),
                   "over55_logloss_const": ll(np.full(len(df), over55.mean()), over55),
                   "over65_logloss_model": ll(df.model_over65, over65),
                   "over65_logloss_const": ll(np.full(len(df), over65.mean()), over65)},
        "sog": {"model_mean": float(df.model_sog.mean()), "actual_mean": float(df.actual_sog.mean())},
        "ot": {"model_rate": float(df.model_ot.mean()), "actual_rate": float(df.actual_ot.mean())},
        "calibration_ml": _calibration(df.p_model.to_numpy(), y),
    }
    report["gate_pass"] = bool(report["ml_logloss"]["model"] < report["ml_logloss"]["poisson_baseline"]
                               and report["ml_logloss"]["model"] < report["ml_logloss"]["home_only_baseline"])
    return report


def _calibration(p: np.ndarray, y: np.ndarray, bins: int = 8) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == bins - 1 else p < edges[i + 1])
        if m.sum():
            out.append({"p_mean": round(float(p[m].mean()), 3), "win_rate": round(float(y[m].mean()), 3),
                        "n": int(m.sum())})
    return out


# ----------------------------------------------------------------------------- entry

def run(mode: str) -> dict:
    hcfg = settings()["history"]
    version = _version()
    available = sorted(int(p.stem) for p in (data_dir() / "games").glob("*.parquet"))
    if mode == "backtest":
        fit = hcfg["train_seasons"][:-1]
        calib = hcfg["train_seasons"][-1]
        game_seasons = hcfg["train_seasons"]
        test = hcfg["test_season"]
    else:
        complete = [s for s in available if s != max(available)] or available
        fit = complete[:-1] or complete
        calib = complete[-1]
        game_seasons = available
        test = None

    xg, xg_metrics = train_xg(fit, calib, version)
    h = H.load(xg, seasons=sorted(set(game_seasons + ([test] if test else []))))
    rows = H.training_rows(h)
    fit_rows = rows[rows.season.isin(game_seasons)]
    gm = fit_game_model(fit_rows, h.states[h.states.season.isin(game_seasons)],
                        h.games[h.games.season.isin(game_seasons)], version)
    log.info("game model: %s", json.dumps(json.loads(_dump(gm)), indent=1))

    result = {"version": version, "mode": mode, "xg": xg_metrics, "game_model": json.loads(_dump(gm))}
    if test:
        tr_games = h.games[h.games.season.isin(game_seasons)]
        home_rate = float((tr_games.home_score > tr_games.away_score).mean())
        result["backtest"] = backtest(h, gm, test, home_rate)
        log.info("backtest: %s", json.dumps(result["backtest"], indent=1))

    out = data_dir() / "models"
    out.mkdir(parents=True, exist_ok=True)
    if mode == "production":
        xg.save(out)
        rep = out / "ablation_report.json"
        if rep.exists():  # phase-1 fallback also gets its held-out dispersion factor
            gm.constants.dispersion = float(json.loads(rep.read_text()).get("final", {}).get("phase1_dispersion", 1.0))
        gm.save(out / "game_model.json")
    (out / f"train_report_{mode}.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def _dump(gm: GameModel) -> str:
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gm.json"
        gm.save(p)
        return p.read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["backtest", "production"], default="production")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(a.mode)


if __name__ == "__main__":
    main()
