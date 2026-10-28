"""Ablation backtest: which phase-2 input groups actually improve held-out predictions?

python -m nhlmodel.pipelines.ablation [--sims 2000] [--props-games 300]

Train on settings.history.train_seasons, test on test_season (never seen in fitting).
Order: phase-1 model -> phase-2 base (special teams + penalties) -> + lineup -> + refs ->
+ goalie_detail -> + travel -> + edge -> + pace. A group is KEPT only if the paired bootstrap
90% interval of the log-loss change is entirely below 0 on moneyline or totals, and the other
metric does not get worse beyond noise. Every variant uses the same random numbers per game,
so differences come from the model, not simulation noise.
Writes models/ablation_report.json and models/ablation_preds.parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from nhlmodel.config import data_dir, settings
from nhlmodel.features import assemble as A
from nhlmodel.features import history as H
from nhlmodel.features import players as PL
from nhlmodel.pipelines.train import train_xg
from nhlmodel.sim import model as M2
from nhlmodel.sim import props as P
from nhlmodel.sim.engine import simulate
from nhlmodel.sim.params import fit_game_model

log = logging.getLogger("ablation")


def _ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _sim_one(args) -> dict:
    gid, gi, consts, n_sims, step = args
    rng = np.random.default_rng(1000 + gid % 1_000_000)  # common random numbers across variants
    s = simulate(gi, consts, n=n_sims, step=step, rng=rng)
    tot = s.final_h + s.final_a
    return {"game_id": gid, "p_home": float(s.home_win.mean()), "p_o55": float((tot > 5.5).mean()),
            "p_o65": float((tot > 6.5).mean()), "exp_total": float(tot.mean()), "p_tie": float(s.went_ot.mean()),
            "pp_per_game": float((s.pp_count_h + s.pp_count_a).mean()) if s.pp_count_h is not None else np.nan}


def sim_preds(inputs: dict[int, object], consts, n_sims: int, step: int) -> pd.DataFrame:
    """Simulate every test game (in parallel across CPU cores)."""
    jobs = [(gid, gi, consts, n_sims, step) for gid, gi in inputs.items()]
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 2)) as ex:
        recs = list(ex.map(_sim_one, jobs, chunksize=16))
    return pd.DataFrame(recs)


def fit_dispersion(inputs: dict, consts, games: pd.DataFrame, n_sims: int, step: int,
                   grid=(1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4)) -> tuple[float, dict]:
    """Choose the dispersion factor that minimizes moneyline log-loss on a held-out season."""
    from dataclasses import replace
    table = {}
    for k in grid:
        d = score(sim_preds(inputs, replace(consts, dispersion=k), n_sims, step), games)
        table[k] = {"ml": float(d.ll_ml.mean()), "tot": float(d.ll_tot.mean())}
        log.info("dispersion %.2f: %s", k, table[k])
    best = min(table, key=lambda k: table[k]["ml"])
    return float(best), {str(k): v for k, v in table.items()}


def score(pred: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    d = pred.merge(games[["game_id", "home_score", "away_score", "home_reg_goals", "away_reg_goals"]], on="game_id")
    y = (d.home_score > d.away_score).astype(int).to_numpy()
    t = (d.home_score + d.away_score).to_numpy()
    d["ll_ml"] = _ll(d.p_home.to_numpy(), y)
    d["ll_tot"] = (_ll(d.p_o55.to_numpy(), (t > 5.5).astype(int)) + _ll(d.p_o65.to_numpy(), (t > 6.5).astype(int))) / 2
    return d


def bootstrap_delta(a: pd.DataFrame, b: pd.DataFrame, col: str, n: int = 2000, seed: int = 3) -> dict:
    """Paired bootstrap of mean(b[col] - a[col]) over games. Negative = b is better."""
    m = a[["game_id", col]].merge(b[["game_id", col]], on="game_id", suffixes=("_a", "_b"))
    diff = (m[f"{col}_b"] - m[f"{col}_a"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return {"delta": float(diff.mean()), "lo90": float(np.quantile(boots, 0.05)), "hi90": float(np.quantile(boots, 0.95)),
            "n": int(len(diff))}


def summary(d: pd.DataFrame, games: pd.DataFrame) -> dict:
    g = games.set_index("game_id").loc[d.game_id]
    return {"ml_logloss": float(d.ll_ml.mean()), "tot_logloss": float(d.ll_tot.mean()),
            "exp_total": float(d.exp_total.mean()), "actual_total": float((g.home_score + g.away_score).mean()),
            "p_tie": float(d.p_tie.mean()), "actual_tie": float((g.home_reg_goals == g.away_reg_goals).mean()),
            "pp_per_game": float(d.pp_per_game.mean())}


def run(n_sims: int, props_games: int, fixed_groups: list[str] | None = None, fit_disp: bool = False) -> dict:
    """Two stages, so the reported test numbers are not inflated by the choice of inputs:
      1. SELECT: fit on train_seasons[:-1], choose input groups greedily on train_seasons[-1].
      2. TEST:   refit on all train_seasons, evaluate phase 1, phase-2 base and the chosen model ONCE
                 on test_season (no decisions are made on the test season)."""
    hc = settings()["history"]
    train_s, test_s = hc["train_seasons"], hc["test_season"]
    sel_fit, sel_val = train_s[:-1], train_s[-1]
    step = settings()["sim"]["step_seconds"]
    t0 = time.time()
    xg, _ = train_xg(train_s[:-1], train_s[-1], "ablation")
    h = H.load(xg, seasons=[*train_s, test_s])
    ex = A.load_extras([*train_s, test_s])
    rapm, _st = A.rapm_by_season(h, ex, [*train_s[1:], test_s])
    rows = A.assemble(h, ex, rapm)
    ok = rows.valid.fillna(False) & (rows.lgs_ev_sogf > 0) & (rows.s_ev_sogf > 0) & (rows.o_ev_soga > 0)
    rows = rows[ok]
    v1_rows = H.training_rows(h)
    log.info("data ready in %.0fs", time.time() - t0)
    hg = h.games[h.games.season.isin(sel_fit)]
    pace_var = float(max(np.cov(hg.home_reg_goals - hg.home_en_goals, hg.away_reg_goals - hg.away_en_goals)[0, 1], 0)
                     / ((hg.home_reg_goals - hg.home_en_goals).mean() * (hg.away_reg_goals - hg.away_en_goals).mean()))

    def stage(fit_seasons, eval_season):
        ev = rows[rows.season == eval_season]
        pairs = {gid: (g[g.is_home == 1].iloc[0], g[g.is_home == 0].iloc[0], int(g.game_type.iloc[0]))
                 for gid, g in ev.groupby("game_id") if len(g) == 2}
        return {"fit_rows": rows[rows.season.isin(fit_seasons)], "pairs": pairs,
                "games": h.games[h.games.game_id.isin(pairs)],
                "states": h.states[h.states.season.isin(fit_seasons)],
                "fit_games": h.games[h.games.season.isin(fit_seasons)],
                "pens": ex.penalties[ex.penalties.season.isin(fit_seasons)], "fit_seasons": fit_seasons,
                "scales": None}

    def phase1(st, k: float = 1.0, keep: dict | None = None):
        gm1 = fit_game_model(v1_rows[v1_rows.season.isin(st["fit_seasons"])], st["states"], st["fit_games"], "v1")
        gm1.constants.dispersion = k
        inp = {}
        for gid, g in v1_rows[v1_rows.game_id.isin(st["pairs"])].groupby("game_id"):
            if len(g) == 2:
                inp[gid] = gm1.game_input(g[g.is_home == 1].iloc[0], g[g.is_home == 0].iloc[0], st["pairs"][gid][2])
        if keep is not None:
            keep.update({"inputs": inp, "consts": gm1.constants})
        return score(sim_preds(inp, gm1.constants, n_sims, step), st["games"])

    def evaluate(st, groups, pace, k: float = 1.0, keep: dict | None = None):
        gm = M2.fit(st["fit_rows"], groups, st["states"], st["fit_games"], st["pens"], "+".join(groups) or "base",
                    pace_var=pace)
        if st["scales"] is None:
            M2.calibrate(gm, st["fit_rows"], st["fit_games"])
            st["scales"] = (gm.constants.pull_time_scale, gm.constants.pen_scale, gm.constants.goal_scale)
        gm.constants.pull_time_scale, gm.constants.pen_scale, gm.constants.goal_scale = st["scales"]
        gm.constants.dispersion = k
        inputs = {gid: gm.game_input(hr, ar, gt) for gid, (hr, ar, gt) in st["pairs"].items()}
        if keep is not None:
            keep.update({"inputs": inputs, "consts": gm.constants})
        return gm, score(sim_preds(inputs, gm.constants, n_sims, step), st["games"])

    report: dict = {"protocol": "select on validation season, test once", "train_seasons": train_s,
                    "selection": {"fit": sel_fit, "validate": sel_val, "variants": {}, "decisions": []},
                    "test": {"season": test_s, "variants": {}, "comparisons": {}}, "sims": n_sims}

    # ---- 1. selection on the validation season
    sel = stage(sel_fit, sel_val)
    sp: dict[str, pd.DataFrame] = {}
    keep1: dict = {}
    sp["phase1"] = phase1(sel, keep=keep1)
    report["selection"]["variants"]["phase1"] = summary(sp["phase1"], h.games)
    if fixed_groups is None:
        _, sp["base"] = evaluate(sel, [], 0.0)
        report["selection"]["variants"]["base"] = summary(sp["base"], h.games)
        decide("base", "phase1", sp, report["selection"])
        incumbent, kept, pace_on = "base", [], False
        candidates = [["lineup_basic", "lineup_repl", "lineup"], ["refs"], ["goalie_detail"], ["travel"], ["edge"], ["pace"]]
        for options in candidates:
            passing = []
            for grp in options:
                name = "+" + grp
                if grp == "pace":
                    if pace_var <= 0:
                        continue
                    _, sp[name] = evaluate(sel, kept, pace_var)
                else:
                    _, sp[name] = evaluate(sel, [*kept, grp], pace_var if pace_on else 0.0)
                report["selection"]["variants"][name] = summary(sp[name], h.games)
                decide(name, incumbent, sp, report["selection"])
                if report["selection"]["decisions"][-1]["kept"]:
                    passing.append((report["selection"]["decisions"][-1]["ml"]["delta"], grp, name))
            if passing:
                _, grp, name = min(passing)
                if grp == "pace":
                    pace_on = True
                else:
                    kept.append(grp)
                incumbent = name
        final_pace = pace_var if pace_on else 0.0
    else:
        kept, final_pace = list(fixed_groups), 0.0
    # dispersion: both models are refit-free here; only the stretch factor changes
    keep2: dict = {}
    _, sp["final_k1"] = evaluate(sel, kept, final_pace, 1.0, keep=keep2)
    if fit_disp:
        k1, t1 = fit_dispersion(keep1["inputs"], keep1["consts"], sel["games"], max(1000, n_sims // 2), step)
        k2, t2 = fit_dispersion(keep2["inputs"], keep2["consts"], sel["games"], max(1000, n_sims // 2), step)
    else:
        k1, t1, k2, t2 = 1.0, {}, 1.0, {}
    report["selection"]["dispersion"] = {"phase1": {"best": k1, "grid": t1}, "final": {"best": k2, "grid": t2}}
    report["final"] = {"groups": kept, "pace_var": final_pace, "dispersion": k2, "phase1_dispersion": k1}
    log.info("selected groups %s pace %s dispersion phase1 %.2f final %.2f", kept, final_pace, k1, k2)

    # ---- 2. one evaluation on the untouched test season
    te = stage(train_s, test_s)
    tp = {"phase1": phase1(te), "phase1_disp": phase1(te, k1)}
    _, tp["final_raw"] = evaluate(te, kept, final_pace, 1.0)
    gm_final, tp["final"] = evaluate(te, kept, final_pace, k2)
    for k, d in tp.items():
        report["test"]["variants"][k] = summary(d, h.games)
    for a, b in (("phase1", "phase1_disp"), ("phase1", "final_raw"), ("phase1", "final"), ("phase1_disp", "final")):
        report["test"]["comparisons"][f"{b} vs {a}"] = {"ml": bootstrap_delta(tp[a], tp[b], "ll_ml"),
                                                       "tot": bootstrap_delta(tp[a], tp[b], "ll_tot")}
    log.info("test: %s", json.dumps(report["test"], default=float))

    if props_games > 0:
        gm_final.constants.sog_var = float(settings()["props"].get("team_sog_var", 0.0))
        report["props"] = props_backtest(h, ex, gm_final, te["pairs"], props_games, n_sims, step,
                                         player_var=float(settings()["props"].get("player_var", 0.0)))

    out = data_dir() / "models"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_report.json").write_text(json.dumps(report, indent=2, default=float))
    allp = pd.concat([d.assign(stage="select", variant=k) for k, d in sp.items()] +
                     [d.assign(stage="test", variant=k) for k, d in tp.items()], ignore_index=True)
    allp.to_parquet(out / "ablation_preds.parquet", index=False)
    return report


def decide(name: str, incumbent: str, preds: dict, report: dict, always_keep: bool = False) -> None:
    """Gate: keep if the 90% interval of the log-loss change is below 0 on moneyline or totals and the
    other metric does not get worse by more than its own interval width."""
    a, b = preds[incumbent], preds[name]
    ml = bootstrap_delta(a, b, "ll_ml")
    tot = bootstrap_delta(a, b, "ll_tot")
    better = (ml["hi90"] < 0 and tot["delta"] <= abs(tot["hi90"] - tot["lo90"])) or \
             (tot["hi90"] < 0 and ml["delta"] <= abs(ml["hi90"] - ml["lo90"]))
    if name == "base":
        # the structural change is kept unless it is clearly worse (it is needed for PP props)
        better = ml["lo90"] < 0.0005 and tot["lo90"] < 0.0005
    report["decisions"].append({"variant": name, "vs": incumbent, "ml": ml, "tot": tot,
                                "kept": bool(better or always_keep)})
    log.info("decision %s vs %s: kept=%s ml=%s tot=%s", name, incumbent, better, ml, tot)


# ----------------------------------------------------------------------------- props backtest

def props_backtest(h, ex, gm, pairs: dict, n_games: int, n_sims: int, step: int, player_var: float = 0.0) -> dict:
    """Projected vs actual box-score props on a sample of test games, phase-1 allocation vs phase-2."""
    rng0 = np.random.default_rng(9)
    gids = sorted(pairs)
    gids = [gids[i] for i in rng0.choice(len(gids), size=min(n_games, len(gids)), replace=False)]
    ixg = PL.player_ixg(h.shots)
    events = PL.player_state_events(h.shots)
    a_rates = PL.league_assist_rates(h.shots)
    a_state = PL.league_assist_rates_by_state(h.shots)
    pg = h.player_games
    actual = pg.set_index(["game_id", "player_id"])
    gg = h.goalie_games
    lines = {"sog": [1.5, 2.5, 3.5], "points": [0.5, 1.5], "goals": [0.5], "assists": [0.5], "ppp": [0.5],
             "blocks": [0.5, 1.5]}
    recs = []
    for gid in gids:
        hr, ar, gt = pairs[gid]
        gi = gm.game_input(hr, ar, gt)
        asof = pd.to_datetime(hr.game_date).date() if hr.game_date is not None else None
        sim = simulate(gi, gm.constants, n=n_sims, step=step, rng=np.random.default_rng(2000 + gid % 1_000_000))
        for side, team_id in (("home", hr.team_id), ("away", ar.team_id)):
            ids = pg[(pg.game_id == gid) & (pg.team_id == team_id)].player_id.tolist()
            if not ids:
                continue
            r1 = PL.player_rates(pg, ixg, asof, ids)
            tp1 = P.TeamProps(r1, a_rates["p_a1"], a_rates["p_a2_given_a1"],
                              PL.team_pp_goal_frac(h.shots, team_id, asof, a_rates["pp_goal_frac"]),
                              a_rates["pp_pts_per_goal"], PL.team_block_ratio(pg, h.games, team_id, asof))
            gk, ek = getattr(sim, f"goals_{side[0]}"), getattr(sim, f"en_{side[0]}")
            sk, ok = getattr(sim, f"sog_{side[0]}"), getattr(sim, f"sog_{'a' if side == 'home' else 'h'}")
            o1 = P.team_player_sims(gk, ek, sk, ok, tp1, np.random.default_rng(gid))
            r2 = PL.state_weights(PL.player_state_rates(pg, events, ex.player_toi, asof, ids))
            tp2 = P.TeamPropsV2(r2, a_state, tp1.block_ratio, player_var)
            o2 = P.team_player_sims_v2(sim, side, tp2, np.random.default_rng(gid))
            for j, pid in enumerate(ids):
                act = actual.loc[(gid, pid)]
                for stat, lns in lines.items():
                    y_val = act[stat]
                    for ln in lns:
                        y = int(y_val > ln)
                        recs.append({"stat": stat, "line": ln, "y": y,
                                     "p1": float((o1[stat][:, j] > ln).mean()), "p2": float((o2[stat][:, j] > ln).mean())})
        for side, opp in (("home", "away"), ("away", "home")):
            gl = gg[(gg.game_id == gid) & gg.starter & (gg.is_home == (side == "home"))]
            if gl.empty or gl.toi.iloc[0] < 50:
                continue
            saves = getattr(sim, f"sog_{opp[0]}") - getattr(sim, f"goals_{opp[0]}")
            for ln in (22.5, 25.5, 28.5):
                p = float((saves > ln).mean())
                recs.append({"stat": "saves", "line": ln, "y": int(gl.saves.iloc[0] > ln), "p1": p, "p2": p})
    d = pd.DataFrame(recs)
    out = {}
    for (stat, ln), g in d.groupby(["stat", "line"]):
        out[f"{stat}>{ln}"] = {"n": int(len(g)), "rate": float(g.y.mean()),
                               "pred1": float(g.p1.mean()), "pred2": float(g.p2.mean()),
                               "brier1": float(((g.p1 - g.y) ** 2).mean()), "brier2": float(((g.p2 - g.y) ** 2).mean()),
                               "brier_const": float(((g.y.mean() - g.y) ** 2).mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--props-games", type=int, default=300)
    ap.add_argument("--groups", nargs="*", default=None,
                    help="skip group selection and use these groups (e.g. --groups lineup_repl)")
    ap.add_argument("--dispersion", action="store_true",
                    help="also fit a dispersion factor on the validation season (tested Oct 2026: did not transfer)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(a.sims, a.props_games, a.groups, a.dispersion)


if __name__ == "__main__":
    main()
