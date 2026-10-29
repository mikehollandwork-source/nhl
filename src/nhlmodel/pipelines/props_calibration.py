"""Props calibration study: are player prop probabilities pulled too far toward average?

python -m nhlmodel.pipelines.props_calibration [--games 400] [--sims 2000]

Same protocol as the ablation:
  1. SELECT: game model fit on train_seasons[:-1]; candidates scored on a sample of train_seasons[-1].
  2. TEST (once): refit on all train_seasons; the current settings vs the chosen candidate on a sample of
     test_season. Kept only if the paired (by game) bootstrap 90% interval of the log-loss change is
     entirely below 0.
Each game is simulated once; only the player allocation changes between candidates, with the same random
numbers, so differences come from the settings. Candidates vary the shrinkage of player rates toward the
position average (prior minutes / xG), the recency half-life, and the per-game player variance; the last
candidate adds a per-market logit recalibration fitted on the selection season.
Writes models/props_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nhlmodel.config import data_dir, settings
from nhlmodel.features import assemble as A
from nhlmodel.features import history as H
from nhlmodel.features import players as PL
from nhlmodel.pipelines.train import train_xg
from nhlmodel.sim import model as M2
from nhlmodel.sim import props as P
from nhlmodel.sim.engine import simulate

log = logging.getLogger("props_calibration")
LINES = {"goals": [0.5, 1.5], "points": [0.5, 1.5, 2.5], "assists": [0.5, 1.5], "sog": [1.5, 2.5, 3.5]}
STAR_PPG = 0.85          # recent points per game that counts as a "star" in the tier table


def candidates(base: dict, base_var: float) -> dict[str, dict]:
    def c(var=base_var, **kw):
        return {"params": {**base, **kw}, "player_var": var}
    return {
        "current": c(),
        "ev150": c(ev_prior=150), "ev75": c(ev_prior=75), "ev30": c(ev_prior=30),
        "pp30": c(pp_prior=30), "pp15": c(pp_prior=15),
        "hl15": c(halflife=15), "hl60": c(halflife=60),
        "fin50": c(finishing_prior=50), "fin12": c(finishing_prior=12),
        "var0": c(var=0.0),
        "ev75_pp30": c(ev_prior=75, pp_prior=30),
        "ev75_pp30_var0": c(0.0, ev_prior=75, pp_prior=30),
        "ev30_pp15": c(ev_prior=30, pp_prior=15),
        "ev30_pp15_var0": c(0.0, ev_prior=30, pp_prior=15),
        "ev75_pp30_hl60": c(ev_prior=75, pp_prior=30, halflife=60),
    }


# globals for forked workers (set before the pool starts)
_CTX: dict = {}


def _game(gid: int) -> list[dict]:
    h, ex, gm, pairs, cands, n_sims, step, events, a_state = (
        _CTX[k] for k in ("h", "ex", "gm", "pairs", "cands", "n_sims", "step", "events", "a_state"))
    pg = h.player_games
    hr, ar, gt = pairs[gid]
    asof = pd.to_datetime(hr.game_date).date()
    sim = simulate(gm.game_input(hr, ar, gt), gm.constants, n=n_sims, step=step,
                   rng=np.random.default_rng(2000 + gid % 1_000_000))
    out = []
    for side, team_id in (("home", hr.team_id), ("away", ar.team_id)):
        act = pg[(pg.game_id == gid) & (pg.team_id == team_id) & (pg.position != "G")]
        ids = act.player_id.tolist()
        if not ids:
            continue
        hist, prior = PL.state_history(pg, events, ex.player_toi, asof, ids)
        pts = pg[(pg.player_id.isin(ids)) & (pd.to_datetime(pg.game_date).dt.date < asof)
                 & (pd.to_datetime(pg.game_date).dt.date >= asof - timedelta(days=365))].groupby("player_id").points.agg(["sum", "size"])
        star = {p: bool(p in pts.index and pts.loc[p, "size"] >= 20 and pts.loc[p, "sum"] / pts.loc[p, "size"] >= STAR_PPG)
                for p in ids}
        block_ratio = PL.team_block_ratio(pg, h.games, team_id, asof)
        actual = act.set_index("player_id")
        for name, cand in cands.items():
            r2 = PL.state_weights(PL.rates_from_history(hist, prior, ids, cand["params"]))
            tp = P.TeamPropsV2(r2, a_state, block_ratio, cand["player_var"])
            o = P.team_player_sims_v2(sim, side, tp, np.random.default_rng(gid))
            for j, pid in enumerate(ids):
                for stat, lns in LINES.items():
                    v = actual.loc[pid, stat]
                    for ln in lns:
                        out.append({"cand": name, "game_id": gid, "player_id": pid, "stat": stat, "line": ln,
                                    "y": int(v > ln), "p": float((o[stat][:, j] > ln).mean()), "star": star[pid]})
    return out


def run_games(ctx: dict, gids: list[int]) -> pd.DataFrame:
    _CTX.clear()
    _CTX.update(ctx)
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 2)) as pool:
        recs = [r for rs in pool.map(_game, gids, chunksize=4) for r in rs]
    return pd.DataFrame(recs)


def _ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_recal(d: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-market logit recalibration p' = sigmoid(a + b * logit p), fitted by maximum likelihood."""
    out = {}
    for stat, g in d.groupby("stat"):
        p = np.clip(g.p.to_numpy(), 1e-4, 1 - 1e-4)
        z, y = np.log(p / (1 - p)), g.y.to_numpy()

        def nll(ab, z=z, y=y):
            q = 1 / (1 + np.exp(-(ab[0] + ab[1] * z)))
            return _ll(q, y).sum()
        r = minimize(nll, x0=[0.0, 1.0], method="Nelder-Mead")
        out[stat] = (float(r.x[0]), float(r.x[1]))
    return out


def apply_recal(d: pd.DataFrame, ab: dict) -> pd.DataFrame:
    d = d.copy()
    p = np.clip(d.p.to_numpy(), 1e-4, 1 - 1e-4)
    a = d.stat.map(lambda s: ab.get(s, (0.0, 1.0))[0]).to_numpy()
    b = d.stat.map(lambda s: ab.get(s, (0.0, 1.0))[1]).to_numpy()
    d["p"] = 1 / (1 + np.exp(-(a + b * np.log(p / (1 - p)))))
    return d


def tiers(d: pd.DataFrame) -> dict:
    """Mean predicted vs actual for stars and the rest, and by predicted-probability bucket (points 0.5)."""
    out = {}
    for t, g in d[(d.stat == "points") & (d.line == 0.5)].groupby("star"):
        out["stars" if t else "others"] = {"n": int(len(g)), "pred": float(g.p.mean()), "actual": float(g.y.mean())}
    b = d[(d.stat == "points") & (d.line == 0.5)].assign(bucket=lambda x: pd.cut(x.p, [0, .2, .35, .5, .65, 1]))
    for k, g in b.groupby("bucket", observed=True):
        out[f"p {k}"] = {"n": int(len(g)), "pred": float(g.p.mean()), "actual": float(g.y.mean())}
    return out


def by_game(d: pd.DataFrame) -> pd.Series:
    return d.assign(ll=_ll(d.p, d.y)).groupby("game_id").ll.mean()


def boot(a: pd.Series, b: pd.Series, n=2000, seed=5) -> dict:
    diff = (b - a.reindex(b.index)).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return {"delta": float(diff.mean()), "lo90": float(np.quantile(bs, .05)), "hi90": float(np.quantile(bs, .95)),
            "n_games": int(len(diff))}


def run(n_games: int, n_sims: int) -> dict:
    hc = settings()["history"]
    train_s, test_s = hc["train_seasons"], hc["test_season"]
    step = settings()["sim"]["step_seconds"]
    t0 = time.time()
    xg, _ = train_xg(train_s[:-1], train_s[-1], "ablation")
    h = H.load(xg, seasons=[*train_s, test_s])
    ex = A.load_extras([*train_s, test_s])
    rapm, _ = A.rapm_by_season(h, ex, [*train_s[1:], test_s])
    rows = A.assemble(h, ex, rapm)
    rows = rows[rows.valid.fillna(False) & (rows.lgs_ev_sogf > 0) & (rows.s_ev_sogf > 0) & (rows.o_ev_soga > 0)]
    rep = data_dir() / "models" / "ablation_report.json"
    groups = json.loads(rep.read_text())["final"]["groups"] if rep.exists() else ["lineup_repl"]
    events = PL.player_state_events(h.shots)
    a_state = PL.league_assist_rates_by_state(h.shots)
    base, base_var = PL.rate_params(), float(settings()["props"].get("player_var", 0.0))
    log.info("data ready in %.0fs; groups %s; current settings %s var %.3f", time.time() - t0, groups, base, base_var)

    def stage(fit_seasons, season):
        t1 = time.time()
        fr = rows[rows.season.isin(fit_seasons)]
        fg = h.games[h.games.season.isin(fit_seasons)]
        gm = M2.fit(fr, groups, h.states[h.states.season.isin(fit_seasons)], fg,
                    ex.penalties[ex.penalties.season.isin(fit_seasons)], "+".join(groups))
        M2.calibrate(gm, fr, fg)
        gm.constants.sog_var = float(settings()["props"].get("team_sog_var", 0.0))
        ev = rows[rows.season == season]
        pairs = {gid: (g[g.is_home == 1].iloc[0], g[g.is_home == 0].iloc[0], int(g.game_type.iloc[0]))
                 for gid, g in ev.groupby("game_id") if len(g) == 2}
        rng = np.random.default_rng(11 + season % 100)
        gids = sorted(pairs)
        gids = [int(x) for x in rng.choice(gids, size=min(n_games, len(gids)), replace=False)]
        log.info("game model for %s fit in %.0fs; %d games sampled", season, time.time() - t1, len(gids))
        return gm, pairs, gids

    cands = candidates(base, base_var)
    report: dict = {"current": {"params": base, "player_var": base_var}, "lines": LINES, "sims": n_sims}

    # ---- 1. selection
    gm, pairs, gids = stage(train_s[:-1], train_s[-1])
    sel = run_games({"h": h, "ex": ex, "gm": gm, "pairs": pairs, "cands": cands, "n_sims": n_sims, "step": step,
                     "events": events, "a_state": a_state}, gids)
    log.info("selection games scored in %.0fs", time.time() - t0)
    table = {k: float(_ll(g.p, g.y).mean()) for k, g in sel.groupby("cand")}
    best = min(table, key=table.get)
    recal = fit_recal(sel[sel.cand == best])
    report["selection"] = {"season": train_s[-1], "games": len(gids), "logloss": table, "best": best,
                           "recal": recal, "tiers_current": tiers(sel[sel.cand == "current"]),
                           "tiers_best": tiers(sel[sel.cand == best])}
    log.info("selection log-loss: %s", json.dumps(dict(sorted(table.items(), key=lambda x: x[1])), indent=1))
    log.info("best %s; recalibration %s", best, recal)

    # ---- 2. test once: current vs best (and best + recalibration)
    gm, pairs, gids = stage(train_s, test_s)
    keep = {"current": cands["current"], best: cands[best]}
    te = run_games({"h": h, "ex": ex, "gm": gm, "pairs": pairs, "cands": keep, "n_sims": n_sims, "step": step,
                    "events": events, "a_state": a_state}, gids)
    cur, bst = te[te.cand == "current"], te[te.cand == best]
    bst_r = apply_recal(bst, recal)
    report["test"] = {
        "season": test_s, "games": len(gids),
        "logloss": {"current": float(_ll(cur.p, cur.y).mean()), best: float(_ll(bst.p, bst.y).mean()),
                    f"{best}+recal": float(_ll(bst_r.p, bst_r.y).mean())},
        "vs_current": {best: boot(by_game(cur), by_game(bst)), f"{best}+recal": boot(by_game(cur), by_game(bst_r))},
        "by_market": {s: {"current": float(_ll(g.p, g.y).mean()),
                          best: float(_ll(bst[bst.stat == s].p, bst[bst.stat == s].y).mean()),
                          f"{best}+recal": float(_ll(bst_r[bst_r.stat == s].p, bst_r[bst_r.stat == s].y).mean())}
                      for s, g in cur.groupby("stat")},
        "tiers": {"current": tiers(cur), best: tiers(bst), f"{best}+recal": tiers(bst_r)}}
    # decision: the better of best / best+recal, kept only if its interval is entirely below 0
    opts = {k: v for k, v in report["test"]["vs_current"].items() if v["hi90"] < 0 and k != "current"}
    if best == "current":
        opts = {k: v for k, v in opts.items() if k.endswith("+recal")}
    choice = min(opts, key=lambda k: opts[k]["delta"]) if opts else None
    report["decision"] = {"keep": choice, "params": cands[best]["params"] if choice else base,
                          "player_var": cands[best]["player_var"] if choice else base_var,
                          "recal": recal if choice and choice.endswith("+recal") else None}
    log.info("test: %s", json.dumps(report["test"]["vs_current"], indent=1))
    log.info("decision: %s", report["decision"])
    out = data_dir() / "models"
    out.mkdir(parents=True, exist_ok=True)
    (out / "props_calibration.json").write_text(json.dumps(report, indent=2, default=float))
    pd.concat([sel.assign(stage="select"), te.assign(stage="test")]).to_parquet(out / "props_calibration_preds.parquet")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--sims", type=int, default=2000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(a.games, a.sims)


if __name__ == "__main__":
    main()
