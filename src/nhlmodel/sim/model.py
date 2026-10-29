"""Phase-2 game model: separate even-strength, power-play, shorthanded and penalty components.

Seven Poisson regressions (with exposure offsets), each fit on team-game rows:
  ev_sog   : even-strength shots on goal per second of even-strength time
  ev_goal  : even-strength goals per shot on goal
  pp_sog   : power-play shots per second of power-play time
  pp_goal  : power-play goals per shot
  sh_sog   : shorthanded shots per second shorthanded
  sh_goal  : shorthanded goals per shot
  pen      : power plays given up (times the team went shorthanded) per second of regulation
Optional feature groups (lineup, refs, goalie_detail, travel, edge) are added per the ablation
and switched on/off in config. Constants: score effects are estimated WITHIN games (each team-game
is compared with itself), which removes the bias that better teams both lead more and shoot more.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from nhlmodel.sim.engine import GameInput, SimConstants
from nhlmodel.sim.params import fit_constants, fit_poisson

BASE = {
    "ev_sog": ["off_ev_sog", "opp_def_ev_sog", "is_home", "b2b", "opp_b2b", "three_in_four", "opp_three_in_four"],
    "ev_goal": ["off_ev_q", "opp_def_ev_q", "finishing", "opp_goalie", "is_home", "b2b", "opp_b2b"],
    "pp_sog": ["off_pp_sog", "opp_def_sh_sog", "is_home"],
    "pp_goal": ["off_pp_q", "opp_def_sh_q", "opp_goalie"],
    "sh_sog": ["off_sh_sog"],
    "sh_goal": ["opp_goalie"],
    "pen": ["pen_taken", "opp_pen_drawn", "is_home"],
}
GROUPS = {
    # three mutually exclusive versions of the lineup adjustment (see features/lineup.py)
    "lineup_basic": {"ev_sog": ["lu0_off_sog", "lu0_opp_def_sog"], "ev_goal": ["lu0_off_xg", "lu0_opp_def_xg"]},
    "lineup_repl": {"ev_sog": ["lu_off_sog", "lu_opp_def_sog"], "ev_goal": ["lu_off_xg", "lu_opp_def_xg"]},
    "lineup": {"ev_sog": ["lu_off_sog", "lu_opp_def_sog"], "ev_goal": ["lu_off_xg", "lu_opp_def_xg", "lu_fin"],
               "pp_goal": ["lu_pp"]},
    "refs": {"pen": ["ref"]},
    "goalie_detail": {"ev_goal": ["opp_g_high", "opp_g_lowmid", "opp_g_rebound", "opp_g_starts7", "opp_g_b2b"],
                      "pp_goal": ["opp_g_high"]},
    "travel": {"ev_sog": ["tz_big", "opp_tz_big", "games_last7", "opp_games_last7", "road_trip_long"],
               "ev_goal": ["tz_big", "opp_tz_big"]},
    "edge": {"ev_sog": ["edge_bursts", "edge_oz", "opp_edge_bursts"], "ev_goal": ["edge_shots90"]},
    # official NHL team stats, rebuilt per game from play-by-play (features/macro.py)
    "macro": {"ev_sog": ["fo_diff", "zs_diff", "take_rel", "give_rel", "opp_give_rel"],
              "pp_sog": ["fo_pp"], "ev_goal": ["opp_give_rel"]},
    "physical": {"ev_sog": ["hits_rel", "opp_hits_rel"]},
    # per-player NHL EDGE (prior season), tonight's lineup vs the team's typical one (features/edge_player.py)
    "edge_player": {"ev_sog": ["ep_speed", "ep_bursts", "ep_oz", "ep_opp_speed"], "ev_goal": ["ep_oz"]},
}
# component -> (target count, exposure column or None, parent component whose prediction is the offset)
TARGETS = {
    "ev_sog": ("ev_sogf", "ev_secs", None), "ev_goal": ("ev_gf", None, "ev_sog"),
    "pp_sog": ("pp_sogf", "pp_secs", None), "pp_goal": ("pp_gf", None, "pp_sog"),
    "sh_sog": ("sh_sogf", "sh_secs", None), "sh_goal": ("sh_gf", None, "sh_sog"),
    "pen": ("pk_starts", "reg_secs", None),
}
# league base rate per component (per second, or per shot for goal components)
LEAGUE = {
    "ev_sog": lambda r: r.lgs_ev_sogf, "ev_goal": lambda r: r.lgs_ev_gf / r.lgs_ev_sogf,
    "pp_sog": lambda r: r.lgs_pp_sogf, "pp_goal": lambda r: r.lgs_pp_gf / r.lgs_pp_sogf,
    "sh_sog": lambda r: r.lgs_sh_sogf, "sh_goal": lambda r: r.lgs_sh_xgf / r.lgs_sh_sogf,
    "pen": lambda r: r.lgs_pen_taken,
}


def features(groups: list[str]) -> dict[str, list[str]]:
    f = {k: list(v) for k, v in BASE.items()}
    for gname in groups:
        for comp, cols in GROUPS.get(gname, {}).items():
            f[comp] += [c for c in cols if c not in f[comp]]
    return f


def design(rows: pd.DataFrame) -> pd.DataFrame:
    """All candidate features from assembled rows (s_* team, o_* opponent strength ratings, lgs_* league)."""
    eps = 1e-12
    r = rows
    X = pd.DataFrame(index=r.index)
    lg_evq = r.lgs_ev_xgf / r.lgs_ev_sogf
    X["off_ev_sog"] = np.log(r.s_ev_sogf / r.lgs_ev_sogf + eps)
    X["opp_def_ev_sog"] = np.log(r.o_ev_soga / r.lgs_ev_soga + eps)
    X["off_ev_q"] = np.log((r.s_ev_xgf / r.s_ev_sogf) / lg_evq + eps)
    X["opp_def_ev_q"] = np.log((r.o_ev_xga / r.o_ev_soga) / (r.lgs_ev_xga / r.lgs_ev_soga) + eps)
    X["finishing"] = np.log((r.s_ev_gf + eps) / (r.s_ev_xgf + eps))
    X["opp_goalie"] = np.log(r.opp_goalie_factor.fillna(1.0))
    X["off_pp_sog"] = np.log(r.s_pp_sogf / r.lgs_pp_sogf + eps)
    X["opp_def_sh_sog"] = np.log(r.o_sh_soga / r.lgs_sh_soga + eps)
    X["off_pp_q"] = np.log((r.s_pp_xgf / r.s_pp_sogf) / (r.lgs_pp_xgf / r.lgs_pp_sogf) + eps)
    X["opp_def_sh_q"] = np.log((r.o_sh_xga / r.o_sh_soga) / (r.lgs_sh_xga / r.lgs_sh_soga) + eps)
    X["off_sh_sog"] = np.log(r.s_sh_sogf / r.lgs_sh_sogf + eps)
    X["pen_taken"] = np.log(r.s_pen_taken / r.lgs_pen_taken + eps)
    X["opp_pen_drawn"] = np.log(r.o_pen_drawn / r.lgs_pen_drawn + eps)
    for c in ("is_home", "b2b", "opp_b2b", "three_in_four", "opp_three_in_four"):
        X[c] = r[c].astype(float)
    ev60_sog = r.lgs_ev_sogf * 3600
    ev60_xg = r.lgs_ev_xgf * 3600
    X["lu_off_sog"] = r.get("d_off_sog", 0.0) / ev60_sog
    X["lu_opp_def_sog"] = r.get("o_d_def_sog", 0.0) / ev60_sog
    X["lu_off_xg"] = r.get("d_off_xg", 0.0) / ev60_xg
    X["lu_opp_def_xg"] = r.get("o_d_def_xg", 0.0) / ev60_xg
    X["lu0_off_sog"] = r.get("d0_off_sog", 0.0) / ev60_sog
    X["lu0_opp_def_sog"] = r.get("o_d0_def_sog", 0.0) / ev60_sog
    X["lu0_off_xg"] = r.get("d0_off_xg", 0.0) / ev60_xg
    X["lu0_opp_def_xg"] = r.get("o_d0_def_xg", 0.0) / ev60_xg
    X["lu_fin"] = r.get("d_fin", 0.0)
    X["lu_pp"] = r.get("d_pp", 0.0) / (r.lgs_pp_xgf * 3600 / 5)
    X["ref"] = np.log(r.get("ref_factor", pd.Series(1.0, index=r.index)).fillna(1.0))
    for c in ("opp_g_high", "opp_g_lowmid", "opp_g_rebound"):
        X[c] = np.log(r.get(c, pd.Series(1.0, index=r.index)).fillna(1.0))
    for c in ("opp_g_starts7", "opp_g_b2b", "tz_big", "opp_tz_big", "games_last7", "opp_games_last7"):
        X[c] = r.get(c, pd.Series(0.0, index=r.index)).fillna(0.0).astype(float)
    X["road_trip_long"] = (r.get("road_trip", pd.Series(0, index=r.index)).fillna(0) >= 4).astype(float)
    for c in ("edge_bursts", "edge_oz", "opp_edge_bursts", "edge_shots90"):
        X[c] = r.get(c, pd.Series(0.0, index=r.index)).fillna(0.0).astype(float)
    def col(name, default):
        return r[name] if name in r else pd.Series(default, index=r.index)

    def logit(p):
        p = np.clip(p.astype(float), 0.01, 0.99)
        return np.log(p / (1 - p))

    X["fo_diff"] = logit(col("m_fo_pct", 0.5)) - logit(col("om_fo_pct", 0.5))
    X["zs_diff"] = col("m_oz_start_pct", 0.5) - col("om_oz_start_pct", 0.5)
    X["take_rel"] = np.log(col("m_takeaways60", 1.0) / col("lgm_takeaways60", 1.0))
    X["give_rel"] = np.log(col("m_giveaways60", 1.0) / col("lgm_giveaways60", 1.0))
    X["opp_give_rel"] = np.log(col("om_giveaways60", 1.0) / col("lgm_giveaways60", 1.0))
    X["fo_pp"] = logit(col("m_fo_pp_pct", 0.5)) - logit(col("lgm_fo_pp_pct", 0.5))
    X["hits_rel"] = np.log(col("m_hits60", 1.0) / col("lgm_hits60", 1.0))
    X["opp_hits_rel"] = np.log(col("om_hits60", 1.0) / col("lgm_hits60", 1.0))
    for name in ("speed", "bursts", "oz"):
        X[f"ep_{name}"] = col(f"d_e_{name}", 0.0).astype(float)
    X["ep_opp_speed"] = col("o_d_e_speed", 0.0).astype(float)
    return X.replace([np.inf, -np.inf], 0.0).fillna(0.0)


class GameModelV2:
    def __init__(self, coefs: dict[str, np.ndarray], feats: dict[str, list[str]], constants: SimConstants, meta: dict):
        self.coefs = coefs
        self.feats = feats
        self.constants = constants
        self.meta = meta

    @property
    def version(self) -> str:
        return self.meta.get("version", "dev")

    def _eta(self, comp: str, X: pd.DataFrame) -> np.ndarray:
        b = self.coefs[comp]
        return b[0] + X[self.feats[comp]].to_numpy() @ b[1:]

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Per-60 rates and per-shot goal probabilities for each team-row."""
        X = design(rows)
        out = pd.DataFrame(index=rows.index)
        for comp in ("ev_sog", "pp_sog", "sh_sog", "pen"):
            base = rows.apply(LEAGUE[comp], axis=1).to_numpy()
            out[comp] = base * np.exp(self._eta(comp, X)) * 3600
        for comp in ("ev_goal", "pp_goal", "sh_goal"):
            base = rows.apply(LEAGUE[comp], axis=1).to_numpy()
            out[comp] = np.clip(base * np.exp(self._eta(comp, X)), 0.005, 0.6)
        return out

    def game_input(self, home_row: pd.Series, away_row: pd.Series, game_type: int = 2) -> GameInput:
        p = self.predict(pd.DataFrame([home_row, away_row]).reset_index(drop=True))
        h, a = p.iloc[0], p.iloc[1]
        return GameInput(sog_h=h.ev_sog, sog_a=a.ev_sog, gp_h=h.ev_goal, gp_a=a.ev_goal, game_type=game_type,
                         pen_h=h.pen, pen_a=a.pen, pp_sog_h=h.pp_sog, pp_sog_a=a.pp_sog,
                         pp_gp_h=h.pp_goal, pp_gp_a=a.pp_goal, sh_sog_h=h.sh_sog, sh_sog_a=a.sh_sog,
                         sh_gp_h=h.sh_goal, sh_gp_a=a.sh_goal)

    def save(self, path: Path) -> None:
        c = asdict(self.constants)
        c["pull_remaining_s"] = {str(k): v for k, v in c["pull_remaining_s"].items()}
        c["pull_prob"] = {str(k): v for k, v in c["pull_prob"].items()}
        path.write_text(json.dumps({
            "meta": self.meta, "features": self.feats,
            "coefs": {k: dict(zip(["intercept", *self.feats[k]], map(float, v), strict=True)) for k, v in self.coefs.items()},
            "constants": c}, indent=2))

    @classmethod
    def load(cls, path: Path) -> GameModelV2:
        d = json.loads(path.read_text())
        c = d["constants"]
        c["pull_remaining_s"] = {int(k): v for k, v in c["pull_remaining_s"].items()}
        c["pull_prob"] = {int(k): v for k, v in c["pull_prob"].items()}
        coefs = {k: np.array(list(v.values())) for k, v in d["coefs"].items()}
        return cls(coefs, d["features"], SimConstants(**c), d["meta"])


def fit(rows: pd.DataFrame, groups: list[str], states: pd.DataFrame, games: pd.DataFrame,
        penalties: pd.DataFrame, version: str, pace_var: float = 0.0) -> GameModelV2:
    feats = features(groups)
    X = design(rows)
    coefs: dict[str, np.ndarray] = {}
    pred_count: dict[str, np.ndarray] = {}
    for comp in ("ev_sog", "pp_sog", "sh_sog", "pen", "ev_goal", "pp_goal", "sh_goal"):
        target, expo, parent = TARGETS[comp]
        base = rows.apply(LEAGUE[comp], axis=1).to_numpy()
        if parent is None:
            e = rows[expo].to_numpy().astype(float)
            m = e > 0
            off = np.log(np.where(m, e, 1.0)) + np.log(base)
        else:
            pc = pred_count[parent]
            m = pc > 0
            off = np.log(np.where(m, pc, 1.0)) + np.log(base)
        b = fit_poisson(X.loc[m, feats[comp]].to_numpy(), rows.loc[m, target].to_numpy().astype(float), off[m])
        coefs[comp] = b
        if parent is None:
            eta = b[0] + X[feats[comp]].to_numpy() @ b[1:]
            pred_count[comp] = np.where(m, np.exp(off + eta), 0.0)
    consts = fit_constants(states, games)
    sm, gm_ = ev_score_effects(states)
    consts.sog_mult_by_lead, consts.gp_mult_by_lead = sm, gm_
    pen_kinds = penalties[penalties.type_code.isin(["MIN", "MAJ", "BEN"])]
    consts.p_major = float((pen_kinds.type_code == "MAJ").mean()) if len(pen_kinds) else 0.0
    consts.pace_var = pace_var
    meta = {"version": version, "groups": groups, "n_rows": int(len(rows)),
            "seasons": sorted(int(s) for s in rows.season.unique())}
    return GameModelV2(coefs, feats, consts, meta)


def ev_score_effects(states: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Even-strength SOG and goal-per-SOG multipliers by the team's own lead (-2..+2), pooled across
    the league and normalised so the time-weighted average is 1.

    A within-game (fixed-effect) version was tried and rejected: goals change the score state, so
    conditioning on each game's totals makes trailing teams look ~2x better at converting (biased).
    """
    s = states[(states.period_bucket != "ot") & (states.home_goalie == 1) & (states.away_goalie == 1) &
               (states.sk_diff_home == 0)]
    rows = pd.concat([
        pd.DataFrame({"lead": s.score_diff_home.clip(-2, 2), "secs": s.secs, "sog": s.home_sog, "g": s.home_goals}),
        pd.DataFrame({"lead": (-s.score_diff_home).clip(-2, 2), "secs": s.secs, "sog": s.away_sog, "g": s.away_goals}),
    ])
    by = rows.groupby("lead")[["secs", "sog", "g"]].sum()
    rate = by.sog / by.secs
    gp = by.g / by.sog
    sm = rate / (by.sog.sum() / by.secs.sum())
    gm = gp / (by.g.sum() / by.sog.sum())
    return [float(sm.loc[i]) for i in range(-2, 3)], [float(gm.loc[i]) for i in range(-2, 3)]


def calibrate(gm: GameModelV2, rows: pd.DataFrame, games: pd.DataFrame, n_games: int = 200,
              n_sims: int = 800) -> dict:
    """Match two simulated league averages to history: empty-net goals (via pull timing) and power
    plays per game (penalties cannot stack in the sim, which loses a few)."""
    from nhlmodel.sim.engine import simulate
    from nhlmodel.sim.params import calibrate_pull_time

    target_en = float((games.home_en_goals + games.away_en_goals).mean())
    gm.constants.pull_time_scale, sim_en = calibrate_pull_time(gm, rows, target_en, n_games, n_sims)
    target_pp = float(rows[rows.valid.fillna(False)].groupby("game_id").pk_starts.sum().mean())
    pairs = [(g[g.is_home == 1].iloc[0], g[g.is_home == 0].iloc[0]) for _, g in rows.groupby("game_id") if len(g) == 2]
    pick = np.random.default_rng(12).choice(len(pairs), size=min(n_games, len(pairs)), replace=False)
    inputs = [gm.game_input(pairs[i][0], pairs[i][1]) for i in pick]
    for _ in range(3):
        rng = np.random.default_rng(6)
        pp = np.mean([float((s.pp_count_h + s.pp_count_a).mean()) for s in
                      (simulate(gi, gm.constants, n=n_sims, rng=rng) for gi in inputs)])
        gm.constants.pen_scale *= target_pp / pp
    # overall scoring level: simulated settlement totals vs actual totals on these (training) games
    gids = [pairs[i][0].game_id for i in pick]
    g = games.set_index("game_id").loc[[x for x in gids if x in set(games.game_id)]]
    target_total = float((g.home_score + g.away_score).mean())
    for _ in range(3):
        rng = np.random.default_rng(8)
        tot = np.mean([float((s.final_h + s.final_a).mean()) for s in
                       (simulate(gi, gm.constants, n=n_sims, rng=rng) for gi in inputs)])
        gm.constants.goal_scale *= (target_total / tot) ** 1.1
    gm.meta.update({"target_en": target_en, "sim_en": sim_en, "target_pp": target_pp, "sim_pp_before_last": pp,
                    "target_total": target_total, "sim_total_before_last": tot})
    return gm.meta
