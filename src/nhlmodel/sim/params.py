"""Fit the game model from history.

Two Poisson regressions (with offsets), fit on team-game rows from the training seasons:

  SOG model:   log E[SOG]   = log(exposure/60min) + log(lg_sog)
                              + b . [off_sog, opp_def_sog, home, b2b, opp_b2b, 3in4, opp_3in4]
  Goal model:  log E[goals] = log E[SOG] + log(lg_xg_per_sog)
                              + d . [off_quality, opp_def_quality, finishing, opp_goalie, home, b2b, opp_b2b]

where off_sog = log(team SOG rating / league), quality = log((xG/SOG) / league), etc.
These replace the PDF's fixed constants (home ice +2.8-3.2%, back-to-back -3.5%): the
coefficients are estimated from data and written to params.json for inspection.

Game-state constants (score effects, goalie pulls, empty net, OT, shootout) come from the
`states` table and the `games` table.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from nhlmodel.features.team import RATED
from nhlmodel.sim.engine import GameInput, SimConstants

SOG_FEATURES = ["off_sog", "opp_def_sog", "is_home", "b2b", "opp_b2b", "three_in_four", "opp_three_in_four"]
GOAL_FEATURES = ["off_q", "opp_def_q", "finishing", "opp_goalie", "is_home", "b2b", "opp_b2b"]


# ----------------------------------------------------------------------------- design matrix

def design(rows: pd.DataFrame) -> pd.DataFrame:
    """rows must hold team ratings r_*, opponent ratings o_*, league averages lg_*, rest flags,
    opponent goalie factor `opp_goalie_factor`. Returns the feature columns."""
    eps = 1e-9
    lg_q = rows["lg_xgf"] / rows["lg_sogf"]
    X = pd.DataFrame(index=rows.index)
    X["off_sog"] = np.log(rows["r_sogf"] / rows["lg_sogf"] + eps)
    X["opp_def_sog"] = np.log(rows["o_soga"] / rows["lg_soga"] + eps)
    X["off_q"] = np.log((rows["r_xgf"] / rows["r_sogf"]) / lg_q + eps)
    X["opp_def_q"] = np.log((rows["o_xga"] / rows["o_soga"]) / lg_q + eps)
    X["finishing"] = np.log((rows["r_gf"] + 1) / (rows["r_xgf"] + 1))
    X["opp_goalie"] = np.log(rows["opp_goalie_factor"].fillna(1.0))
    for c in ("is_home", "b2b", "opp_b2b", "three_in_four", "opp_three_in_four"):
        X[c] = rows[c].astype(float)
    return X


# ----------------------------------------------------------------------------- Poisson GLM

def fit_poisson(X: np.ndarray, y: np.ndarray, offset: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Poisson regression with offset; returns [intercept, coefs...]."""
    Xi = np.column_stack([np.ones(len(X)), X])

    def nll(beta):
        eta = offset + Xi @ beta
        mu = np.exp(eta)
        val = (mu - y * eta).sum() + ridge * (beta[1:] ** 2).sum() * len(y)
        grad = Xi.T @ (mu - y) + np.r_[0, 2 * ridge * beta[1:] * len(y)]
        return val, grad

    b0 = np.zeros(Xi.shape[1])
    b0[0] = np.log(max(y.mean(), 1e-6)) - offset.mean()
    res = minimize(nll, b0, jac=True, method="L-BFGS-B")
    return res.x


class GameModel:
    def __init__(self, sog_beta: np.ndarray, goal_beta: np.ndarray, constants: SimConstants, meta: dict):
        self.sog_beta = np.asarray(sog_beta)
        self.goal_beta = np.asarray(goal_beta)
        self.constants = constants
        self.meta = meta

    @property
    def version(self) -> str:
        return self.meta.get("version", "dev")

    def predict_rates(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected SOG per 60 and P(goal | SOG) for each team-row."""
        X = design(rows)
        lg_sog = rows["lg_sogf"].to_numpy()
        lg_q = (rows["lg_xgf"] / rows["lg_sogf"]).to_numpy()
        sog = np.exp(np.log(lg_sog) + self.sog_beta[0] + X[SOG_FEATURES].to_numpy() @ self.sog_beta[1:])
        gp = np.exp(np.log(lg_q) + self.goal_beta[0] + X[GOAL_FEATURES].to_numpy() @ self.goal_beta[1:])
        return sog, gp

    def game_input(self, home_row: pd.Series, away_row: pd.Series, game_type: int = 2) -> GameInput:
        rows = pd.DataFrame([home_row, away_row]).reset_index(drop=True)
        sog, gp = self.predict_rates(rows)
        return GameInput(sog_h=float(sog[0]), sog_a=float(sog[1]), gp_h=float(gp[0]), gp_a=float(gp[1]),
                         game_type=game_type)

    def save(self, path: Path) -> None:
        c = asdict(self.constants)
        c["pull_remaining_s"] = {str(k): v for k, v in c["pull_remaining_s"].items()}
        c["pull_prob"] = {str(k): v for k, v in c["pull_prob"].items()}
        path.write_text(json.dumps({
            "meta": self.meta,
            "sog": dict(zip(["intercept", *SOG_FEATURES], map(float, self.sog_beta), strict=True)),
            "goal": dict(zip(["intercept", *GOAL_FEATURES], map(float, self.goal_beta), strict=True)),
            "constants": c,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> GameModel:
        d = json.loads(path.read_text())
        c = d["constants"]
        c["pull_remaining_s"] = {int(k): v for k, v in c["pull_remaining_s"].items()}
        c["pull_prob"] = {int(k): v for k, v in c["pull_prob"].items()}
        return cls(np.array(list(d["sog"].values())), np.array(list(d["goal"].values())),
                   SimConstants(**c), d["meta"])


# ----------------------------------------------------------------------------- assembling rows

def attach_opponent(tr: pd.DataFrame) -> pd.DataFrame:
    """Add opponent ratings (o_*) and opponent rest flags to each team-game row."""
    opp = tr[["game_id", "team_id", *[f"r_{m}" for m in RATED], "b2b", "three_in_four"]].rename(
        columns={"team_id": "opp_id", **{f"r_{m}": f"o_{m}" for m in RATED},
                 "b2b": "opp_b2b", "three_in_four": "opp_three_in_four"})
    return tr.merge(opp, on=["game_id", "opp_id"], how="left")


def exposure(states: pd.DataFrame) -> pd.DataFrame:
    """Regulation seconds with both goalies in net, per game."""
    s = states[(states.period_bucket != "ot") & (states.home_goalie == 1) & (states.away_goalie == 1)]
    return s.groupby("game_id")["secs"].sum().rename("exposure_s").reset_index()


def fit_game_model(rows: pd.DataFrame, states: pd.DataFrame, games: pd.DataFrame, version: str) -> GameModel:
    """rows: team-game rows for training seasons with ratings, opponent ratings, league averages,
    rest flags, opp_goalie_factor and observed sogf / gf, plus exposure_s."""
    rows = rows.dropna(subset=["r_sogf", "o_soga", "lg_sogf", "exposure_s"])
    rows = rows[rows.exposure_s > 1800]
    X = design(rows)
    expo = np.log(rows["exposure_s"].to_numpy() / 3600.0)
    off_sog = expo + np.log(rows["lg_sogf"].to_numpy())
    sog_beta = fit_poisson(X[SOG_FEATURES].to_numpy(), rows["sogf"].to_numpy(), off_sog)
    mu_sog = np.exp(off_sog + sog_beta[0] + X[SOG_FEATURES].to_numpy() @ sog_beta[1:])
    lg_q = (rows["lg_xgf"] / rows["lg_sogf"]).to_numpy()
    off_g = np.log(mu_sog) + np.log(lg_q)
    goal_beta = fit_poisson(X[GOAL_FEATURES].to_numpy(), rows["gf"].to_numpy(), off_g)
    consts = fit_constants(states, games)
    meta = {"version": version, "n_rows": int(len(rows)),
            "seasons": sorted(int(s) for s in rows["season"].unique())}
    gm = GameModel(sog_beta, goal_beta, consts, meta)
    target_en = float((games.home_en_goals + games.away_en_goals).mean())
    gm.constants.pull_time_scale, sim_en = calibrate_pull_time(gm, rows, target_en)
    gm.meta.update({"target_en_goals": target_en, "sim_en_goals": sim_en})
    return gm


def calibrate_pull_time(gm: GameModel, rows: pd.DataFrame, target_en: float, n_games: int = 250,
                        n_sims: int = 1000) -> tuple[float, float]:
    """Scale pull times so simulated empty-net goals per game match history.

    Pull times measured from play-by-play run late (a pull is only seen at the next event),
    so the raw medians understate how early teams pull. Bisection on a common scale factor.
    """
    from nhlmodel.sim.engine import simulate

    pairs = []
    for _gid, g in rows.groupby("game_id"):
        if len(g) == 2:
            pairs.append((g[g.is_home == 1].iloc[0], g[g.is_home == 0].iloc[0]))
    rs = np.random.default_rng(11)
    pick = rs.choice(len(pairs), size=min(n_games, len(pairs)), replace=False)
    inputs = [gm.game_input(pairs[i][0], pairs[i][1]) for i in pick]

    def sim_en(scale: float) -> float:
        gm.constants.pull_time_scale = scale
        rng = np.random.default_rng(5)
        tot = 0.0
        for gi in inputs:
            s = simulate(gi, gm.constants, n=n_sims, step=10, rng=rng)
            tot += float((s.en_h + s.en_a).mean())
        return tot / len(inputs)

    lo, hi = 0.5, 4.0
    for _ in range(10):
        mid = (lo + hi) / 2
        if sim_en(mid) < target_en:
            lo = mid
        else:
            hi = mid
    scale = (lo + hi) / 2
    return scale, sim_en(scale)


# ----------------------------------------------------------------------------- game-state constants

def fit_constants(states: pd.DataFrame, games: pd.DataFrame) -> SimConstants:
    st = states[states.game_type.isin([2, 3])]
    reg = st[st.period_bucket != "ot"]
    both_in = reg[(reg.home_goalie == 1) & (reg.away_goalie == 1)]

    # team-perspective rows: lead, secs, sog, goals
    team_rows = pd.concat([
        pd.DataFrame({"lead": both_in.score_diff_home, "secs": both_in.secs,
                      "sog": both_in.home_sog, "goals": both_in.home_goals}),
        pd.DataFrame({"lead": -both_in.score_diff_home, "secs": both_in.secs,
                      "sog": both_in.away_sog, "goals": both_in.away_goals}),
    ])
    team_rows["lead"] = team_rows["lead"].clip(-2, 2)
    by = team_rows.groupby("lead")[["secs", "sog", "goals"]].sum()
    all_rate = team_rows["sog"].sum() / team_rows["secs"].sum()
    all_gp = team_rows["goals"].sum() / team_rows["sog"].sum()
    sog_mult = [float((by.loc[L, "sog"] / by.loc[L, "secs"]) / all_rate) for L in range(-2, 3)]
    gp_mult = [float((by.loc[L, "goals"] / by.loc[L, "sog"]) / all_gp) for L in range(-2, 3)]
    lg_goals_per60 = float(team_rows["goals"].sum() / team_rows["secs"].sum() * 3600)

    # empty net: last 3 minutes of regulation, one goalie pulled by the trailing team
    late = reg[reg.period_bucket == "p3_late"]
    hp = late[(late.home_goalie == 0) & (late.away_goalie == 1) & (late.score_diff_home.between(-2, -1))]
    ap = late[(late.away_goalie == 0) & (late.home_goalie == 1) & (late.score_diff_home.between(1, 2))]
    pulled_secs = hp.secs.sum() + ap.secs.sum()
    en_goals = hp.away_goals.sum() + ap.home_goals.sum()
    trail_sog = hp.home_sog.sum() + ap.away_sog.sum()
    trail_goals = hp.home_goals.sum() + ap.away_goals.sum()
    trailing_rate = (by.loc[-2, "sog"] + by.loc[-1, "sog"]) / (by.loc[-2, "secs"] + by.loc[-1, "secs"])
    trailing_gp = (by.loc[-2, "goals"] + by.loc[-1, "goals"]) / (by.loc[-2, "sog"] + by.loc[-1, "sog"])
    en_rate = float(en_goals / pulled_secs) if pulled_secs > 0 else 0.004
    ea_sog_mult = float((trail_sog / pulled_secs) / trailing_rate) if pulled_secs > 0 else 1.6
    ea_gp_mult = float((trail_goals / max(trail_sog, 1)) / trailing_gp) if trail_sog > 0 else 1.0

    # pull timing and pull probability by deficit
    pull_remaining, pull_prob = {}, {}
    for d in (1, 2):
        times = pd.concat([games.loc[games.home_pull_deficit == d, "home_pull_remaining"],
                           games.loc[games.away_pull_deficit == d, "away_pull_remaining"]]).dropna()
        times = times[times <= 600]
        pull_remaining[d] = float(times.median()) if len(times) else 150.0
        # games where a team trailed by d with both goalies in during the last 3 minutes
        trailed_h = set(late.loc[(late.score_diff_home == -d) & (late.secs > 0), "game_id"])
        trailed_a = set(late.loc[(late.score_diff_home == d) & (late.secs > 0), "game_id"])
        pulled_h = set(games.loc[games.home_pull_deficit == d, "game_id"])
        pulled_a = set(games.loc[games.away_pull_deficit == d, "game_id"])
        n_trail = len(trailed_h) + len(trailed_a)
        n_pull = len(trailed_h & pulled_h) + len(trailed_a & pulled_a)
        pull_prob[d] = float(n_pull / n_trail) if n_trail else 0.5

    # overtime (regular season 3v3)
    ot = st[(st.period_bucket == "ot") & (st.game_type == 2)]
    ot_secs = ot.secs.sum()
    ot_goal_rate = float((ot.home_goals.sum() + ot.away_goals.sum()) / ot_secs) if ot_secs else 0.004
    ot_sog_rate = float((ot.home_sog.sum() + ot.away_sog.sum()) / ot_secs) if ot_secs else 0.03

    so = games[(games.last_period == "SO") & (games.game_type == 2)].copy()
    so_home = float((so.home_score > so.away_score).mean()) if len(so) else 0.5

    pot = st[(st.period_bucket == "ot") & (st.game_type == 3)]
    p5 = reg[(reg.game_type == 3) & (reg.sk_diff_home == 0) & (reg.home_goalie == 1) & (reg.away_goalie == 1)]
    if pot.secs.sum() > 3600 * 20 and p5.secs.sum() > 0:
        r_ot = (pot.home_goals.sum() + pot.away_goals.sum()) / pot.secs.sum()
        r_5 = (p5.home_goals.sum() + p5.away_goals.sum()) / p5.secs.sum()
        playoff_ot_mult = float(r_ot / r_5)
    else:
        playoff_ot_mult = 1.0

    return SimConstants(sog_mult_by_lead=sog_mult, gp_mult_by_lead=gp_mult,
                        pull_remaining_s=pull_remaining, pull_prob=pull_prob, en_goal_rate=en_rate,
                        ea_sog_mult=ea_sog_mult, ea_gp_mult=ea_gp_mult, ot_goal_rate=ot_goal_rate,
                        ot_sog_rate=ot_sog_rate, so_home_win=so_home, playoff_ot_mult=playoff_ot_mult,
                        lg_goals_per60=lg_goals_per60)
