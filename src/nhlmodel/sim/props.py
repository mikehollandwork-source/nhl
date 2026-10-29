"""Player props from the same simulated games (top-down decomposition, PDF section 8).

Inside each simulated game:
- each team goal gets a scorer (by goal share), a primary assist (prob p_a1) and a secondary
  assist (prob p_a2|a1), never the same player twice on one goal;
- saved shots (team SOG - goals) are split by each player's non-goal shot share;
- team blocks ~ Poisson(block ratio x opponent SOG), split by block share;
- power-play goals ~ Binomial(non-EN goals, team PP goal share); each skater collects a PP point on a
  PP goal with prob = share of team PP points x points per PP goal;
- goalie saves = opponent SOG - opponent goals.
Because props come from the same simulations as the game result, joint probabilities
(e.g. team wins AND player scores) are available directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhlmodel.sim.engine import SimResult


@dataclass
class TeamProps:
    players: pd.DataFrame        # rate frame from features.players.player_rates (+ name, team)
    p_a1: float
    p_a2_given_a1: float
    pp_goal_frac: float
    pp_pts_per_goal: float
    block_ratio: float


def _norm(w: np.ndarray) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 1e-9, None)
    return w / w.sum()


def _gumbel_pick(logw: np.ndarray, n: int, rng: np.random.Generator, exclude: list[np.ndarray]) -> np.ndarray:
    """Draw one player index per sim with prob ∝ w, excluding given per-sim indices."""
    scores = logw[None, :] + rng.gumbel(size=(n, len(logw)))
    rows = np.arange(n)
    for ex in exclude:
        valid = ex >= 0
        scores[rows[valid], ex[valid]] = -np.inf
    return scores.argmax(axis=1)


def team_player_sims(goals: np.ndarray, en_goals: np.ndarray, sog: np.ndarray, opp_sog: np.ndarray,
                     tp: TeamProps, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Per-player stat arrays of shape (n_sims, n_players), plus the scorer of the team's first goal."""
    n = len(goals)
    P = len(tp.players)
    g_share = _norm(tp.players["w_goal"].to_numpy())
    a_logw = np.log(_norm(tp.players["w_ast"].to_numpy()))
    out = {k: np.zeros((n, P), np.int16) for k in ("goals", "assists", "sog", "blocks", "ppp")}
    first_scorer = np.full(n, -1)
    cum = np.cumsum(g_share)
    rows = np.arange(n)
    for k in range(int(goals.max()) if n else 0):
        active = goals > k
        scorer = np.minimum(np.searchsorted(cum, rng.random(n)), P - 1)
        if k == 0:
            first_scorer = np.where(active, scorer, -1)
        np.add.at(out["goals"], (rows[active], scorer[active]), 1)
        has_a1 = active & (rng.random(n) < tp.p_a1)
        a1 = _gumbel_pick(a_logw, n, rng, [scorer])
        np.add.at(out["assists"], (rows[has_a1], a1[has_a1]), 1)
        has_a2 = has_a1 & (rng.random(n) < tp.p_a2_given_a1)
        a2 = _gumbel_pick(a_logw, n, rng, [scorer, a1])
        np.add.at(out["assists"], (rows[has_a2], a2[has_a2]), 1)

    # shots on goal: goals are shots; the rest are saved shots split by non-goal shot share
    ng_w = np.clip(tp.players["w_sog"].to_numpy() - tp.players["w_goal"].to_numpy(), 1e-6, None)
    saved = np.maximum(sog - goals, 0)
    out["sog"] = (out["goals"] + rng.multinomial(saved, _norm(ng_w))).astype(np.int16)

    team_blocks = rng.poisson(tp.block_ratio * opp_sog)
    out["blocks"] = rng.multinomial(team_blocks, _norm(tp.players["w_blk"].to_numpy())).astype(np.int16)

    ppg = rng.binomial(np.maximum(goals - en_goals, 0), tp.pp_goal_frac)
    p_pt = np.clip(_norm(tp.players["w_ppp"].to_numpy()) * tp.pp_pts_per_goal, 0, 0.95)
    out["ppp"] = rng.binomial(ppg[:, None], p_pt[None, :]).astype(np.int16)

    out["points"] = out["goals"] + out["assists"]
    out["first_scorer"] = first_scorer
    return out


def simulate_props(sim: SimResult, home: TeamProps, away: TeamProps, rng: np.random.Generator):
    h = team_player_sims(sim.goals_h, sim.en_h, sim.sog_h, sim.sog_a, home, rng)
    a = team_player_sims(sim.goals_a, sim.en_a, sim.sog_a, sim.sog_h, away, rng)
    saves_h_goalie = sim.sog_a - sim.goals_a   # home goalie faces away shots
    saves_a_goalie = sim.sog_h - sim.goals_h
    return h, a, saves_h_goalie, saves_a_goalie


def prop_table(stats: dict[str, np.ndarray], players: pd.DataFrame, first_goal_mask: np.ndarray,
               lines: dict[str, list[float]]) -> pd.DataFrame:
    """Rows: player x market x line with P(over), plus mean and anytime/first goal probabilities."""
    recs = []
    for j, p in enumerate(players.itertuples(index=False)):
        for stat in ("sog", "goals", "assists", "points", "ppp", "blocks"):
            col = stats[stat][:, j]
            mean = float(col.mean())
            for ln in lines.get(stat, []):
                recs.append({"player_id": p.player_id, "player": p.name, "team": p.team, "market": stat,
                             "line": ln, "p_over": float((col > ln).mean()), "mean": mean})
        anytime = float((stats["goals"][:, j] > 0).mean())
        first = float(((stats["first_scorer"] == j) & first_goal_mask).mean())
        recs.append({"player_id": p.player_id, "player": p.name, "team": p.team, "market": "anytime_goal",
                     "line": 0.5, "p_over": anytime, "mean": float(stats["goals"][:, j].mean())})
        recs.append({"player_id": p.player_id, "player": p.name, "team": p.team, "market": "first_goal",
                     "line": None, "p_over": first, "mean": None})
    return pd.DataFrame(recs)


def goalie_table(saves: np.ndarray, goalie_id: int | None, name: str, team: str,
                 lines: list[float]) -> pd.DataFrame:
    mean = float(saves.mean())
    return pd.DataFrame([{"player_id": goalie_id, "player": name, "team": team, "market": "saves",
                          "line": ln, "p_over": float((saves > ln).mean()), "mean": mean} for ln in lines])


# ----------------------------------------------------------------------------- phase 2: by strength

@dataclass
class TeamPropsV2:
    players: pd.DataFrame          # features.players.state_weights output (+ name, team)
    assist: dict                   # state -> (p_a1, p_a2_given_a1)
    block_ratio: float
    player_var: float = 0.0        # per-game "hot night" variance on each player's shot share


def _multinomial_rows(counts: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multinomial with a different probability vector per row (sequential conditional binomials)."""
    n, P = p.shape
    out = np.zeros((n, P), np.int64)
    left = counts.astype(np.int64).copy()
    rest = np.ones(n)
    for j in range(P - 1):
        q = np.clip(p[:, j] / np.maximum(rest, 1e-12), 0, 1)
        out[:, j] = rng.binomial(left, q)
        left -= out[:, j]
        rest -= p[:, j]
    out[:, -1] = left
    return out


def _alloc_goals(n_goals: np.ndarray, w_goal: np.ndarray, w_a1: np.ndarray, w_a2: np.ndarray,
                 p_a1: float, p_a2: float, rng: np.random.Generator, out: dict, pp: bool,
                 first_scorer: np.ndarray | None = None) -> None:
    n = len(n_goals)
    P = len(w_goal)
    rows = np.arange(n)
    cum = np.cumsum(_norm(w_goal))
    l1, l2 = np.log(_norm(w_a1)), np.log(_norm(w_a2))
    for k in range(int(n_goals.max()) if n else 0):
        active = n_goals > k
        scorer = np.minimum(np.searchsorted(cum, rng.random(n)), P - 1)
        if first_scorer is not None and k == 0:
            first_scorer[:] = np.where(active, scorer, first_scorer)
        np.add.at(out["goals"], (rows[active], scorer[active]), 1)
        has1 = active & (rng.random(n) < p_a1)
        a1 = _gumbel_pick(l1, n, rng, [scorer])
        np.add.at(out["assists"], (rows[has1], a1[has1]), 1)
        has2 = has1 & (rng.random(n) < p_a2)
        a2 = _gumbel_pick(l2, n, rng, [scorer, a1])
        np.add.at(out["assists"], (rows[has2], a2[has2]), 1)
        if pp:
            np.add.at(out["ppp"], (rows[active], scorer[active]), 1)
            np.add.at(out["ppp"], (rows[has1], a1[has1]), 1)
            np.add.at(out["ppp"], (rows[has2], a2[has2]), 1)


def team_player_sims_v2(sim: SimResult, side: str, tp: TeamPropsV2, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Allocate a team's simulated events to players by strength state."""
    g = getattr(sim, f"goals_{side[0]}")
    sog = getattr(sim, f"sog_{side[0]}")
    opp_sog = getattr(sim, f"sog_{'a' if side == 'home' else 'h'}")
    ppg = getattr(sim, f"pp_goals_{side[0]}")
    shg = getattr(sim, f"sh_goals_{side[0]}")
    pps = getattr(sim, f"pp_sog_{side[0]}")
    shs = getattr(sim, f"sh_sog_{side[0]}")
    n, P = len(g), len(tp.players)
    pl = tp.players
    out = {k: np.zeros((n, P), np.int16) for k in ("goals", "assists", "sog", "blocks", "ppp")}
    evg = np.maximum(g - ppg - shg, 0)
    k0 = {}
    for st, cnt in (("ev", evg), ("pp", ppg), ("sh", shg)):
        k0[st] = np.full(n, -1)
        if cnt.max() == 0:
            continue
        p1, p2 = tp.assist.get(st, (0.9, 0.7))
        _alloc_goals(cnt, pl[f"w_{st}_goal"].to_numpy(), pl[f"w_{st}_a1"].to_numpy(), pl[f"w_{st}_a2"].to_numpy(),
                     p1, p2, rng, out, pp=(st == "pp"), first_scorer=k0[st])
    # the team's first goal is one of its goals; pick its strength state in proportion to the counts
    u = rng.random(n) * np.maximum(g, 1)
    first = np.where(u < evg, k0["ev"], np.where(u < evg + ppg, k0["pp"], k0["sh"]))
    first = np.where(g > 0, first, -1)
    ev_saved = np.maximum((sog - pps - shs) - evg, 0)
    pp_saved = np.maximum(pps - ppg, 0)
    sh_saved = np.maximum(shs - shg, 0)
    saved = np.zeros((n, P), np.int64)
    hot = None
    if tp.player_var > 0:
        k = 1.0 / tp.player_var
        hot = rng.gamma(k, 1.0 / k, (n, P))
    for st, cnt in (("ev", ev_saved), ("pp", pp_saved), ("sh", sh_saved)):
        ng = np.clip(pl[f"w_{st}_sog"].to_numpy() - pl[f"w_{st}_goal"].to_numpy(), 1e-6, None)
        if hot is None:
            saved += rng.multinomial(cnt, _norm(ng))
        else:
            w = ng[None, :] * hot
            saved += _multinomial_rows(cnt, w / w.sum(axis=1, keepdims=True), rng)
    out["sog"] = (out["goals"] + saved).astype(np.int16)
    team_blocks = rng.poisson(tp.block_ratio * opp_sog)
    out["blocks"] = rng.multinomial(team_blocks, _norm(pl["w_blk"].to_numpy())).astype(np.int16)
    out["points"] = out["goals"] + out["assists"]
    out["first_scorer"] = first
    return out
