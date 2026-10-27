"""Vectorized time-step Monte Carlo game simulator.

Changes from the PDF's pseudocode (section 7):
- Simulates shots on goal AND goals (goals are a subset of shots), so shots and saves props can be priced.
- Score effects and goal-per-shot changes by lead are fit from data, not hard-coded.
- Empty-net logic works for whichever team trails (the PDF only handled the home team).
- Regular season OT is 5 min 3-on-3, then a shootout. Playoff OT is sudden-death 5-on-5, no shootout.
- Final score adds 1 goal to the shootout winner, matching how books settle totals and puck lines.

Every rate here is "per second". All arrays are shape (n_sims,).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REG_SECONDS = 3600


@dataclass
class SimConstants:
    """Game-state constants fit from history (see sim/params.py)."""
    # multiplier on a team's SOG rate / goal-per-SOG by its own lead, index lead+2 for lead in -2..2
    sog_mult_by_lead: list[float] = field(default_factory=lambda: [1.12, 1.06, 1.0, 0.93, 0.88])
    gp_mult_by_lead: list[float] = field(default_factory=lambda: [0.97, 0.99, 1.0, 1.02, 1.04])
    pull_remaining_s: dict[int, float] = field(default_factory=lambda: {1: 150.0, 2: 200.0})
    pull_prob: dict[int, float] = field(default_factory=lambda: {1: 0.9, 2: 0.6})
    en_goal_rate: float = 0.0040          # leading team's goals/sec into an empty net
    ea_sog_mult: float = 1.6              # trailing team's SOG rate multiplier with goalie pulled
    ea_gp_mult: float = 1.0               # trailing team's goal-per-SOG multiplier with goalie pulled
    ot_goal_rate: float = 0.0040          # total OT (3v3) goals/sec, both teams, average teams
    ot_sog_rate: float = 0.030            # total OT SOG/sec, both teams
    so_home_win: float = 0.5
    playoff_ot_mult: float = 1.0          # playoff OT goal-rate multiplier vs regulation 5v5
    lg_goals_per60: float = 2.9           # league goals per team per 60 (goalies in), scales OT rate
    pull_time_scale: float = 1.0          # calibrated so simulated empty-net goals match history


@dataclass
class GameInput:
    sog_h: float          # expected regulation SOG (goalies in, average game state) per 60
    sog_a: float
    gp_h: float           # P(goal | SOG) for home shots vs the away goalie
    gp_a: float
    game_type: int = 2    # 2 regular season, 3 playoffs


@dataclass
class SimResult:
    goals_h: np.ndarray      # final goals incl. OT (not the shootout bonus)
    goals_a: np.ndarray
    reg_h: np.ndarray        # regulation goals
    reg_a: np.ndarray
    sog_h: np.ndarray        # shots on goal incl. OT and empty-net goals
    sog_a: np.ndarray
    en_h: np.ndarray         # empty-net goals scored by home
    en_a: np.ndarray
    went_ot: np.ndarray
    went_so: np.ndarray
    home_win: np.ndarray     # includes OT/SO
    first_goal: np.ndarray   # 1 home, -1 away, 0 none
    final_h: np.ndarray      # settlement score (shootout winner +1)
    final_a: np.ndarray

    @property
    def n(self) -> int:
        return len(self.goals_h)


def simulate(g: GameInput, c: SimConstants, n: int = 50_000, step: int = 10,
             rng: np.random.Generator | None = None) -> SimResult:
    rng = rng or np.random.default_rng()
    steps = REG_SECONDS // step
    base_sog_h = g.sog_h / 3600.0 * step
    base_sog_a = g.sog_a / 3600.0 * step
    sm = np.asarray(c.sog_mult_by_lead)
    gm = np.asarray(c.gp_mult_by_lead)

    # Decide in advance whether each team would pull its goalie when trailing by 1 / 2,
    # and when (seconds remaining). One draw per sim keeps the behavior consistent within a game.
    pull_h = {d: rng.random(n) < c.pull_prob.get(d, 0) for d in (1, 2)}
    pull_a = {d: rng.random(n) < c.pull_prob.get(d, 0) for d in (1, 2)}

    gh = np.zeros(n, np.int32)
    ga = np.zeros(n, np.int32)
    sh = np.zeros(n, np.int32)
    sa = np.zeros(n, np.int32)
    enh = np.zeros(n, np.int32)
    ena = np.zeros(n, np.int32)
    first = np.zeros(n, np.int8)
    h_pulled = np.zeros(n, bool)
    a_pulled = np.zeros(n, bool)

    for i in range(steps):
        remaining = REG_SECONDS - i * step
        lead_h = np.clip(gh - ga, -2, 2)
        idx_h = lead_h + 2
        idx_a = -lead_h + 2

        # Goalie pulls: a trailing team pulls once its deficit-specific time is reached, and the
        # goalie stays out while the team still trails (by up to 3). A tie brings the goalie back.
        for d in (1, 2):
            t_pull = c.pull_remaining_s.get(d, 0) * c.pull_time_scale
            if remaining <= t_pull:
                h_pulled |= ((ga - gh) == d) & pull_h[d]
                a_pulled |= ((gh - ga) == d) & pull_a[d]
        h_pulled &= ((ga - gh) >= 1) & ((ga - gh) <= 3)
        a_pulled &= ((gh - ga) >= 1) & ((gh - ga) <= 3)

        mu_sog_h = base_sog_h * sm[idx_h] * np.where(h_pulled, c.ea_sog_mult, 1.0)
        mu_sog_a = base_sog_a * sm[idx_a] * np.where(a_pulled, c.ea_sog_mult, 1.0)
        p_h = np.clip(g.gp_h * gm[idx_h] * np.where(h_pulled, c.ea_gp_mult, 1.0), 0, 0.95)
        p_a = np.clip(g.gp_a * gm[idx_a] * np.where(a_pulled, c.ea_gp_mult, 1.0), 0, 0.95)

        # Leading team facing an empty net scores at the EN rate instead of its normal rate.
        en_mu_h = np.where(a_pulled, c.en_goal_rate * step, 0.0)
        en_mu_a = np.where(h_pulled, c.en_goal_rate * step, 0.0)
        mu_sog_h = np.where(a_pulled, 0.0, mu_sog_h)
        mu_sog_a = np.where(h_pulled, 0.0, mu_sog_a)

        # Goals and saved shots as independent Poissons = Poisson shots thinned by P(goal).
        g_h = rng.poisson(mu_sog_h * p_h)
        g_a = rng.poisson(mu_sog_a * p_a)
        s_h = rng.poisson(mu_sog_h * (1 - p_h))
        s_a = rng.poisson(mu_sog_a * (1 - p_a))
        e_h = rng.poisson(en_mu_h) if a_pulled.any() else 0
        e_a = rng.poisson(en_mu_a) if h_pulled.any() else 0

        new_h = g_h + e_h
        new_a = g_a + e_a
        # first goal of the game (ties inside one 10 s step are split at random)
        nf = (first == 0) & ((new_h > 0) | (new_a > 0))
        if nf.any():
            both = nf & (new_h > 0) & (new_a > 0)
            coin = rng.random(n) < 0.5
            first = np.where(nf & (new_h > 0) & ~(both & coin), 1, first)
            first = np.where(nf & (new_a > 0) & ((new_h == 0) | (both & coin)), -1, first)

        gh += new_h
        ga += new_a
        sh += g_h + s_h + e_h
        sa += g_a + s_a + e_a
        enh += e_h
        ena += e_a

    reg_h = gh.copy()
    reg_a = ga.copy()
    tied = gh == ga
    went_ot = tied.copy()
    went_so = np.zeros(n, bool)
    home_win = gh > ga

    if tied.any():
        k = int(tied.sum())
        if g.game_type == 3:
            # playoff sudden death at 5v5 regulation rates until a goal
            rh = g.sog_h / 3600.0 * g.gp_h * c.playoff_ot_mult
            ra = g.sog_a / 3600.0 * g.gp_a * c.playoff_ot_mult
            t = rng.exponential(1.0 / (rh + ra), k)
            hw = rng.random(k) < rh / (rh + ra)
            ot_sog_h = rng.poisson(g.sog_h / 3600.0 * t)
            ot_sog_a = rng.poisson(g.sog_a / 3600.0 * t)
            gh[tied] += hw
            ga[tied] += ~hw
            sh[tied] += ot_sog_h + hw
            sa[tied] += ot_sog_a + ~hw
            home_win[tied] = hw
            f = first[tied]
            first[tied] = np.where(f == 0, np.where(hw, 1, -1), f)
        else:
            # 3v3: total rate scaled by these teams' combined scoring vs league; split by relative strength
            share_h = (g.sog_h * g.gp_h) / (g.sog_h * g.gp_h + g.sog_a * g.gp_a)
            # scale the league OT rate by these two teams' combined scoring vs league average
            strength = (g.sog_h * g.gp_h + g.sog_a * g.gp_a) / 2.0
            r_tot = c.ot_goal_rate * strength / c.lg_goals_per60
            t = rng.exponential(1.0 / r_tot, k)
            scored = t < 300
            t_play = np.minimum(t, 300)
            hw_goal = rng.random(k) < share_h
            sog_share_h = g.sog_h / (g.sog_h + g.sog_a)
            ot_sog = rng.poisson(c.ot_sog_rate * t_play)
            ot_sog_h = rng.binomial(ot_sog, sog_share_h)
            ot_sog_a = ot_sog - ot_sog_h
            add_h = scored & hw_goal
            add_a = scored & ~hw_goal
            gh[tied] += add_h
            ga[tied] += add_a
            sh[tied] += ot_sog_h + add_h
            sa[tied] += ot_sog_a + add_a
            so = ~scored
            so_hw = rng.random(k) < c.so_home_win
            hw = np.where(so, so_hw, add_h)
            home_win[tied] = hw
            so_full = np.zeros(n, bool)
            so_full[tied] = so
            went_so = so_full
            # first goal: if regulation was 0-0, the OT winner scored first
            f = first[tied]
            first[tied] = np.where((f == 0) & add_h, 1, np.where((f == 0) & add_a, -1, f))

    final_h = gh + (went_so & home_win)
    final_a = ga + (went_so & ~home_win)
    return SimResult(goals_h=gh, goals_a=ga, reg_h=reg_h, reg_a=reg_a, sog_h=sh, sog_a=sa,
                     en_h=enh, en_a=ena, went_ot=went_ot, went_so=went_so, home_win=home_win,
                     first_goal=first, final_h=final_h, final_a=final_a)
