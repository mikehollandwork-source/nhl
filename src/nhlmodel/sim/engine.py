"""Vectorized time-step Monte Carlo game simulator.

Changes from the PDF's pseudocode (section 7):
- Simulates shots on goal AND goals (goals are a subset of shots), so shots and saves props can be priced.
- Score effects and goal-per-shot changes by lead are fit from data, not hard-coded.
- Empty-net logic works for whichever team trails (the PDF only handled the home team).
- Regular season OT is 5 min 3-on-3, then a shootout. Playoff OT is sudden-death 5-on-5, no shootout.
- Final score adds 1 goal to the shootout winner, matching how books settle totals and puck lines.

Phase 2 additions (all optional; with pen_h = pen_a = 0 and pace_var = 0 the phase-1 model is
reproduced exactly):
- Penalties arrive as a Poisson process. A minor gives the other team a 2-minute power play that
  ends early on a power-play goal; a share of penalties are 5-minute majors that do not end early.
  Even-strength, power-play and shorthanded shot and goal rates are separate inputs, and every
  goal and shot is tagged by strength (for props).
- A shared per-game pace factor (Gamma, mean 1) scales both teams' rates, adding the small positive
  correlation between the teams' scoring seen in real games.

Every rate here is per 60 minutes in the inputs and converted to per step internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REG_SECONDS = 3600
PP_SECONDS = 120
MAJOR_SECONDS = 300


@dataclass
class SimConstants:
    """Game-state constants fit from history (see sim/params.py and sim/model.py)."""
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
    p_major: float = 0.0                  # share of power plays that are 5-minute majors
    pace_var: float = 0.0                 # variance of the shared per-game pace factor (0 = off)
    pen_scale: float = 1.0                # calibrated so simulated power plays per game match history
    dispersion: float = 1.0               # stretches matchup differences (tested; 1.0 = off)
    goal_scale: float = 1.0               # calibrated so simulated totals match training-season totals
    sog_var: float = 0.0                  # extra per-game variance of a team's SAVED shots (shots/saves
                                          # props); goals are untouched so win/total odds do not change


@dataclass
class GameInput:
    sog_h: float          # expected SOG per 60 at even strength (phase 1: all situations)
    sog_a: float
    gp_h: float           # P(goal | SOG) at even strength for home shots vs the away goalie
    gp_a: float
    game_type: int = 2    # 2 regular season, 3 playoffs
    pen_h: float = 0.0    # power plays the home team gives up per 60 (0 = no penalty model)
    pen_a: float = 0.0
    pp_sog_h: float = 0.0  # home SOG per 60 while on the power play
    pp_sog_a: float = 0.0
    pp_gp_h: float = 0.0
    pp_gp_a: float = 0.0
    sh_sog_h: float = 0.0  # home SOG per 60 while shorthanded
    sh_sog_a: float = 0.0
    sh_gp_h: float = 0.0
    sh_gp_a: float = 0.0


def stretch(g: GameInput, k: float) -> GameInput:
    """Widen (k > 1) or narrow (k < 1) the gap between the two teams while keeping the game's TOTAL
    shots and TOTAL expected goals exactly the same (only the split between the teams changes).
    Applied to even-strength and power-play shots/goals and to penalties.

    Fit on a held-out season: both the phase-1 and phase-2 models were under-confident
    (calibration slope ~1.5), i.e. real team differences were larger than the raw model said."""
    if k == 1.0:
        return g
    from dataclasses import replace

    def split(a: float, b: float) -> tuple[float, float]:
        if a <= 0 or b <= 0:
            return a, b
        wa, wb = a ** k, b ** k
        return (a + b) * wa / (wa + wb), (a + b) * wb / (wa + wb)

    def shots_goals(sh: float, sa: float, gh: float, ga: float) -> tuple[float, float, float, float]:
        if sh <= 0 or sa <= 0:
            return sh, sa, gh, ga
        lh, la = split(sh * gh, sa * ga)
        sh2, sa2 = split(sh, sa)
        return sh2, sa2, lh / sh2, la / sa2

    sog_h, sog_a, gp_h, gp_a = shots_goals(g.sog_h, g.sog_a, g.gp_h, g.gp_a)
    pp_h, pp_a, ppg_h, ppg_a = shots_goals(g.pp_sog_h, g.pp_sog_a, g.pp_gp_h, g.pp_gp_a)
    pen_h, pen_a = split(g.pen_h, g.pen_a)
    return replace(g, sog_h=sog_h, sog_a=sog_a, gp_h=gp_h, gp_a=gp_a, pp_sog_h=pp_h, pp_sog_a=pp_a,
                   pp_gp_h=ppg_h, pp_gp_a=ppg_a, pen_h=pen_h, pen_a=pen_a)


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
    pp_goals_h: np.ndarray | None = None   # goals scored on the power play
    pp_goals_a: np.ndarray | None = None
    sh_goals_h: np.ndarray | None = None   # goals scored shorthanded
    sh_goals_a: np.ndarray | None = None
    pp_sog_h: np.ndarray | None = None     # shots on goal on the power play
    pp_sog_a: np.ndarray | None = None
    sh_sog_h: np.ndarray | None = None
    sh_sog_a: np.ndarray | None = None
    pp_count_h: np.ndarray | None = None   # power plays awarded to home
    pp_count_a: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.goals_h)


def simulate(g: GameInput, c: SimConstants, n: int = 50_000, step: int = 10,
             rng: np.random.Generator | None = None) -> SimResult:
    rng = rng or np.random.default_rng()
    g = stretch(g, c.dispersion)
    if c.goal_scale != 1.0:
        from dataclasses import replace as _replace
        gs = c.goal_scale
        g = _replace(g, gp_h=g.gp_h * gs, gp_a=g.gp_a * gs, pp_gp_h=g.pp_gp_h * gs, pp_gp_a=g.pp_gp_a * gs,
                     sh_gp_h=g.sh_gp_h * gs, sh_gp_a=g.sh_gp_a * gs)
    steps = REG_SECONDS // step
    k = step / 3600.0
    sm = np.asarray(c.sog_mult_by_lead)
    gm = np.asarray(c.gp_mult_by_lead)
    penalties = (g.pen_h > 0) or (g.pen_a > 0)

    vol_h = vol_a = 1.0
    if c.sog_var > 0:
        sh_ = 1.0 / c.sog_var
        vol_h = rng.gamma(sh_, 1.0 / sh_, n)
        vol_a = rng.gamma(sh_, 1.0 / sh_, n)
    pace = np.ones(n)
    if c.pace_var > 0:
        shape = 1.0 / c.pace_var
        pace = rng.gamma(shape, 1.0 / shape, n)

    # Decide in advance whether each team would pull its goalie when trailing by 1 / 2.
    pull_h = {d: rng.random(n) < c.pull_prob.get(d, 0) for d in (1, 2)}
    pull_a = {d: rng.random(n) < c.pull_prob.get(d, 0) for d in (1, 2)}

    z = lambda dt=np.int32: np.zeros(n, dt)  # noqa: E731
    gh, ga, sh, sa, enh, ena = z(), z(), z(), z(), z(), z()
    ppg_h, ppg_a, shg_h, shg_a = z(), z(), z(), z()
    pps_h, pps_a, shs_h, shs_a = z(), z(), z(), z()
    ppc_h, ppc_a = z(), z()
    first = z(np.int8)
    h_pulled = np.zeros(n, bool)
    a_pulled = np.zeros(n, bool)
    pk_h = np.zeros(n)   # seconds of shorthanded time left for home (home took a penalty)
    pk_a = np.zeros(n)
    maj_h = np.zeros(n, bool)
    maj_a = np.zeros(n, bool)

    for i in range(steps):
        remaining = REG_SECONDS - i * step
        lead_h = np.clip(gh - ga, -2, 2)
        idx_h = lead_h + 2
        idx_a = -lead_h + 2

        # penalties (only while that team is not already shorthanded)
        if penalties:
            new_h = (pk_h <= 0) & (rng.random(n) < g.pen_h * c.pen_scale * k * pace)
            new_a = (pk_a <= 0) & (rng.random(n) < g.pen_a * c.pen_scale * k * pace)
            if c.p_major > 0:
                mj_h = new_h & (rng.random(n) < c.p_major)
                mj_a = new_a & (rng.random(n) < c.p_major)
            else:
                mj_h = mj_a = np.zeros(n, bool)
            pk_h = np.where(new_h, np.where(mj_h, MAJOR_SECONDS, PP_SECONDS), pk_h)
            pk_a = np.where(new_a, np.where(mj_a, MAJOR_SECONDS, PP_SECONDS), pk_a)
            maj_h = np.where(new_h, mj_h, maj_h)
            maj_a = np.where(new_a, mj_a, maj_a)
            ppc_a += new_h
            ppc_h += new_a
            home_pp = (pk_a > 0) & (pk_h <= 0)
            away_pp = (pk_h > 0) & (pk_a <= 0)
        else:
            home_pp = away_pp = np.zeros(n, bool)

        # goalie pulls: trailing team, late; goalie stays out while still trailing (by up to 3)
        for d in (1, 2):
            if remaining <= c.pull_remaining_s.get(d, 0) * c.pull_time_scale:
                h_pulled |= ((ga - gh) == d) & pull_h[d]
                a_pulled |= ((gh - ga) == d) & pull_a[d]
        h_pulled &= ((ga - gh) >= 1) & ((ga - gh) <= 3)
        a_pulled &= ((gh - ga) >= 1) & ((gh - ga) <= 3)

        # shot rates by strength (score effects apply at even strength)
        sog_h = np.where(home_pp, g.pp_sog_h, np.where(away_pp, g.sh_sog_h, g.sog_h * sm[idx_h])) * k * pace
        sog_a = np.where(away_pp, g.pp_sog_a, np.where(home_pp, g.sh_sog_a, g.sog_a * sm[idx_a])) * k * pace
        gp_h = np.where(home_pp, g.pp_gp_h, np.where(away_pp, g.sh_gp_h, g.gp_h * gm[idx_h]))
        gp_a = np.where(away_pp, g.pp_gp_a, np.where(home_pp, g.sh_gp_a, g.gp_a * gm[idx_a]))
        sog_h = np.where(h_pulled, g.sog_h * sm[idx_h] * k * pace * c.ea_sog_mult, sog_h)
        sog_a = np.where(a_pulled, g.sog_a * sm[idx_a] * k * pace * c.ea_sog_mult, sog_a)
        gp_h = np.clip(np.where(h_pulled, g.gp_h * gm[idx_h] * c.ea_gp_mult, gp_h), 0, 0.95)
        gp_a = np.clip(np.where(a_pulled, g.gp_a * gm[idx_a] * c.ea_gp_mult, gp_a), 0, 0.95)

        # leading team facing an empty net scores at the EN rate instead of its normal rate
        en_mu_h = np.where(a_pulled, c.en_goal_rate * step, 0.0)
        en_mu_a = np.where(h_pulled, c.en_goal_rate * step, 0.0)
        sog_h = np.where(a_pulled, 0.0, sog_h)
        sog_a = np.where(h_pulled, 0.0, sog_a)

        g_h = rng.poisson(sog_h * gp_h)
        g_a = rng.poisson(sog_a * gp_a)
        s_h = rng.poisson(sog_h * (1 - gp_h) * vol_h)
        s_a = rng.poisson(sog_a * (1 - gp_a) * vol_a)
        e_h = rng.poisson(en_mu_h) if a_pulled.any() else 0
        e_a = rng.poisson(en_mu_a) if h_pulled.any() else 0

        if penalties:
            # a minor ends at the first power-play goal; a second goal in the same step is even strength
            ppg_h += np.where(home_pp, np.where(maj_a, g_h, np.minimum(g_h, 1)), 0)
            ppg_a += np.where(away_pp, np.where(maj_h, g_a, np.minimum(g_a, 1)), 0)
            shg_h += np.where(away_pp, g_h, 0)
            shg_a += np.where(home_pp, g_a, 0)
            pps_h += np.where(home_pp, g_h + s_h, 0)
            pps_a += np.where(away_pp, g_a + s_a, 0)
            shs_h += np.where(away_pp, g_h + s_h, 0)
            shs_a += np.where(home_pp, g_a + s_a, 0)
            # a power-play goal ends a minor penalty
            pk_a = np.where(home_pp & (g_h > 0) & ~maj_a, 0.0, pk_a)
            pk_h = np.where(away_pp & (g_a > 0) & ~maj_h, 0.0, pk_h)
            pk_h = np.maximum(pk_h - step, 0.0)
            pk_a = np.maximum(pk_a - step, 0.0)

        new_h_g = g_h + e_h
        new_a_g = g_a + e_a
        nf = (first == 0) & ((new_h_g > 0) | (new_a_g > 0))
        if nf.any():
            both = nf & (new_h_g > 0) & (new_a_g > 0)
            coin = rng.random(n) < 0.5
            first = np.where(nf & (new_h_g > 0) & ~(both & coin), 1, first)
            first = np.where(nf & (new_a_g > 0) & ((new_h_g == 0) | (both & coin)), -1, first)

        gh += new_h_g
        ga += new_a_g
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
        kt = int(tied.sum())
        if g.game_type == 3:
            rh = g.sog_h / 3600.0 * g.gp_h * c.playoff_ot_mult
            ra = g.sog_a / 3600.0 * g.gp_a * c.playoff_ot_mult
            t = rng.exponential(1.0 / (rh + ra), kt)
            hw = rng.random(kt) < rh / (rh + ra)
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
            share_h = (g.sog_h * g.gp_h) / (g.sog_h * g.gp_h + g.sog_a * g.gp_a)
            strength = (g.sog_h * g.gp_h + g.sog_a * g.gp_a) / 2.0
            r_tot = c.ot_goal_rate * strength / c.lg_goals_per60
            t = rng.exponential(1.0 / r_tot, kt)
            scored = t < 300
            t_play = np.minimum(t, 300)
            hw_goal = rng.random(kt) < share_h
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
            so_hw = rng.random(kt) < c.so_home_win
            hw = np.where(so, so_hw, add_h)
            home_win[tied] = hw
            so_full = np.zeros(n, bool)
            so_full[tied] = so
            went_so = so_full
            f = first[tied]
            first[tied] = np.where((f == 0) & add_h, 1, np.where((f == 0) & add_a, -1, f))

    final_h = gh + (went_so & home_win)
    final_a = ga + (went_so & ~home_win)
    extra = {}
    if penalties:
        extra = dict(pp_goals_h=ppg_h, pp_goals_a=ppg_a, sh_goals_h=shg_h, sh_goals_a=shg_a,
                     pp_sog_h=pps_h, pp_sog_a=pps_a, sh_sog_h=shs_h, sh_sog_a=shs_a,
                     pp_count_h=ppc_h, pp_count_a=ppc_a)
    return SimResult(goals_h=gh, goals_a=ga, reg_h=reg_h, reg_a=reg_a, sog_h=sh, sog_a=sa,
                     en_h=enh, en_a=ena, went_ot=went_ot, went_so=went_so, home_win=home_win,
                     first_goal=first, final_h=final_h, final_a=final_a, **extra)
