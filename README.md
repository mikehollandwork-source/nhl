# NHL projection model

Projects NHL game results and player props every day, compares them with sportsbook lines,
logs every disagreement, grades the disagreements overnight, and posts everything to Discord
(Telegram optional).
A separate experiment tries to reverse-engineer how books set their lines.

This is a personal projection and tracking tool, **not a sportsbook**.

## What runs when (GitHub Actions, times UTC)

| Job | Schedule | Posts to Discord |
|---|---|---|
| `morning` | 15:43 daily | Game cards **only for games with a Polymarket/Kalshi line**, each with model vs the best buy price (as American odds) for ML, puck line and total; flagged disagreements; **props where the model is above Kalshi's price**; full CSV |
| `pregame` | 22:11 and 01:23 daily | Only changes: a new starting goalie, a new disagreement, or new Kalshi props that opened after the morning post. Also snapshots lines for closing-line value |
| `nightly` | 09:17 daily | Results recap: model vs book accuracy, flagged-pick record, closing-line value, props vs Kalshi record |
| `warmup` | every 15 min, 16:00-03:45 | Only when a game reaches warmups and something material changed: referees, scratches, confirmed goalies, re-run projection |
| `weekly` | Mon 08:37 | Retrain (phase 1 + phase 2) + reverse-engineering report |
| `ablation` | 1st of month 07:41 | Re-tests which input groups help, then retrains |
| `ci` | every push | Lint + tests |
| `bootstrap` | manual only | Rebuilds the `data` branch from scratch |

Every job can also be run by hand from the Actions tab (`Run workflow`), optionally for another
date or in dry-run mode (prints instead of posting).

**Timing:** GitHub's own schedule can start jobs hours late, so the jobs are also started by an external
timer (cron-job.org calling GitHub's "run workflow" API). The morning, nightly and weekly jobs post
only once per day/week no matter how many times they are started (marker files in `data/runs/`);
tick `force` when running by hand to post again.

## How the model works

1. **Data**: free NHL API (play-by-play, box scores, schedule). Starting goalies from DailyFaceoff
   (unofficial scrape; if it fails we guess the starter from recent usage and back-to-backs).
2. **Expected goals (xG)**: XGBoost on shot distance, angle, shot type, rebound, rush, strength,
   score state; isotonic calibration on a held-out season.
3. **Ratings** (point-in-time, never use future games): team shots for/against and shot quality,
   recency-weighted and shrunk toward league average; goalie goals-allowed vs xG, shrunk toward
   career then league average.
4. **Game model**: two Poisson regressions (shots on goal, then goals per shot) whose coefficients
   (home ice, back-to-back, 3-in-4, goalie, finishing) are fit from data, not hard-coded.
5. **Simulation**: 50,000 simulated games per matchup in 10-second steps, with score effects,
   goalie pulls for either team, empty-net goals, 3-on-3 OT + shootout (playoffs: sudden-death OT).
   All game-state constants are measured from play-by-play; pull timing is calibrated so simulated
   empty-net goals match history.
6. **Props**: inside each simulated game, goals, assists, shots, blocks and power-play points are
   handed out to players by their shrunk rates; goalie saves = opponent shots - goals.
   Markets: SOG, goals, assists, points, PPP, blocks, anytime/first goal, saves.

### Backtest (held-out 2025-26 season, 1,394 games)

Trained on 2021-22 to 2024-25, tested on 2025-26 (never seen in training):

| | Model | Simple Poisson | Home-ice only |
|---|---|---|---|
| Moneyline log-loss (lower = better) | **0.6797** | 0.6886 | 0.6933 |
| Moneyline Brier | **0.2434** | 0.2477 | 0.2501 |

| Check | Model | Actual |
|---|---|---|
| Goals per game | 6.22 | 6.23 |
| Shots on goal per game | 55.9 | 55.7 |
| Games reaching OT | 20.0% | 25.0% (training seasons 20.7-23.0%) |

The gate (beat both baselines) passes. Overtime is under-projected by about 1.5 points
structurally, plus 2025-26 was an unusually tie-heavy season. Not yet compared with book
closing lines (historical odds are paid-only); my unverified estimate is that NHL closing
moneylines score roughly 0.675-0.68, i.e. similar to or slightly better than this model.
The disagreement tracker will measure this directly from the first day lines are collected. Full report: `models/train_report_backtest.json` on the data branch.

## Phase 2: every input that measurably helps

Phase 2 added every public input that could plausibly help, then **kept only what improved
held-out predictions**. Adding inputs that don't help makes a model worse, so each group had
to pass a gate.

**New data** (all from free NHL feeds; 2021-22 onward):
- Shift charts → who was on the ice for every second ("stints"). Ice time rebuilt from them
  matches the NHL's official figures exactly on the power play and penalty kill, and within
  about 1% at even strength.
- Penalties from play-by-play.
- Referees, linesmen, scratches and head coaches for every game.
- Venue time zones.
- NHL EDGE tracking: skating speed bursts, zone time, shot speed (season level, per team and per skater).
- DailyFaceoff lines, PP/PK units, injuries and game-time decisions (live only).

**New model structure** (`sim/model.py`, `sim/engine.py`):
- Separate even-strength, power-play and shorthanded shot and goal rates, plus a penalty model.
- The simulator plays out penalties: 2-minute minors end on a power-play goal; 5-minute majors
  do not.
- Every simulated goal and shot is tagged by strength, so power-play points and goals come from
  each player's actual power-play role.

**Candidate input groups:**

| Group | What it is |
|---|---|
| lineup | Player impact ratings (ridge regression on 5v5 stints). Tonight's lineup vs the team's usual one; plus finishing skill and PP-unit strength. Three versions were tested. |
| refs | Referee crew penalty tendency (shrunk). Referees are only posted at warmups. |
| goalie_detail | Goalie results split by shot danger, rebound control, workload, back-to-backs |
| travel | Time-zone changes, games in the last 7 days, long road trips |
| edge | Prior-season EDGE skating speed, offensive-zone time, hard shots |
| edge_player | Per-skater prior-season EDGE (top speed, bursts, zone time), tonight's lineup vs the usual one |
| pace | Shared per-game pace factor (small positive correlation between the teams' scoring) |

**Selection protocol**, so the reported test result is not inflated by picking inputs:
1. Fit on 2021-24 and pick groups on 2024-25.
2. Refit on 2021-25 and test the final model once on 2025-26.

A group is kept only if the 90% bootstrap interval of its log-loss change is entirely below
zero on moneyline or totals, without the other getting worse. Results:
`models/ablation_report.json` on the data branch.

**Calibration finding (dispersion):**
- On 2024-25 both models were under-confident: the calibration slope was about 1.5, where 1.0 is
  perfect.
- A "dispersion" stretch fitted on that season did not hold up on 2025-26, so it is off
  (`--dispersion` re-tests it).

**Selection, choices made on 2024-25 only** (log-loss change vs the model before it; negative is
better; ⚠ = the 90% interval excludes 0):

| Candidate | Moneyline | Totals | Kept? |
|---|---|---|---|
| Special-teams base vs phase 1 | +0.0046 ⚠ worse | -0.0005 | structure kept (needed for PP props); see test |
| Lineup, basic (unrated players = average) | -0.0010 | -0.0004 | no |
| **Lineup, replacement level** | **-0.0022 ⚠** | -0.0003 | **yes** |
| Lineup, full (+ finishing, + PP unit) | -0.0009 | -0.0005 | no |
| Referees | +0.0009 | +0.0011 ⚠ worse | no |
| Goalie detail | +0.0013 | -0.0001 | no |
| Travel | +0.0013 | +0.0004 | no |
| EDGE speed / zone time | +0.0006 | -0.0002 | no |
| Faceoffs (overall, by zone, PP), zone starts, giveaways, takeaways | +0.0003 | -0.0006 | no |
| Hits | +0.0006 | -0.0002 | no |
| Per-player EDGE speed / bursts / zone time (lineup vs usual) | +0.0007 | +0.0000 | no |
| Shared pace factor | +0.0016 ⚠ worse | -0.0007 | no |
| Dispersion stretch (fit 1.45-1.8 on 2024-25) | helped 2024-25, hurt 2025-26 | | no (did not transfer) |
| Rolling totals-level correction | | -0.0003 | no |

**Final test, untouched 2025-26 season** (1,382 games; model refit on 2021-25):

| Model | Moneyline log-loss | Totals log-loss | Goals per game |
|---|---|---|---|
| Phase 1 | 0.6799 | 0.6829 | 6.22 |
| **Phase 2 (special teams + lineup)** | **0.6762** | 0.6851 | 6.04 (actual 6.23) |

- **Moneyline:** -0.0037 (90% interval -0.0062 to -0.0012), the clearest gain.
- **Totals:** +0.0022, not significant. No version clearly beats a constant guess on totals.

**Props backtest** (300 games of 2025-26, about 10,800 player-lines per market):
- Every market beats the no-skill baseline.
- Shots on goal: Brier 0.146 vs 0.171 at 2.5, 0.209 vs 0.244 at 1.5.
- Points over 0.5: 0.205 vs 0.226.
- Saves: 0.198 vs 0.209 at 28.5.
- Measured player and team shot-volume variance fixed the thin high tails: saves over 28.5 predicted 28.2% vs 29.7% actual.

**Props calibration** (`pipelines/props_calibration.py`, Oct 2026). Player rates were pulled too hard toward
the position average, so stars were under-predicted and depth players over-predicted. Chosen on 400
games of 2024-25, tested once on 400 games of 2025-26 (goals, points, assists and shots lines):

| Settings | Test log-loss | Stars: points over 0.5 (pred / actual) | Others (pred / actual) |
|---|---|---|---|
| Old: EV prior 300 min, PP 60, half-life 30 games | 0.3423 | 55.2% / 63.7% | 32.4% / 31.0% |
| **New: EV prior 75 min, PP 30, half-life 60 games** | **0.3416** | 57.0% / 63.7% | 31.9% / 31.0% |

- Change −0.0007, 90% interval −0.0010 to −0.0004: kept.
- An extra per-market logit recalibration did not help on the test season, so it is off.
- Stars ("star" = 0.85+ points per game) are still about 7 points under-predicted, so props on them
  will often show the model below the market. Props are only posted when the model is above the
  price, so this mostly means fewer star props get posted, not worse ones.

The faceoff, giveaway/takeaway, zone-start and hit stats are the official NHL team stats, rebuilt
game by game from play-by-play so the backtest uses only earlier games. They match the NHL's own
team pages: faceoff % exactly for all 32 teams in 2025-26; hits and takeaways exactly; giveaways
within 3%. Faceoffs and hits were tested after the main run, on top of the lineup model, with the
same protocol.

**What did not help**, and why that is plausible:
- **Referees:** they shift power-play counts, but only by a few hundredths of a goal.
- **EDGE speed:** already reflected in shots and xG, at team level and for lineup changes.
- **Travel and goalie detail:** too small or too noisy at this sample size.
- **Faceoffs, giveaways, takeaways, hits:** these show up later in shots and chances, which the model
  already measures directly; hits are also recorded inconsistently from rink to rink.

They stay in the data store and are re-tested monthly by the `ablation` job.

**Live additions:**
- The morning cards list who is out, game-time decisions and PP1.
- A `warmup` job runs every 15 minutes during game windows. When a game reaches warmups it
  pulls referees, scratches and confirmed starters, re-runs that game, and posts a "final" update
  only if something material changed.
- The nightly recap adds moneyline accuracy that needs no odds.

## Disagreement tracking

For every game line the books offer (moneyline, puck line, total):

- `book_prob` = median across books of each book's de-vigged probability
- `edge` = model probability - book probability, `flagged` when edge >= threshold
  (3 pts moneyline, 4 pts puck line/totals; see `config/settings.yaml`)
- Every row is stored, flagged or not. The model is **never blended** with the market in these numbers.

Nightly grading answers: when we disagree, whose probability was closer (log-loss)? Did the
closing line move toward us (CLV)? What is the flat-bet record? Broken down by market, edge size
and goalie-confirmation status.

**Lines**: all prices come from Polymarket and Kalshi (free, no API keys). Exchange history is not
collected before the day collection starts, so all market comparisons start from then.

## Line reverse-engineering (`market/tells.py`)

Assumes books price the way the uploaded spec describes (scoring rates -> simulated prices ->
margin). Tests, with Benjamini-Hochberg multiple-testing correction, and only reported after
600+ games with lines:

1. Margin shape: which de-vig method fits each book; do underdogs carry more margin?
2. Implied scoring rates (solved from each book's ML + total) vs our model.
3. Internal consistency: is a book's puck line consistent with its own ML + total?
4. Goalie news: do lines move more after a starter change?
5. Team shading: do some teams consistently price differently from the model?

Expect most of these to find nothing. Internal-consistency gaps and slow goalie adjustments are
the likeliest places for a real tell.

## Setup

1. **Discord**: in your server, open the channel's settings -> Integrations -> Webhooks -> New
   Webhook -> Copy Webhook URL.
2. **GitHub secret** (Settings -> Secrets and variables -> Actions): `DISCORD_WEBHOOK_URL`. That is
   the only one; no odds API key is needed. Optional: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` to
   also post to Telegram.
3. The `data` branch holds all stored data and trained models. If it is ever missing, run the
   `bootstrap` workflow (type `rebuild`).
4. Run `morning` once by hand with `dry_run` checked, then without it, to see the Discord output.

## Line sources (all free, no keys)

| What | Source | Notes |
|---|---|---|
| Game lines | Polymarket + Kalshi | Moneyline (includes OT and shootout, same as the model), puck lines, totals. Games without either are not posted |
| Player props | Kalshi | Goals, points, assists (N+ = over N-0.5). Posted only when the model is above the price |
| Shots, saves, blocks, PP points | none free | Projections stay in the CSV only |

Every price is the **buy price** (the ask, what you would actually pay), converted from cents to
American odds: 38c -> +163, 62c -> -163. Edges are measured against that price, so a flagged pick
beats a price you can actually get. Markets wider than 10 cents between bid and ask are skipped.
DraftKings is not used: its site blocks cloud servers, and the free odds API that carries it needs a key.

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
python -m nhlmodel.pipelines.backfill --all            # ~10 min, ~15 MB
python -m nhlmodel.pipelines.train --mode backtest     # out-of-sample report
python -m nhlmodel.pipelines.train --mode production   # models used by the daily jobs
NHL_DRY_RUN=1 python -m nhlmodel.pipelines.daily --run morning
pytest -q
```

## Known limits

- The lineup ratings are 5v5 based. A player's power-play value and finishing are added
  separately, but special-teams defense of individual players is not modelled.
- Game-time-decision players are weighted at 50% in the lineup adjustment. DailyFaceoff keeps no
  history, so this cannot be measured; it is an assumption (`lineups.gtd_play_prob`).
- Referees and scratches are only public at warmups, so the morning projection never has them.
- Lineups come from DailyFaceoff; if that page fails, the model falls back to the team's last lineup.
- Goalie pulled mid-game for performance is not modeled (affects saves props).
- Scheduled GitHub runs can start 5-30+ minutes late. In public repos GitHub disables schedules
  after 60 days without repo activity; the daily data commits should keep it active.
- Afternoon games start before the 22:11 pregame snapshot, so their "closing" line is the morning one.
- DailyFaceoff is unofficial and may change or block requests; the fallback guess is less accurate.
