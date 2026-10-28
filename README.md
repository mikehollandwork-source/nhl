# NHL projection model

Projects NHL game results and player props every day, compares them with sportsbook lines,
logs every disagreement, grades the disagreements overnight, and posts everything to Telegram.
A separate experiment tries to reverse-engineer how books set their lines.

This is a personal projection and tracking tool, **not a sportsbook**.

## What runs when (GitHub Actions, times UTC)

| Job | Schedule | Posts to Telegram |
|---|---|---|
| `morning` | 15:43 daily | Game cards (score, win %, puck line, totals, fair odds), flagged disagreements, top prop projections, full CSV |
| `pregame` | 22:11 and 01:23 daily | Only changes: a new starting goalie or a new disagreement. Also snapshots lines for closing-line value |
| `nightly` | 09:17 daily | Results recap: model vs book accuracy, flagged-pick record, closing-line value |
| `warmup` | every 15 min, 16:00-03:45 | Only when a game reaches warmups and something material changed: referees, scratches, confirmed goalies, re-run projection |
| `weekly` | Mon 08:37 | Retrain (phase 1 + phase 2) + reverse-engineering report |
| `ablation` | 1st of month 07:41 | Re-tests which input groups help, then retrains |
| `ci` | every push | Lint + tests |
| `bootstrap` | manual only | Rebuilds the `data` branch from scratch |

Every job can also be run by hand from the Actions tab (`Run workflow`), optionally for another
date or in dry-run mode (prints instead of posting).

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
- NHL EDGE tracking: skating speed bursts, zone time, shot speed (season level).
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

**What did not help**, and why that is plausible:
- **Referees:** they shift power-play counts, but only by a few hundredths of a goal.
- **EDGE speed:** already reflected in shots and xG.
- **Travel and goalie detail:** too small or too noisy at this sample size.

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

**Odds plan**: the free The Odds API plan (500 credits/month) covers game lines only (~9 credits/day).
Props are projected but not compared to books. To compare props, upgrade and list markets in
`odds.prop_markets` (e.g. `player_shots_on_goal`, `player_points`, `player_total_saves`).
Historical odds are paid-only, so all book comparisons start from the day collection starts.

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

1. **Telegram**: message @BotFather, `/newbot`, copy the token. Create a channel (or group), add
   the bot as admin, post one message, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` to read the chat id (channels start with `-100`).
2. **The Odds API**: free key at https://the-odds-api.com.
3. **GitHub secrets** (Settings -> Secrets and variables -> Actions):
   `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `ODDS_API_KEY`.
4. The `data` branch holds all stored data and trained models. If it is ever missing, run the
   `bootstrap` workflow (type `rebuild`).
5. Run `morning` once by hand with `dry_run` checked, then without it, to see the Telegram output.

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
