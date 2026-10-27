import pandas as pd

from nhlmodel.ingest import odds_api
from nhlmodel.tracking import disagree as DG

TEAMS = pd.DataFrame([{"abbrev": "MTL", "common": "Canadiens"}, {"abbrev": "STL", "common": "Blues"},
                      {"abbrev": "UTA", "common": "Mammoth"}, {"abbrev": "TOR", "common": "Maple Leafs"}])

# Shape follows The Odds API v4 /odds response
EVENTS = [{
    "id": "abc", "sport_key": "icehockey_nhl", "commence_time": "2026-10-28T23:00:00Z",
    "home_team": "Montréal Canadiens", "away_team": "St Louis Blues",
    "bookmakers": [{
        "key": "draftkings", "title": "DraftKings", "last_update": "2026-10-28T15:00:00Z",
        "markets": [
            {"key": "h2h", "outcomes": [{"name": "Montréal Canadiens", "price": -130},
                                        {"name": "St Louis Blues", "price": 110}]},
            {"key": "spreads", "outcomes": [{"name": "Montréal Canadiens", "price": 180, "point": -1.5},
                                            {"name": "St Louis Blues", "price": -220, "point": 1.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -105, "point": 6.5},
                                           {"name": "Under", "price": -115, "point": 6.5}]},
        ]}]}]


def test_team_matcher_handles_accents_and_abbreviations():
    m = odds_api.team_matcher(TEAMS)
    assert m("Montreal Canadiens") == "MTL"
    assert m("Montréal Canadiens") == "MTL"
    assert m("St. Louis Blues") == "STL"
    assert m("Utah Mammoth") == "UTA"
    assert m("Toronto Maple Leafs") == "TOR"
    assert m("Unknown Team") is None


def test_normalize_and_devig_pipeline():
    df = odds_api.normalize_game_odds(EVENTS, odds_api.team_matcher(TEAMS), "open", ts="2026-10-28T15:00:00Z")
    assert len(df) == 6
    assert set(df.selection) == {"home", "away", "over", "under"}
    assert df[(df.market == "spreads") & (df.selection == "home")].line.iloc[0] == -1.5
    fair = DG.per_book_fair(df)
    assert len(fair) == 6
    assert abs(fair.groupby("market").fair.sum() - 1).max() < 1e-9
