"""Expected goals (xG) model: P(goal | unblocked shot features).

Trained on unblocked attempts (shots on goal, goals, misses) with a goalie in net.
Blocked shots are excluded because their coordinates are the block location, not the shot.
Empty-net shots get a separate constant rate.
XGBoost classifier + isotonic calibration fit on a held-out season.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from nhlmodel.config import data_dir

log = logging.getLogger(__name__)

SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around", "poke",
              "bat", "between-legs", "cradle", "unknown"]
FEATURES = ["distance", "angle", "rebound", "rush", "sk_for", "sk_against", "is_home",
            "score_diff", "is_ot", "own_goalie_pulled"] + [f"st_{s}" for s in SHOT_TYPES]


def model_dir() -> Path:
    p = data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def featurize(shots: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=shots.index)
    X["distance"] = shots["distance"].astype(float)
    X["angle"] = shots["angle"].astype(float)
    X["rebound"] = shots["rebound"].astype(int)
    X["rush"] = shots["rush"].astype(int)
    X["sk_for"] = shots["sk_for"].clip(3, 6).astype(int)
    X["sk_against"] = shots["sk_against"].clip(3, 6).astype(int)
    X["is_home"] = shots["is_home"].astype(int)
    X["score_diff"] = shots["score_diff"].clip(-3, 3).astype(int)
    X["is_ot"] = (shots["period"] > 3).astype(int)
    X["own_goalie_pulled"] = shots["own_goalie_pulled"].astype(int)
    st = shots["shot_type"].fillna("unknown").where(shots["shot_type"].isin(SHOT_TYPES), "unknown")
    for s in SHOT_TYPES:
        X[f"st_{s}"] = (st == s).astype(int)
    return X[FEATURES]


def trainable(shots: pd.DataFrame) -> pd.DataFrame:
    m = shots["event"].isin(["sog", "goal", "miss"]) & ~shots["empty_net"] & shots["distance"].notna()
    return shots[m]


class XGModel:
    def __init__(self, booster: xgb.Booster, iso_x: np.ndarray, iso_y: np.ndarray, en_rate: float, version: str):
        self.booster = booster
        self.iso_x = iso_x
        self.iso_y = iso_y
        self.en_rate = en_rate
        self.version = version

    def predict(self, shots: pd.DataFrame) -> np.ndarray:
        """xG for every row: model for unblocked shots on a goalie, constant for empty net, 0 for blocked."""
        out = np.zeros(len(shots))
        if len(shots) == 0:
            return out
        unblocked = shots["event"].isin(["sog", "goal", "miss"]).to_numpy()
        en = shots["empty_net"].to_numpy(bool)
        has_xy = shots["distance"].notna().to_numpy()
        m = unblocked & ~en & has_xy
        if m.any():
            raw = self.booster.predict(xgb.DMatrix(featurize(shots[m])))
            out[m] = np.interp(raw, self.iso_x, self.iso_y)
        out[unblocked & en] = self.en_rate
        # unblocked shots with no coordinates: use league average for that event
        out[unblocked & ~en & ~has_xy] = float(np.mean(self.iso_y)) if len(self.iso_y) else 0.07
        return out

    def save(self, path: Path | None = None) -> Path:
        path = path or model_dir()
        self.booster.save_model(path / "xg_booster.json")
        (path / "xg_meta.json").write_text(json.dumps({
            "iso_x": self.iso_x.tolist(), "iso_y": self.iso_y.tolist(),
            "en_rate": self.en_rate, "version": self.version}))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> XGModel:
        path = path or model_dir()
        b = xgb.Booster()
        b.load_model(path / "xg_booster.json")
        meta = json.loads((path / "xg_meta.json").read_text())
        return cls(b, np.array(meta["iso_x"]), np.array(meta["iso_y"]), meta["en_rate"], meta["version"])


def train(shots: pd.DataFrame, fit_seasons: list[int], calib_season: int, version: str) -> tuple[XGModel, dict]:
    """Fit on fit_seasons, calibrate (isotonic) on calib_season. Returns model + metrics on calib season."""
    tr = trainable(shots[shots.season.isin(fit_seasons)])
    ca = trainable(shots[shots.season == calib_season])
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5, "eta": 0.05,
              "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 20, "seed": 7}
    dtr = xgb.DMatrix(featurize(tr), label=tr["is_goal"].astype(int))
    dca = xgb.DMatrix(featurize(ca), label=ca["is_goal"].astype(int))
    booster = xgb.train(params, dtr, num_boost_round=1500, evals=[(dca, "calib")],
                        early_stopping_rounds=50, verbose_eval=False)
    booster = booster[: booster.best_iteration + 1]
    raw = booster.predict(dca)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.99).fit(raw, ca["is_goal"].astype(int))
    grid = np.linspace(0, 1, 1001)
    iso_y = iso.predict(grid)

    en = shots[shots.season.isin(fit_seasons) & shots.empty_net & shots.event.isin(["sog", "goal", "miss"])]
    en_rate = float(en["is_goal"].mean()) if len(en) else 0.3

    model = XGModel(booster, grid, iso_y, en_rate, version)
    y = ca["is_goal"].astype(int).to_numpy()
    p_raw = raw
    p_cal = np.interp(raw, grid, iso_y)
    metrics = {
        "n_train": int(len(tr)), "n_calib": int(len(ca)), "best_rounds": int(booster.num_boosted_rounds()),
        "calib_goal_rate": float(y.mean()), "calib_mean_xg_raw": float(p_raw.mean()),
        "logloss_raw": float(log_loss(y, p_raw)), "logloss_cal": float(log_loss(y, p_cal)),
        "logloss_base": float(log_loss(y, np.full_like(p_raw, y.mean()))),
        "brier_cal": float(brier_score_loss(y, p_cal)), "auc": float(roc_auc_score(y, p_raw)),
        "en_rate": en_rate,
    }
    return model, metrics
