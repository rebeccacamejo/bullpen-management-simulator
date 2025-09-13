"""Bullpen Management Simulator (BMS) core package.

This package contains domain models, feature builders, model wrappers and the
recommender used by the API.  See :mod:`bms.model_expected_runs` for the
primary regression model and :mod:`bms.recommender` for the business logic
used to select a reliever.
"""

from .config import K_BATTERS, MODEL_PATH
from .domain import GameState, Reliever
from .features import feature_df_from_events
from .model_expected_runs import ExpectedRunsModel
from .recommender import BullpenRecommender

__all__ = [
    "GameState",
    "Reliever",
    "ExpectedRunsModel",
    "BullpenRecommender",
    "feature_df_from_events",
    "K_BATTERS",
    "MODEL_PATH",
]