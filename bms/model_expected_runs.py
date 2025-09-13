"""Model wrapper for predicting expected runs allowed by a reliever.

This module defines a thin wrapper around a scikit‑learn regression pipeline.
The model predicts the expected runs allowed over the next ``K`` batters given
features derived from the game state and reliever attributes.  Gradient
Boosting Regressor is chosen as a strong baseline for tabular data.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Categorical and numeric feature names used by the feature builder
CAT = ["park_id", "batter_segment"]
NUM = [
    "outs",
    "runners_on",
    "inning",
    "close_game",
    "late_inning",
    "platoon",
    "rest_days",
    "fatigued",
    "home",
    "leverage_proxy",
]


class ExpectedRunsModel:
    """Regression model for expected runs.

    The model encapsulates a scikit‑learn pipeline composed of a column
    transformer (for one‑hot encoding categorical variables) and a gradient
    boosting regressor.  It exposes convenience methods for training,
    predicting, evaluating and persisting the model.
    """

    def __init__(self) -> None:
        self.pipe: Pipeline | None = None

    def _pipeline(self) -> Pipeline:
        # Preprocess categorical columns with one‑hot encoding and pass
        # numeric columns through unchanged.  Unknown categories are ignored.
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                ("num", "passthrough", NUM),
            ]
        )
        # Gradient boosting regressor parameters chosen for a balance of bias
        # and variance.  They can be tuned via the training script.
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        )
        return Pipeline([("pre", preprocessor), ("model", model)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on the provided features and target.

        Parameters
        ----------
        X: pd.DataFrame
            Feature matrix built by :func:`bms.features.feature_df_from_events`.
        y: pd.Series
            Target vector containing the actual runs allowed over the next
            ``K`` batters.  The caller is responsible for deriving this label.
        """
        self.pipe = self._pipeline()
        self.pipe.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict expected runs for the provided feature matrix.

        Returns
        -------
        pd.Series
            Predicted expected runs for each observation.
        """
        assert self.pipe is not None, "Model has not been trained or loaded"
        return pd.Series(self.pipe.predict(X), index=X.index)

    def evaluate_mae(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute the mean absolute error of predictions on a validation set."""
        return float(mean_absolute_error(y, self.predict(X)))

    def save(self, path: Path) -> None:
        """Persist the model pipeline to disk."""
        assert self.pipe is not None, "Model has not been trained"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipe, path)

    def load(self, path: Path) -> None:
        """Load a persisted model pipeline from disk."""
        self.pipe = joblib.load(path)