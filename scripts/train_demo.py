"""Train a demonstration model for the BMS.

This script trains the expected runs model on either synthetic data or
a small sample of Statcast data (if available).  It logs the run to MLflow
and writes the model artefact to ``models/bms.joblib``.  Use this as a
starting point for building your own training pipeline.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from bms.features import feature_df_from_events
from bms.model_expected_runs import ExpectedRunsModel


def generate_synthetic_data(n: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic dataset for demonstration purposes.

    Returns
    -------
    X: pd.DataFrame
        Feature matrix following the expected schema.
    y: pd.Series
        Simulated runs allowed over the next K batters.
    """
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "outs": rng.integers(0, 3, size=n),
        "runners": rng.choice(["---", "1--", "-2-", "--3", "12-", "1-3", "-23", "123"], size=n),
        "inning": rng.integers(7, 10, size=n),
        "score_diff": rng.integers(-3, 4, size=n),
        "platoon": rng.integers(0, 2, size=n),
        "rest_days": rng.integers(0, 4, size=n),
        "pitches_last_outing": rng.integers(5, 35, size=n),
        "home": rng.integers(0, 2, size=n),
        "park_id": rng.integers(1, 4, size=n),
        "batter_segment": rng.integers(1, 4, size=n),
    })
    # Simulate the target: more runs if late, close game, runners on, low rest and high fatigue
    y = (
        0.2 * ((df["inning"] >= 8) & (df["score_diff"].abs() <= 1)).astype(int)
        + 0.25 * (df["runners"] != "---").astype(int)
        + 0.15 * (df["rest_days"] == 0).astype(int)
        + 0.08 * (df["pitches_last_outing"] > 20).astype(int)
        + rng.normal(0, 0.05, size=n)
    )
    X = feature_df_from_events(df)
    return X, pd.Series(y, name="runs")


def main() -> None:
    X, y = generate_synthetic_data(1000)
    # Split chronologically: first 70% train, remainder validate
    split_idx = int(len(X) * 0.7)
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
    model = ExpectedRunsModel()
    mlflow.set_experiment("bms_expected_runs")
    with mlflow.start_run():
        model.fit(X_train, y_train)
        val_mae = model.evaluate_mae(X_val, y_val)
        mlflow.log_metric("val_mae", val_mae)
        # Persist model
        artefact_path = Path("models/bms.joblib")
        model.save(artefact_path)
        mlflow.log_artifact(str(artefact_path))
        print(f"Validation MAE: {val_mae:.4f}")


if __name__ == "__main__":
    main()