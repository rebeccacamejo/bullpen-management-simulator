"""Feature engineering for the Bullpen Management Simulator.

This module implements a simple feature builder that converts raw game state
and reliever attributes into a numeric feature set suitable for tabular
regression models.  Each feature is designed to capture an aspect of the
decision: game leverage, base occupancy, platoon advantage, fatigue and
environment.
"""

from __future__ import annotations

import pandas as pd


# Map each base/out encoding to the number of runners on base.  This is a
# coarse representation of traffic on the bases.  Additional features (such
# as base/out state IDs or expected runs) could be incorporated later.
RUNNER_MAP = {
    "---": 0,
    "1--": 1,
    "-2-": 1,
    "--3": 1,
    "12-": 2,
    "1-3": 2,
    "-23": 2,
    "123": 3,
}


def feature_df_from_events(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of event data into a feature matrix.

    The input DataFrame must contain the following columns:

    * ``outs``: int
    * ``runners``: str (one of the keys in RUNNER_MAP)
    * ``inning``: int
    * ``score_diff``: int
    * ``platoon``: int (1 if the reliever has a platoon advantage, else 0)
    * ``rest_days``: int
    * ``pitches_last_outing``: int
    * ``home``: int (1 for home, 0 for away)
    * ``park_id``: int
    * ``batter_segment``: int

    Returns a new DataFrame with numeric and categorical columns ready for
    consumption by scikit‑learn pipelines.  No model‑specific transformations
    (such as one‑hot encoding) are performed here; these are handled in the
    model pipeline.
    """

    out = pd.DataFrame(index=df.index)
    # Outs (0‒2)
    out["outs"] = df["outs"].astype(int).clip(0, 2)
    # Number of runners on base (0‒3)
    out["runners_on"] = df["runners"].map(RUNNER_MAP).fillna(0).astype(int)
    # Inning number
    out["inning"] = df["inning"].astype(int)
    # Close game indicator: absolute score diff <= 1
    out["close_game"] = (df["score_diff"].abs() <= 1).astype(int)
    # Late inning indicator: inning >= 7
    out["late_inning"] = (df["inning"] >= 7).astype(int)
    # Platoon advantage: 1 if favourable, 0 otherwise
    out["platoon"] = df.get("platoon", 0).astype(int)
    # Days of rest (capped at 5 to avoid outliers)
    out["rest_days"] = df["rest_days"].astype(int).clip(0, 5)
    # Fatigue indicator: 1 if last outing had >20 pitches
    out["fatigued"] = (df["pitches_last_outing"] > 20).astype(int)
    # Home indicator (1 for home, 0 for away)
    out["home"] = df["home"].astype(int)
    # Categorical park identifier
    out["park_id"] = df["park_id"].astype(int)
    # Categorical batter segment (1, 2 or 3)
    out["batter_segment"] = df["batter_segment"].astype(int)
    # Simple leverage proxy combining closeness, lateness and traffic on base
    out["leverage_proxy"] = (
        out["late_inning"] * 0.7
        + out["close_game"] * 0.6
        + (out["runners_on"] >= 2).astype(int) * 0.5
    )
    return out