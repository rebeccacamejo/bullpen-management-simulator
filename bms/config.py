"""Configuration values for the Bullpen Management Simulator.

Values are read from environment variables when available and fallback to
defaults otherwise.  These variables control the location of the trained
model artifact and the number of batters used to compute expected runs.
"""

from __future__ import annotations

import os
from pathlib import Path


# Path to the trained model artifact.  Can be overridden via the
# MODEL_PATH environment variable.  When deploying to Cloud Run or other
# container platforms, mount the models directory appropriately.
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/bms.joblib"))

# Number of batters to consider when computing expected runs.  Managers may
# choose to forecast runs over three batters, a full inning, or another
# horizon.  K_BATTERS controls the default horizon used during training and
# inference.  Override via the K_BATTERS environment variable.
try:
    K_BATTERS = int(os.getenv("K_BATTERS", "3"))
except ValueError:
    K_BATTERS = 3