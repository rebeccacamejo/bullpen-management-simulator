"""Domain entities for the Bullpen Management Simulator.

This module defines the data classes used to represent the state of the game and
the relievers under consideration.  These classes are lightweight and
serializable, making them suitable for API payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Enumerated type for base/out state encoded as a string.  Each character
# represents a base; '-' means empty, '1' means a runner on first, '2' on
# second, and '3' on third.  A full state like '123' means bases loaded.
Runners = Literal["---", "1--", "-2-", "--3", "12-", "1-3", "-23", "123"]


@dataclass(frozen=True)
class GameState:
    """Representation of the game context when choosing a reliever.

    Parameters
    ----------
    outs: int
        Number of outs in the inning (0‒2).
    runners: Runners
        Encoded base‑out state (e.g. '1-3' means runners on first and third).
    inning: int
        Current inning number (1‒9+).  In extra innings values can exceed 9.
    score_diff: int
        Home team runs minus away team runs.  A positive value means the
        pitcher’s team is leading, negative means trailing.
    home: bool
        Whether the reliever’s team is at home.  Park effects and last ups
        may influence the leverage of the situation.
    park_id: int
        A categorical identifier for the ballpark.  This placeholder can be
        mapped to altitude, dimensions or climate effects in future work.
    batter_segment: int
        Index for the part of the batting order expected to bat next
        (1 for top, 2 for middle, 3 for bottom).  Used as a rough proxy for
        batter quality and handedness distribution.
    leverage_hint: float | None
        Optional hint overriding the computed leverage proxy.  If supplied,
        this value will be used directly in feature computation to account
        for advanced metrics like LI (leverage index).
    """

    outs: int
    runners: Runners
    inning: int
    score_diff: int
    home: bool
    park_id: int
    batter_segment: int
    leverage_hint: float | None = None


@dataclass(frozen=True)
class Reliever:
    """Representation of a relief pitcher.

    Parameters
    ----------
    reliever_id: str
        Unique identifier for the reliever (name, roster ID or jersey number).
    throws: str
        Handedness of the pitcher ('R' or 'L').
    rest_days: int
        Number of days since the pitcher’s last appearance.  A value of
        ``0`` means they pitched in the previous game.
    pitches_last_outing: int
        Number of pitches thrown in the previous outing.  Higher values
        suggest greater fatigue.
    available: bool, optional
        Whether the pitcher is available for selection.  If false, the
        recommender applies a huge penalty to avoid selection.
    notes: str, optional
        Arbitrary text with any additional information (e.g. injury status).
    """

    reliever_id: str
    throws: Literal["R", "L"]
    rest_days: int
    pitches_last_outing: int
    available: bool = True
    notes: str = ""