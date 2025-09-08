"""Pydantic schemas for the BMS API."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, List

# Type alias for base/out state strings
Runners = Literal["---", "1--", "-2-", "--3", "12-", "1-3", "-23", "123"]


class GameStateIn(BaseModel):
    """Request schema for the current game state."""

    outs: int = Field(ge=0, le=2)
    runners: Runners
    inning: int = Field(ge=1, le=12)
    score_diff: int
    home: bool
    park_id: int = Field(ge=0)
    batter_segment: int = Field(ge=1, le=3)


class RelieverIn(BaseModel):
    """Request schema for a reliever candidate."""

    reliever_id: str
    throws: Literal["R", "L"]
    rest_days: int = Field(ge=0, le=5)
    pitches_last_outing: int = Field(ge=0, le=60)
    available: bool = True
    notes: str | None = None


class RecommendIn(BaseModel):
    """Schema for the POST /recommend endpoint."""

    state: GameStateIn
    bullpen: List[RelieverIn]


class CandidateOut(BaseModel):
    """Response schema for each candidate's predicted values."""

    reliever_id: str
    expected_runs: float
    expected_runs_penalised: float


class RecommendOut(BaseModel):
    """Response schema for the recommendation endpoint."""

    recommendation: dict
    candidates: List[CandidateOut]
    explanations: dict
    state: dict