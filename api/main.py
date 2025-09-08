"""FastAPI application entry point for the BMS service."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from bms.config import MODEL_PATH
from bms.domain import GameState, Reliever
from bms.model_expected_runs import ExpectedRunsModel
from bms.recommender import BullpenRecommender
from .schemas import RecommendIn, RecommendOut, CandidateOut
from .metrics import REQUESTS, LATENCY


app = FastAPI(title="Bullpen Management Simulator", version="0.1.0")

# Allow cross‑origin requests (UI runs on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once at startup
MODEL = ExpectedRunsModel()
if MODEL_PATH.exists():
    MODEL.load(MODEL_PATH)
RECO = BullpenRecommender(MODEL)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next: Any) -> Response:
    """Middleware to record request metrics."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    path = request.url.path
    LATENCY.labels(endpoint=path).observe(duration)
    REQUESTS.labels(endpoint=path, status=str(response.status_code)).inc()
    return response


@app.get("/health")
def healthz() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend", response_model=RecommendOut)
def recommend(payload: RecommendIn) -> RecommendOut:
    """Recommend a reliever based on current game state and bullpen options."""
    state_in = payload.state
    state = GameState(**state_in.model_dump())
    bullpen = [Reliever(**r.model_dump()) for r in payload.bullpen]
    result = RECO.recommend(state, bullpen)
    # Convert candidates into typed objects for response validation
    result["candidates"] = [CandidateOut(**c) for c in result["candidates"]]
    return RecommendOut(**result)