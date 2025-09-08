"""Prometheus metrics definitions for the BMS API."""

from prometheus_client import Counter, Histogram, Gauge

# Count total requests by endpoint and status code
REQUESTS = Counter(
    "bms_requests_total",
    "Total number of requests received",
    ["endpoint", "status"],
)

# Measure request latency (seconds) per endpoint
LATENCY = Histogram(
    "bms_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
)

# Gauge for tracking online model performance (mean absolute error)
ONLINE_MAE = Gauge(
    "bms_online_mae",
    "Online mean absolute error over the most recent prediction window",
)