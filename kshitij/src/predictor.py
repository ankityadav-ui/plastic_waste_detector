import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from src.config import MAX_DETECTION_LIMIT, WEIGHTS, ENV_WEIGHTS


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _normalize_load(detections: int) -> float:
    return min(detections / MAX_DETECTION_LIMIT, 1.0)


def _growth_rate(current: int, previous: int) -> float:
    return (current - previous) / max(previous, 1)


def _environment_score(humidity: float, crowd: float) -> float:
    humidity_score = humidity / 100.0
    return (
        ENV_WEIGHTS["crowd"] * crowd +
        ENV_WEIGHTS["humidity"] * humidity_score
    )


def _prediction_confidence(avg_conf: float, growth_rate: float) -> float:
    value = 0.6 * avg_conf + 0.4 * (1 - min(abs(growth_rate), 1))
    return max(0, min(value, 1))


def predict_risk(payload: dict) -> dict:
    zone_id = payload["zoneId"]

    current = payload["metricsLast24h"]["totalDetections"]
    previous = payload["metricsPrev24h"]["totalDetections"]
    avg_conf = payload["metricsLast24h"]["avgConfidence"]

    humidity = payload["environment"]["humidity"]
    crowd = payload["environment"]["crowdIndex"]

    load_score = _normalize_load(current)
    growth = _growth_rate(current, previous)
    growth_score = _sigmoid(growth)
    env_score = _environment_score(humidity, crowd)

    risk_score = (
        WEIGHTS["load"] * load_score +
        WEIGHTS["growth"] * growth_score +
        WEIGHTS["environment"] * env_score
    )

    expected_risk = int(risk_score * 100)
    confidence = round(_prediction_confidence(avg_conf, growth), 2)

    return {
        "zoneId": zone_id,
        "predictionWindow": "24h",
        "expectedRisk": expected_risk,
        "confidence": confidence
    }
