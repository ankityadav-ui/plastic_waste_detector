import sys
import os
import logging

# allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from src.predictor import predict_risk

app = Flask(__name__)

# -----------------------------
# Logging Configuration
# -----------------------------
LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "logs",
    "server.log"
)

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------
# Health Check
# -----------------------------
@app.route("/")
def home():
    logger.info("Health check accessed")
    return "Plastic Risk Prediction API is running"


# -----------------------------
# Payload Validation
# -----------------------------
def validate_payload(data):
    if not isinstance(data, dict):
        return "Invalid JSON format"

    required_fields = [
        "zoneId",
        "metricsLast24h",
        "metricsPrev24h",
        "environment"
    ]

    for field in required_fields:
        if field not in data:
            return f"Missing field: {field}"

    if "totalDetections" not in data["metricsLast24h"]:
        return "Missing metricsLast24h.totalDetections"

    if "avgConfidence" not in data["metricsLast24h"]:
        return "Missing metricsLast24h.avgConfidence"

    if "totalDetections" not in data["metricsPrev24h"]:
        return "Missing metricsPrev24h.totalDetections"

    if "humidity" not in data["environment"]:
        return "Missing environment.humidity"

    if "crowdIndex" not in data["environment"]:
        return "Missing environment.crowdIndex"

    return None


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict-risk", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        logger.info("Prediction request received")

        if data is None:
            logger.warning("Request missing JSON body")
            return jsonify({"error": "Request body must be JSON"}), 400

        validation_error = validate_payload(data)
        if validation_error:
            logger.warning(f"Validation failed: {validation_error}")
            return jsonify({"error": validation_error}), 400

        result = predict_risk(data)

        logger.info(
            f"Prediction generated | zone={result['zoneId']} "
            f"risk={result['expectedRisk']} "
            f"confidence={result['confidence']}"
        )

        return jsonify(result), 200

    except Exception as e:
        logger.exception("Internal server error occurred")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    logger.info("Server starting...")
    app.run(host="0.0.0.0", port=5000)
