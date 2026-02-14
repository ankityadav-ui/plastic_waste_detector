# Plastic Waste Risk Prediction API

## Base URL
http://localhost:5000

---

## Endpoint
POST /predict-risk

---

## Description
Predicts plastic waste accumulation risk for a zone over the next 24 hours.

---

## Request JSON Format

{
  "zoneId": "string",
  "metricsLast24h": {
    "totalDetections": number,
    "avgConfidence": number (0–1)
  },
  "metricsPrev24h": {
    "totalDetections": number
  },
  "environment": {
    "humidity": number (0–100),
    "crowdIndex": number (0–1)
  }
}

---

## Response JSON Format

{
  "zoneId": "string",
  "predictionWindow": "24h",
  "expectedRisk": number (0–100),
  "confidence": number (0–1)
}

---

## Example Request

{
  "zoneId": "Z-101",
  "metricsLast24h": {
    "totalDetections": 12,
    "avgConfidence": 0.8
  },
  "metricsPrev24h": {
    "totalDetections": 6
  },
  "environment": {
    "humidity": 70,
    "crowdIndex": 0.6
  }
}

---

## Example Response

{
  "zoneId": "Z-101",
  "predictionWindow": "24h",
  "expectedRisk": 46,
  "confidence": 0.48
}