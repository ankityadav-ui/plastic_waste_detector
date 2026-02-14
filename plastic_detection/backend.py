import io
import json
import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("backend")

VALID_PLASTIC_TYPES = [
    "plastic_bottle", "plastic_bag", "plastic_wrapper",
    "plastic_cutlery", "plastic_casing",
]
VALID_MATERIALS = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "OTHER"]


class DetectionEvent(BaseModel):
    zoneId: str = Field(..., examples=["Z-101"])
    plasticType: str = Field(..., examples=["plastic_bottle"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: str = Field(..., examples=["2026-02-14T12:00:00"])
    itemType: str | None = Field(None, examples=["Plastic Bottle"])
    material: str | None = Field(None, examples=["PET"])
    materialName: str | None = Field(None, examples=["Polyethylene Terephthalate"])
    resinCode: int | None = Field(None, ge=1, le=7, examples=[1])
    recyclable: bool | None = Field(None, examples=[True])


class StatsResponse(BaseModel):
    total_detections: int
    by_type: dict[str, int]
    by_zone: dict[str, int]
    latest: list[DetectionEvent]


detections: list[dict] = []
LOG_FILE = Path("detections_log.json")


def _load_log_file() -> None:
    if LOG_FILE.is_file():
        count = 0
        with open(LOG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        detections.append(json.loads(line))
                        count += 1
                    except json.JSONDecodeError:
                        continue
        if count:
            log.info("Loaded %d previous detections from %s", count, LOG_FILE)


def _append_to_log(records: list[dict]) -> None:
    with open(LOG_FILE, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from detector import PlasticDetector
            _detector = PlasticDetector()
            log.info("PlasticDetector loaded for image detection endpoint")
        except Exception as exc:
            log.error("Failed to load detector: %s", exc)
            raise HTTPException(status_code=503, detail=f"Detector not available: {exc}")
    return _detector


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_log_file()
    log.info("Backend server started")
    yield
    log.info("Backend server shutting down")


app = FastAPI(
    title="Plastic Detection Backend",
    description="Central API for the plastic waste surveillance system",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "online", "module": "Plastic Detection Backend", "version": "2.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "detections_stored": len(detections),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/report")
def report_detection(event: DetectionEvent):
    record = event.model_dump()
    record["received_at"] = datetime.now(timezone.utc).isoformat()
    detections.append(record)
    _append_to_log([record])
    return {"status": "accepted", "total": len(detections), "event": record}


@app.post("/report/batch")
def report_batch(events: list[DetectionEvent]):
    results = []
    for event in events:
        record = event.model_dump()
        record["received_at"] = datetime.now(timezone.utc).isoformat()
        detections.append(record)
        results.append(record)

    _append_to_log(results)
    return {"status": "accepted", "count": len(results), "total": len(detections)}


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    zone: str = Query("Z-101", description="Monitoring zone ID"),
    conf: float = Query(0.4, ge=0.0, le=1.0, description="Confidence threshold"),
):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG/PNG)")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    detector = _get_detector()
    detector.conf_threshold = conf
    results = detector.detect(frame)
    detector.draw(frame, results)

    events = []
    for det in results:
        record = {
            "zoneId": zone,
            "plasticType": det["plastic_type"],
            "itemType": det.get("item_type", ""),
            "material": det.get("material", ""),
            "materialName": det.get("material_name", ""),
            "resinCode": det.get("resin_code", 7),
            "recyclable": det.get("recyclable", False),
            "confidence": det["confidence"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "received_at": datetime.now(timezone.utc).isoformat(),
            "source": file.filename,
        }
        detections.append(record)
        events.append(record)
    if events:
        _append_to_log(events)

    _, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_bytes = io.BytesIO(img_encoded.tobytes())

    return StreamingResponse(
        img_bytes,
        media_type="image/jpeg",
        headers={
            "X-Detections-Count": str(len(results)),
            "X-Detections": json.dumps([{
                "type": d["plastic_type"],
                "item": d.get("item_type", ""),
                "material": d.get("material", ""),
                "resin_code": d.get("resin_code", 7),
                "recyclable": d.get("recyclable", False),
                "confidence": d["confidence"],
            } for d in results]),
        },
    )


@app.get("/detections")
def get_detections(
    limit: int = Query(50, ge=1, le=500),
    zone: str | None = None,
    plastic_type: str | None = None,
):
    filtered = detections

    if zone:
        filtered = [d for d in filtered if d.get("zoneId") == zone]
    if plastic_type:
        filtered = [d for d in filtered if d.get("plasticType") == plastic_type]

    return {"total": len(filtered), "detections": filtered[-limit:][::-1]}


@app.get("/stats")
def get_stats():
    by_type: dict[str, int] = {}
    by_zone: dict[str, int] = {}
    by_material: dict[str, int] = {}
    recyclable_count = 0
    non_recyclable_count = 0

    for d in detections:
        ptype = d.get("plasticType", "unknown")
        zone = d.get("zoneId", "unknown")
        mat = d.get("material", "unknown")
        by_type[ptype] = by_type.get(ptype, 0) + 1
        by_zone[zone] = by_zone.get(zone, 0) + 1
        by_material[mat] = by_material.get(mat, 0) + 1
        if d.get("recyclable"):
            recyclable_count += 1
        else:
            non_recyclable_count += 1

    latest = detections[-10:][::-1] if detections else []

    return {
        "total_detections": len(detections),
        "by_type": by_type,
        "by_zone": by_zone,
        "by_material": by_material,
        "recyclable": recyclable_count,
        "non_recyclable": non_recyclable_count,
        "latest": latest,
    }


@app.delete("/detections")
def clear_detections():
    detections.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
