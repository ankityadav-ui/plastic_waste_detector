import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("detector")

RESIN_CODES = {
    "PET":   {"code": 1, "name": "Polyethylene Terephthalate", "recyclable": True},
    "HDPE":  {"code": 2, "name": "High-Density Polyethylene",  "recyclable": True},
    "PVC":   {"code": 3, "name": "Polyvinyl Chloride",         "recyclable": False},
    "LDPE":  {"code": 4, "name": "Low-Density Polyethylene",   "recyclable": True},
    "PP":    {"code": 5, "name": "Polypropylene",              "recyclable": True},
    "PS":    {"code": 6, "name": "Polystyrene",                "recyclable": False},
    "OTHER": {"code": 7, "name": "Other Plastics",             "recyclable": False},
}

PLASTIC_MAP: dict[str, dict] = {
    "bottle": {
        "item_type": "Plastic Bottle",
        "is_plastic": True,
        "material": "PET",
        "resin_code": 1,
        "recyclable": True,
        "description": "PET water/soda bottles",
    },
    "cup": {
        "item_type": "Plastic Cup",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": True,
        "description": "Polypropylene cups",
    },
    "wine glass": {
        "item_type": "Plastic Cup/Glass",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene disposable glass",
    },
    "bowl": {
        "item_type": "Plastic Bowl",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": True,
        "description": "Polypropylene food container",
    },
    "vase": {
        "item_type": "Plastic Container",
        "is_plastic": True,
        "material": "PET",
        "resin_code": 1,
        "recyclable": True,
        "description": "PET container",
    },
    "handbag": {
        "item_type": "Plastic Bag",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE shopping/carry bag",
    },
    "backpack": {
        "item_type": "Plastic Bag (Large)",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE/HDPE large carry bag",
    },
    "suitcase": {
        "item_type": "Plastic Container (Large)",
        "is_plastic": True,
        "material": "HDPE",
        "resin_code": 2,
        "recyclable": True,
        "description": "HDPE rigid container",
    },
    "umbrella": {
        "item_type": "Plastic Item",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "Mixed plastic/nylon",
    },
    "cell phone": {
        "item_type": "Plastic Wrapper/Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "Mixed plastic casing — e-waste",
    },
    "remote": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": True,
        "description": "Polypropylene plastic casing",
    },
    "toothbrush": {
        "item_type": "Plastic Hygiene Item",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene handle — mixed materials",
    },
    "scissors": {
        "item_type": "Plastic Handle Item",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene handles — mixed materials",
    },
    "mouse": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/mixed plastic casing — e-waste",
    },
    "keyboard": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/mixed plastic casing — e-waste",
    },
    "fork": {
        "item_type": "Plastic Cutlery",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene single-use cutlery",
    },
    "knife": {
        "item_type": "Plastic Cutlery",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene single-use cutlery",
    },
    "spoon": {
        "item_type": "Plastic Cutlery",
        "is_plastic": True,
        "material": "PS",
        "resin_code": 6,
        "recyclable": False,
        "description": "Polystyrene single-use cutlery",
    },
    "refrigerator": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "HDPE",
        "resin_code": 2,
        "recyclable": True,
        "description": "HDPE panels — industrial recycling",
    },
    "microwave": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene casing — e-waste",
    },
    "sink": {
        "item_type": "Plastic Fixture",
        "is_plastic": True,
        "material": "PVC",
        "resin_code": 3,
        "recyclable": False,
        "description": "PVC plumbing fixture",
    },
    "toilet": {
        "item_type": "Plastic Fixture",
        "is_plastic": True,
        "material": "PVC",
        "resin_code": 3,
        "recyclable": False,
        "description": "PVC/HDPE plumbing",
    },
    "book": {
        "item_type": "Plastic-Wrapped Item",
        "is_plastic": True,
        "material": "LDPE",
        "resin_code": 4,
        "recyclable": True,
        "description": "LDPE shrink-wrap/lamination",
    },
    "laptop": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/polycarbonate casing — e-waste",
    },
    "tvmonitor": {
        "item_type": "Plastic Casing",
        "is_plastic": True,
        "material": "OTHER",
        "resin_code": 7,
        "recyclable": False,
        "description": "ABS/polycarbonate casing — e-waste",
    },
    "hair drier": {
        "item_type": "Plastic Appliance",
        "is_plastic": True,
        "material": "PP",
        "resin_code": 5,
        "recyclable": False,
        "description": "Polypropylene casing — e-waste",
    },
}


def _get_plastic_category(info: dict) -> str:
    item = info["item_type"].lower()
    if "bottle" in item or "cup" in item or "glass" in item or "bowl" in item or "container" in item:
        return "plastic_bottle"
    elif "bag" in item or "carrier" in item:
        return "plastic_bag"
    elif "cutlery" in item:
        return "plastic_cutlery"
    elif "casing" in item or "appliance" in item or "fixture" in item:
        return "plastic_casing"
    else:
        return "plastic_wrapper"


COLORS: dict[str, tuple[int, int, int]] = {
    "plastic_bottle":  (0, 255, 0),
    "plastic_bag":     (0, 165, 255),
    "plastic_wrapper": (0, 0, 255),
    "plastic_cutlery": (255, 0, 255),
    "plastic_casing":  (255, 255, 0),
}

COLOR_RECYCLABLE = (0, 200, 0)
COLOR_NOT_RECYCLABLE = (0, 0, 200)
DEFAULT_COLOR = (255, 255, 0)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CFG_PATH = MODEL_DIR / "yolov4.cfg"
WEIGHTS_PATH = MODEL_DIR / "yolov4.weights"
NAMES_PATH = MODEL_DIR / "coco.names"
CFG_PATH_TINY = MODEL_DIR / "yolov4-tiny.cfg"
WEIGHTS_PATH_TINY = MODEL_DIR / "yolov4-tiny.weights"

BACKEND_URL = "http://localhost:8000"
REPORT_ENDPOINT = "/report"
BATCH_ENDPOINT = "/report/batch"


class PlasticDetector:
    """YOLOv4 detector that filters for plastic-related COCO objects."""

    def __init__(
        self,
        cfg: str | Path = CFG_PATH,
        weights: str | Path = WEIGHTS_PATH,
        names: str | Path = NAMES_PATH,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        input_size: int = 608,
        use_gpu: bool = False,
        multi_scale: bool = True,
        temporal_smooth: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.temporal_smooth = temporal_smooth
        self._history: list[list[dict]] = []
        self._history_len = 5

        self.class_names = self._load_names(names)
        log.info("Loaded %d COCO class names", len(self.class_names))

        self.index_to_plastic: dict[int, dict] = {}
        for idx, name in enumerate(self.class_names):
            if name in PLASTIC_MAP:
                info = PLASTIC_MAP[name].copy()
                info["plastic_type"] = _get_plastic_category(info)
                self.index_to_plastic[idx] = info
        log.info("Mapped %d COCO classes to plastic categories", len(self.index_to_plastic))

        cfg_str, weights_str = str(cfg), str(weights)
        if not os.path.isfile(cfg_str):
            log.error("Config not found: %s", cfg_str)
            sys.exit(1)
        if not os.path.isfile(weights_str):
            log.error("Weights not found: %s", weights_str)
            sys.exit(1)

        self.net = cv2.dnn.readNetFromDarknet(cfg_str, weights_str)
        log.info("Loaded YOLOv4 network")

        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            log.info("Using CUDA backend")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        try:
            out_indices = self.net.getUnconnectedOutLayers().flatten()
        except AttributeError:
            out_indices = self.net.getUnconnectedOutLayers()
        self.output_layers = [layer_names[i - 1] for i in out_indices]

    @staticmethod
    def _load_names(path: str | Path) -> list[str]:
        p = Path(path)
        if not p.is_file():
            log.error("Names file not found: %s", p)
            sys.exit(1)
        with open(p, "r") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _detect_at_scale(self, frame: np.ndarray, input_size: int) -> tuple[
        list[list[int]], list[float], list[int]
    ]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (input_size, input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id not in self.index_to_plastic:
                    continue
                if confidence < self.conf_threshold:
                    continue

                cx, cy, bw, bh = detection[0:4]
                cx = int(cx * w)
                cy = int(cy * h)
                bw = int(bw * w)
                bh = int(bh * h)
                x = cx - bw // 2
                y = cy - bh // 2

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        return boxes, confidences, class_ids

    def _iou(self, box_a: tuple, box_b: tuple) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _apply_temporal_smoothing(self, detections: list[dict]) -> list[dict]:
        self._history.append(detections)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        if len(self._history) < 2:
            return detections

        for det in detections:
            match_count = 0
            for past_frame in self._history[:-1]:
                for past_det in past_frame:
                    if (past_det["plastic_type"] == det["plastic_type"]
                            and self._iou(det["box"], past_det["box"]) > 0.3):
                        match_count += 1
                        break
            if match_count >= 2:
                det["confidence"] = min(det["confidence"] * 1.15, 0.99)
                det["confidence"] = round(det["confidence"], 4)

        current_boxes = [d["box"] for d in detections]
        for past_det in (self._history[-2] if len(self._history) >= 2 else []):
            matched = any(
                self._iou(past_det["box"], cb) > 0.3 for cb in current_boxes
            )
            if not matched and past_det["confidence"] > 0.35:
                recovered = dict(past_det)
                recovered["confidence"] = round(past_det["confidence"] * 0.85, 4)
                detections.append(recovered)

        return detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        enhanced = self._preprocess(frame)

        all_boxes: list[list[int]] = []
        all_confs: list[float] = []
        all_cids: list[int] = []

        if self.multi_scale:
            scales = [self.input_size, self.input_size - 192, self.input_size + 192]
            scales = [max(320, min(832, (s // 32) * 32)) for s in scales]
            scales = sorted(set(scales))
        else:
            scales = [self.input_size]

        for scale in scales:
            b, c, ids = self._detect_at_scale(enhanced, scale)
            all_boxes.extend(b)
            all_confs.extend(c)
            all_cids.extend(ids)

        results: list[dict] = []
        if all_boxes:
            indices = cv2.dnn.NMSBoxes(
                all_boxes, all_confs, self.conf_threshold, self.nms_threshold,
            )
            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                for i in indices:
                    cid = all_cids[i]
                    info = self.index_to_plastic[cid]
                    resin = RESIN_CODES.get(info["material"], RESIN_CODES["OTHER"])
                    results.append({
                        "label": self.class_names[cid],
                        "plastic_type": info["plastic_type"],
                        "item_type": info["item_type"],
                        "is_plastic": info["is_plastic"],
                        "material": info["material"],
                        "material_name": resin["name"],
                        "resin_code": info["resin_code"],
                        "recyclable": info["recyclable"],
                        "description": info["description"],
                        "confidence": round(all_confs[i], 4),
                        "box": tuple(all_boxes[i]),
                    })

        if self.temporal_smooth:
            results = self._apply_temporal_smoothing(results)

        return results

    @staticmethod
    def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        for det in detections:
            x, y, bw, bh = det["box"]
            ptype = det["plastic_type"]
            material = det.get("material", "?")
            resin = det.get("resin_code", "?")
            recyclable = det.get("recyclable", False)
            item_type = det.get("item_type", det["label"])
            conf = det["confidence"]

            color = COLORS.get(ptype, DEFAULT_COLOR)
            recycle_tag = "RECYCLABLE" if recyclable else "NOT RECYCLABLE"
            recycle_color = COLOR_RECYCLABLE if recyclable else COLOR_NOT_RECYCLABLE

            line1 = f"{item_type} {conf:.0%}"
            line2 = f"{material} (#{resin}) | {recycle_tag}"

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            (tw1, th1), bl1 = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - th1 - bl1 - 4), (x + tw1 + 4, y), color, -1)
            cv2.putText(frame, line1, (x + 2, y - bl1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            (tw2, th2), bl2 = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            y2 = y + bh + th2 + bl2 + 4
            cv2.rectangle(frame, (x, y + bh), (x + tw2 + 4, y2), recycle_color, -1)
            cv2.putText(frame, line2, (x + 2, y2 - bl2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


class BackendReporter:
    """Sends detection events to the FastAPI backend via HTTP."""

    def __init__(
        self,
        url: str = BACKEND_URL,
        zone_id: str = "Z-101",
        batch_size: int = 5,
        cooldown: float = 2.0,
    ):
        if not HAS_REQUESTS:
            log.warning("'requests' not installed — backend reporting disabled")
        self.url = url.rstrip("/")
        self.zone_id = zone_id
        self.batch_size = batch_size
        self.cooldown = cooldown
        self._last_report: dict[str, float] = {}
        self._buffer: list[dict] = []

    def _should_report(self, plastic_type: str) -> bool:
        now = time.time()
        last = self._last_report.get(plastic_type, 0)
        if now - last >= self.cooldown:
            self._last_report[plastic_type] = now
            return True
        return False

    def report(self, detections: list[dict]) -> None:
        if not HAS_REQUESTS:
            return

        for det in detections:
            if not self._should_report(det["plastic_type"]):
                continue
            event = {
                "zoneId": self.zone_id,
                "plasticType": det["plastic_type"],
                "itemType": det.get("item_type", ""),
                "material": det.get("material", ""),
                "materialName": det.get("material_name", ""),
                "resinCode": det.get("resin_code", 7),
                "recyclable": det.get("recyclable", False),
                "confidence": det["confidence"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._buffer.append(event)

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        try:
            if len(self._buffer) == 1:
                r = requests.post(
                    f"{self.url}{REPORT_ENDPOINT}",
                    json=self._buffer[0],
                    timeout=3,
                )
            else:
                r = requests.post(
                    f"{self.url}{BATCH_ENDPOINT}",
                    json=self._buffer,
                    timeout=3,
                )
            if r.ok:
                log.debug("Reported %d event(s) to backend", len(self._buffer))
            else:
                log.warning("Backend responded %d: %s", r.status_code, r.text[:200])
        except requests.ConnectionError:
            log.warning("Backend unreachable at %s", self.url)
        except Exception as exc:
            log.warning("Report failed: %s", exc)
        finally:
            self._buffer.clear()

    def flush_remaining(self) -> None:
        self._flush()


def _overlay_info(frame: np.ndarray, fps: float, count: int, zone: str) -> None:
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {count}",
        f"Zone: {zone}",
    ]
    for i, line in enumerate(info_lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def run_webcam(
    detector: PlasticDetector,
    camera_idx: int = 0,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        log.error("Cannot open webcam (index %d)", camera_idx)
        sys.exit(1)

    log.info("Webcam opened — press 'q' to quit")
    writer = None
    fps = 0.0
    frame_count = 0
    total_detections = 0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame — retrying")
                time.sleep(0.1)
                continue

            detections = detector.detect(frame)
            total_detections += len(detections)
            frame_count += 1

            detector.draw(frame, detections)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            _overlay_info(frame, fps, len(detections), zone)

            if reporter and detections:
                reporter.report(detections)

            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_output, fourcc, 20, (w, h))
            if writer:
                writer.write(frame)

            if not no_display:
                cv2.imshow("Plastic Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                if frame_count % 100 == 0:
                    log.info("Frame %d | FPS %.1f | Detections: %d",
                             frame_count, fps, len(detections))
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        if reporter:
            reporter.flush_remaining()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log.info("Done — %d frames, %d total detections", frame_count, total_detections)


def run_video(
    detector: PlasticDetector,
    source: str,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    if not os.path.isfile(source):
        log.error("Video file not found: %s", source)
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open video: %s", source)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    log.info("Video: %s (%d frames @ %.1f FPS)", source, total_frames, input_fps)

    writer = None
    frame_count = 0
    total_detections = 0

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            total_detections += len(detections)
            frame_count += 1

            detector.draw(frame, detections)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            _overlay_info(frame, fps, len(detections), zone)

            if reporter and detections:
                reporter.report(detections)

            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_output, fourcc, input_fps, (w, h))
            if writer:
                writer.write(frame)

            if not no_display:
                cv2.imshow("Plastic Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

            if frame_count % 50 == 0:
                pct = frame_count / max(total_frames, 1) * 100
                log.info("Progress: %d/%d (%.0f%%)", frame_count, total_frames, pct)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        if reporter:
            reporter.flush_remaining()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log.info("Done — %d/%d frames, %d total detections",
                 frame_count, total_frames, total_detections)


def run_image(
    detector: PlasticDetector,
    source: str,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    paths: list[str] = []
    if os.path.isdir(source):
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            str(p) for p in Path(source).iterdir() if p.suffix.lower() in exts
        )
        if not paths:
            log.error("No images found in %s", source)
            sys.exit(1)
        log.info("Found %d images in %s", len(paths), source)
    elif os.path.isfile(source):
        paths = [source]
    else:
        log.error("Image not found: %s", source)
        sys.exit(1)

    for img_path in paths:
        frame = cv2.imread(img_path)
        if frame is None:
            log.warning("Cannot read image: %s — skipping", img_path)
            continue

        detections = detector.detect(frame)
        detector.draw(frame, detections)
        log.info("%s — %d detection(s): %s", img_path, len(detections),
                 [d["plastic_type"] for d in detections] if detections else "none")

        if reporter and detections:
            reporter.report(detections)

        out_path = save_output
        if out_path is None:
            stem = Path(img_path).stem
            out_path = str(BASE_DIR / f"{stem}_detected.jpg")
        cv2.imwrite(out_path, frame)
        log.info("Saved: %s", out_path)

        if not no_display:
            cv2.imshow("Plastic Detection", frame)
            log.info("Press any key to continue (or 'q' to quit)")
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                break

    if reporter:
        reporter.flush_remaining()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plastic Waste Detection Engine (YOLOv4 / Hybrid Pipeline)",
    )
    p.add_argument("--mode", choices=["webcam", "video", "image"], default="webcam")
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--size", type=int, default=608)
    p.add_argument("--no-multiscale", action="store_true")
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--zone", type=str, default="Z-101")
    p.add_argument("--send", action="store_true")
    p.add_argument("--backend-url", type=str, default=BACKEND_URL)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--use-taco", action="store_true")
    p.add_argument("--use-classifier", action="store_true")
    p.add_argument("--taco-weights", type=str, default=None)
    p.add_argument("--classifier-weights", type=str, default=None)
    p.add_argument("--trashnet-weights", type=str, default=None)
    p.add_argument("--detection-threshold", type=float, default=0.5)
    p.add_argument("--classification-threshold", type=float, default=0.7)
    p.add_argument("--frame-skip", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.mode in ("video", "image") and not args.source:
        log.error("--source is required for %s mode", args.mode)
        sys.exit(1)

    use_hybrid = args.use_taco or args.use_classifier
    hybrid_detector = None

    if use_hybrid:
        try:
            from hybrid_detector import HybridPlasticDetector
            log.info("Hybrid detection pipeline enabled")
        except ImportError:
            log.error("hybrid_detector module not found.")
            sys.exit(1)

    if args.tiny:
        cfg = CFG_PATH_TINY
        weights = WEIGHTS_PATH_TINY
        log.info("Using YOLOv4-tiny (fast mode)")
    else:
        cfg = CFG_PATH
        weights = WEIGHTS_PATH
        log.info("Using full YOLOv4 (high accuracy mode)")

    yolo_detector = PlasticDetector(
        cfg=cfg,
        weights=weights,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        input_size=args.size,
        use_gpu=args.gpu,
        multi_scale=not args.no_multiscale,
        temporal_smooth=not args.no_smooth,
    )

    if use_hybrid:
        try:
            hybrid_detector = HybridPlasticDetector(
                use_taco=args.use_taco,
                taco_weights=args.taco_weights,
                detection_threshold=args.detection_threshold,
                use_classifier=args.use_classifier,
                classifier_weights=args.classifier_weights,
                classification_threshold=args.classification_threshold,
                trashnet_weights=args.trashnet_weights,
                yolo_fallback=yolo_detector,
                frame_skip=args.frame_skip,
            )
            log.info("Hybrid pipeline status: %s", hybrid_detector.pipeline_status)
        except Exception as e:
            log.warning("Failed to initialize hybrid pipeline: %s", e)
            log.warning("Falling back to YOLO-only detection")
            hybrid_detector = None

    active_detector = hybrid_detector if hybrid_detector else yolo_detector

    reporter = None
    if args.send:
        if not HAS_REQUESTS:
            log.error("Install 'requests' to use --send: pip install requests")
            sys.exit(1)
        reporter = BackendReporter(url=args.backend_url, zone_id=args.zone)
        log.info("Backend reporting enabled: %s (zone %s)", args.backend_url, args.zone)

    if args.mode == "webcam":
        run_webcam(active_detector, camera_idx=args.camera, zone=args.zone,
                   reporter=reporter, no_display=args.no_display, save_output=args.save)
    elif args.mode == "video":
        run_video(active_detector, source=args.source, zone=args.zone,
                  reporter=reporter, no_display=args.no_display, save_output=args.save)
    elif args.mode == "image":
        run_image(active_detector, source=args.source, zone=args.zone,
                  reporter=reporter, no_display=args.no_display, save_output=args.save)


if __name__ == "__main__":
    main()
