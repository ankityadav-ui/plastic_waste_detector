"""
=============================================================================
 PLASTIC DETECTION — Real-Time Detection Engine
=============================================================================

 Uses YOLOv4-tiny (OpenCV DNN) to detect plastic waste in real-time from
 webcam, video files, or images.  COCO classes are mapped to plastic
 categories.  Optionally reports detections to the FastAPI backend.

 Usage:
   python detector.py --mode webcam
   python detector.py --mode image  --source photo.jpg
   python detector.py --mode video  --source clip.mp4 --send --zone Z-101

=============================================================================
"""

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

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("detector")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & COCO → PLASTIC MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Which COCO class names map to which plastic category.
# Expand these as needed for your environment.
PLASTIC_MAP: dict[str, str] = {
    # ── plastic_bottle ──
    "bottle":      "plastic_bottle",
    "cup":         "plastic_bottle",
    "wine glass":  "plastic_bottle",
    "vase":        "plastic_bottle",
    # ── plastic_bag ──
    "handbag":     "plastic_bag",
    "backpack":    "plastic_bag",
    "suitcase":    "plastic_bag",
    "umbrella":    "plastic_bag",
    # ── plastic_wrapper ──
    "cell phone":  "plastic_wrapper",
    "remote":      "plastic_wrapper",
    "toothbrush":  "plastic_wrapper",
    "scissors":    "plastic_wrapper",
    "book":        "plastic_wrapper",
    "mouse":       "plastic_wrapper",
    "keyboard":    "plastic_wrapper",
}

# Distinct colours per plastic type (BGR)
COLORS: dict[str, tuple[int, int, int]] = {
    "plastic_bottle":  (0, 255, 0),     # green
    "plastic_bag":     (0, 165, 255),    # orange
    "plastic_wrapper": (0, 0, 255),      # red
}

DEFAULT_COLOR = (255, 255, 0)  # cyan fallback

# Model paths (relative to this file)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
CFG_PATH = MODEL_DIR / "yolov4-tiny.cfg"
WEIGHTS_PATH = MODEL_DIR / "yolov4-tiny.weights"
NAMES_PATH = MODEL_DIR / "coco.names"

# Backend defaults
BACKEND_URL = "http://localhost:8000"
REPORT_ENDPOINT = "/report"
BATCH_ENDPOINT = "/report/batch"


# ─────────────────────────────────────────────────────────────────────────────
# YOLO DETECTOR CLASS
# ─────────────────────────────────────────────────────────────────────────────


class PlasticDetector:
    """YOLOv4-tiny detector that filters for plastic-related COCO objects."""

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

        # Temporal smoothing state (tracks recent detections across frames)
        self._history: list[list[dict]] = []
        self._history_len = 5  # number of frames to remember

        # ── Load class names ──
        self.class_names = self._load_names(names)
        log.info("Loaded %d COCO class names", len(self.class_names))

        # ── Build lookup: COCO-index → plastic_type (only mapped classes) ──
        self.index_to_plastic: dict[int, str] = {}
        for idx, name in enumerate(self.class_names):
            if name in PLASTIC_MAP:
                self.index_to_plastic[idx] = PLASTIC_MAP[name]
        log.info(
            "Mapped %d COCO classes → plastic categories", len(self.index_to_plastic)
        )

        # ── Load network ──
        cfg_str, weights_str = str(cfg), str(weights)
        if not os.path.isfile(cfg_str):
            log.error("Config not found: %s", cfg_str)
            sys.exit(1)
        if not os.path.isfile(weights_str):
            log.error("Weights not found: %s", weights_str)
            sys.exit(1)

        self.net = cv2.dnn.readNetFromDarknet(cfg_str, weights_str)
        log.info("Loaded YOLOv4-tiny network")

        # ── GPU acceleration (if available) ──
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            log.info("Using CUDA backend")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # ── Get output layer names ──
        layer_names = self.net.getLayerNames()
        try:
            # OpenCV 4.x
            out_indices = self.net.getUnconnectedOutLayers().flatten()
        except AttributeError:
            out_indices = self.net.getUnconnectedOutLayers()
        self.output_layers = [layer_names[i - 1] for i in out_indices]

    # ── helpers ──

    @staticmethod
    def _load_names(path: str | Path) -> list[str]:
        """Read class names file, one name per line."""
        p = Path(path)
        if not p.is_file():
            log.error("Names file not found: %s", p)
            sys.exit(1)
        with open(p, "r") as f:
            return [line.strip() for line in f if line.strip()]

    # ── preprocessing ──

    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame for better detection in varied lighting.
        Uses CLAHE (adaptive histogram equalization) on the L channel.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ── single-scale inference ──

    def _detect_at_scale(self, frame: np.ndarray, input_size: int) -> tuple[
        list[list[int]], list[float], list[int]
    ]:
        """Run YOLO at a specific input resolution. Returns raw boxes, confs, class_ids."""
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

    # ── temporal smoothing ──

    def _iou(self, box_a: tuple, box_b: tuple) -> float:
        """Compute IoU between two (x, y, w, h) boxes."""
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
        """
        Boost confidence of detections that appear consistently across recent
        frames, and recover detections from recent history if they disappeared
        for only 1-2 frames (reduces flickering).
        """
        self._history.append(detections)
        if len(self._history) > self._history_len:
            self._history.pop(0)

        if len(self._history) < 2:
            return detections

        # Boost current detections that match history
        for det in detections:
            match_count = 0
            for past_frame in self._history[:-1]:
                for past_det in past_frame:
                    if (past_det["plastic_type"] == det["plastic_type"]
                            and self._iou(det["box"], past_det["box"]) > 0.3):
                        match_count += 1
                        break
            # Boost confidence for consistent detections
            if match_count >= 2:
                det["confidence"] = min(det["confidence"] * 1.15, 0.99)
                det["confidence"] = round(det["confidence"], 4)

        # Recover recent detections that vanished (fill gaps)
        current_boxes = [d["box"] for d in detections]
        for past_det in (self._history[-2] if len(self._history) >= 2 else []):
            matched = any(
                self._iou(past_det["box"], cb) > 0.3 for cb in current_boxes
            )
            if not matched and past_det["confidence"] > 0.35:
                # Re-inject with decayed confidence
                recovered = dict(past_det)
                recovered["confidence"] = round(past_det["confidence"] * 0.85, 4)
                detections.append(recovered)

        return detections

    # ── core detection ──

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns a list of dicts:
            {
                "label":        "bottle",
                "plastic_type": "plastic_bottle",
                "confidence":   0.87,
                "box":          (x, y, w, h),   # pixel coords
            }
        """
        # Preprocess for better contrast/lighting
        enhanced = self._preprocess(frame)

        # Multi-scale detection: run at multiple resolutions and merge
        all_boxes: list[list[int]] = []
        all_confs: list[float] = []
        all_cids: list[int] = []

        if self.multi_scale:
            scales = [self.input_size, self.input_size - 192, self.input_size + 192]
            # e.g. 608 → [608, 416, 800]; clamp to [320, 832] and round to 32
            scales = [max(320, min(832, (s // 32) * 32)) for s in scales]
            scales = sorted(set(scales))  # deduplicate
        else:
            scales = [self.input_size]

        for scale in scales:
            b, c, ids = self._detect_at_scale(enhanced, scale)
            all_boxes.extend(b)
            all_confs.extend(c)
            all_cids.extend(ids)

        # Final NMS across all scales
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
                    results.append(
                        {
                            "label": self.class_names[cid],
                            "plastic_type": self.index_to_plastic[cid],
                            "confidence": round(all_confs[i], 4),
                            "box": tuple(all_boxes[i]),
                        }
                    )

        # Temporal smoothing for stable detections across frames
        if self.temporal_smooth:
            results = self._apply_temporal_smoothing(results)

        return results

    # ── drawing ──

    @staticmethod
    def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes and labels on frame (in-place)."""
        for det in detections:
            x, y, bw, bh = det["box"]
            ptype = det["plastic_type"]
            label_text = f'{det["plastic_type"]} ({det["label"]}) {det["confidence"]:.0%}'
            color = COLORS.get(ptype, DEFAULT_COLOR)

            # Rectangle
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
            )
            cv2.rectangle(
                frame, (x, y - th - baseline - 4), (x + tw + 4, y), color, -1,
            )
            cv2.putText(
                frame, label_text, (x + 2, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )
        return frame


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND REPORTER
# ─────────────────────────────────────────────────────────────────────────────


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
        self.cooldown = cooldown  # seconds between reports of the same type
        self._last_report: dict[str, float] = {}
        self._buffer: list[dict] = []

    def _should_report(self, plastic_type: str) -> bool:
        """Rate-limit reports per plastic type to avoid flooding."""
        now = time.time()
        last = self._last_report.get(plastic_type, 0)
        if now - last >= self.cooldown:
            self._last_report[plastic_type] = now
            return True
        return False

    def report(self, detections: list[dict]) -> None:
        """Buffer and send detections to backend."""
        if not HAS_REQUESTS:
            return

        for det in detections:
            if not self._should_report(det["plastic_type"]):
                continue
            event = {
                "zoneId": self.zone_id,
                "plasticType": det["plastic_type"],
                "confidence": det["confidence"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._buffer.append(event)

        # Flush when buffer is full
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        """Send buffered events to backend."""
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
                log.debug("Reported %d event(s) → backend", len(self._buffer))
            else:
                log.warning("Backend responded %d: %s", r.status_code, r.text[:200])
        except requests.ConnectionError:
            log.warning("Backend unreachable at %s", self.url)
        except Exception as exc:
            log.warning("Report failed: %s", exc)
        finally:
            self._buffer.clear()

    def flush_remaining(self) -> None:
        """Call at shutdown to send any leftover events."""
        self._flush()


# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS  (webcam / video / image)
# ─────────────────────────────────────────────────────────────────────────────


def _overlay_info(
    frame: np.ndarray,
    fps: float,
    count: int,
    zone: str,
) -> None:
    """Draw FPS and detection count on the top-left corner."""
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Detections: {count}",
        f"Zone: {zone}",
    ]
    for i, line in enumerate(info_lines):
        y = 22 + i * 22
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
        )


def run_webcam(
    detector: PlasticDetector,
    camera_idx: int = 0,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    """Continuous webcam detection loop."""
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
                log.warning("Failed to read frame — retrying…")
                time.sleep(0.1)
                continue

            detections = detector.detect(frame)
            total_detections += len(detections)
            frame_count += 1

            # Draw
            detector.draw(frame, detections)
            fps = 1.0 / max(time.time() - t0, 1e-6)
            _overlay_info(frame, fps, len(detections), zone)

            # Report to backend
            if reporter and detections:
                reporter.report(detections)

            # Save video output
            if save_output and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_output, fourcc, 20, (w, h))
            if writer:
                writer.write(frame)

            # Display
            if not no_display:
                cv2.imshow("Plastic Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or Esc
                    break
            else:
                # In headless mode, just log periodically
                if frame_count % 100 == 0:
                    log.info(
                        "Frame %d | FPS %.1f | Detections this frame: %d",
                        frame_count, fps, len(detections),
                    )
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        if reporter:
            reporter.flush_remaining()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        log.info(
            "Done — %d frames processed, %d total detections",
            frame_count, total_detections,
        )


def run_video(
    detector: PlasticDetector,
    source: str,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    """Process a video file frame-by-frame."""
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

            # Progress
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
        log.info(
            "Done — %d/%d frames, %d total detections",
            frame_count, total_frames, total_detections,
        )


def run_image(
    detector: PlasticDetector,
    source: str,
    zone: str = "Z-101",
    reporter: Optional[BackendReporter] = None,
    no_display: bool = False,
    save_output: Optional[str] = None,
) -> None:
    """Detect plastic in a single image (or a directory of images)."""
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
        log.info(
            "%s — %d detection(s): %s",
            img_path,
            len(detections),
            [d["plastic_type"] for d in detections] if detections else "none",
        )

        if reporter and detections:
            reporter.report(detections)

        # Save annotated output
        out_path = save_output
        if out_path is None:
            stem = Path(img_path).stem
            out_path = str(BASE_DIR / f"{stem}_detected.jpg")
        cv2.imwrite(out_path, frame)
        log.info("Saved → %s", out_path)

        if not no_display:
            cv2.imshow("Plastic Detection", frame)
            log.info("Press any key to continue (or 'q' to quit)")
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                break

    if reporter:
        reporter.flush_remaining()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plastic Waste Detection Engine (YOLOv4-tiny)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detector.py --mode webcam
  python detector.py --mode image --source photo.jpg
  python detector.py --mode video --source clip.mp4
  python detector.py --mode webcam --send --zone Z-101
  python detector.py --mode video --source clip.mp4 --save output.mp4
        """,
    )
    p.add_argument(
        "--mode", choices=["webcam", "video", "image"], default="webcam",
        help="Detection mode (default: webcam)",
    )
    p.add_argument(
        "--source", type=str, default=None,
        help="Path to image/video file or image directory",
    )
    p.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0)",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    p.add_argument(
        "--nms", type=float, default=0.5,
        help="NMS threshold (default: 0.5)",
    )
    p.add_argument(
        "--size", type=int, default=608,
        help="YOLO input size (default: 608)",
    )
    p.add_argument(
        "--no-multiscale", action="store_true",
        help="Disable multi-scale inference (faster but less accurate)",
    )
    p.add_argument(
        "--no-smooth", action="store_true",
        help="Disable temporal smoothing across frames",
    )
    p.add_argument(
        "--zone", type=str, default="Z-101",
        help="Monitoring zone ID (default: Z-101)",
    )
    p.add_argument(
        "--send", action="store_true",
        help="Send detections to backend API",
    )
    p.add_argument(
        "--backend-url", type=str, default=BACKEND_URL,
        help=f"Backend API URL (default: {BACKEND_URL})",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Run in headless mode (no GUI window)",
    )
    p.add_argument(
        "--save", type=str, default=None,
        help="Save annotated output to file",
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Use CUDA GPU acceleration (requires OpenCV with CUDA)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate source for non-webcam modes
    if args.mode in ("video", "image") and not args.source:
        log.error("--source is required for %s mode", args.mode)
        sys.exit(1)

    # Build detector
    detector = PlasticDetector(
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        input_size=args.size,
        use_gpu=args.gpu,
        multi_scale=not args.no_multiscale,
        temporal_smooth=not args.no_smooth,
    )

    # Build reporter
    reporter = None
    if args.send:
        if not HAS_REQUESTS:
            log.error("Install 'requests' to use --send:  pip install requests")
            sys.exit(1)
        reporter = BackendReporter(url=args.backend_url, zone_id=args.zone)
        log.info("Backend reporting enabled → %s (zone %s)", args.backend_url, args.zone)

    # Dispatch
    if args.mode == "webcam":
        run_webcam(
            detector, camera_idx=args.camera, zone=args.zone,
            reporter=reporter, no_display=args.no_display,
            save_output=args.save,
        )
    elif args.mode == "video":
        run_video(
            detector, source=args.source, zone=args.zone,
            reporter=reporter, no_display=args.no_display,
            save_output=args.save,
        )
    elif args.mode == "image":
        run_image(
            detector, source=args.source, zone=args.zone,
            reporter=reporter, no_display=args.no_display,
            save_output=args.save,
        )


if __name__ == "__main__":
    main()
