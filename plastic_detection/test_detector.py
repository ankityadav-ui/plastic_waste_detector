"""
=============================================================================
 PLASTIC DETECTION — Automated Test Suite
=============================================================================

 Validates that the detector loads correctly, processes frames, maps classes
 properly, and the backend endpoints work.

 Run:
   python test_detector.py

=============================================================================
"""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import (
    PlasticDetector,
    BackendReporter,
    PLASTIC_MAP,
    COLORS,
    BASE_DIR,
    CFG_PATH,
    WEIGHTS_PATH,
    NAMES_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestPlasticMap(unittest.TestCase):
    """Verify the COCO → plastic mapping is correct."""

    def test_all_mapped_types_are_valid(self):
        valid = {"plastic_bottle", "plastic_bag", "plastic_wrapper"}
        for coco_name, ptype in PLASTIC_MAP.items():
            self.assertIn(ptype, valid, f"'{coco_name}' maps to invalid type '{ptype}'")

    def test_all_types_have_colors(self):
        for ptype in set(PLASTIC_MAP.values()):
            self.assertIn(ptype, COLORS, f"No colour defined for '{ptype}'")

    def test_key_classes_are_mapped(self):
        for name in ["bottle", "handbag", "cell phone"]:
            self.assertIn(name, PLASTIC_MAP, f"Expected '{name}' in PLASTIC_MAP")


class TestModelFiles(unittest.TestCase):
    """Ensure all model files exist before loading."""

    def test_cfg_exists(self):
        self.assertTrue(CFG_PATH.is_file(), f"Missing {CFG_PATH}")

    def test_weights_exists(self):
        self.assertTrue(WEIGHTS_PATH.is_file(), f"Missing {WEIGHTS_PATH}")

    def test_names_exists(self):
        self.assertTrue(NAMES_PATH.is_file(), f"Missing {NAMES_PATH}")

    def test_names_has_80_classes(self):
        with open(NAMES_PATH) as f:
            names = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(names), 80, "COCO should have 80 classes")


class TestDetectorInit(unittest.TestCase):
    """Test detector instantiation."""

    @classmethod
    def setUpClass(cls):
        cls.detector = PlasticDetector(conf_threshold=0.3)

    def test_class_names_loaded(self):
        self.assertEqual(len(self.detector.class_names), 80)

    def test_plastic_index_mapping(self):
        self.assertGreater(len(self.detector.index_to_plastic), 0)
        # "bottle" is class index 39 in COCO
        bottle_idx = self.detector.class_names.index("bottle")
        self.assertIn(bottle_idx, self.detector.index_to_plastic)
        self.assertEqual(self.detector.index_to_plastic[bottle_idx], "plastic_bottle")

    def test_output_layers(self):
        self.assertGreater(len(self.detector.output_layers), 0)


class TestDetection(unittest.TestCase):
    """Test the detection pipeline on synthetic frames."""

    @classmethod
    def setUpClass(cls):
        cls.detector = PlasticDetector(conf_threshold=0.3)

    def test_detect_returns_list(self):
        """A blank frame should return an empty list (no objects)."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(blank)
        self.assertIsInstance(results, list)

    def test_detect_result_schema(self):
        """If any detection occurs, validate dict schema."""
        # Use a real-ish test: load a test image if available
        test_img = BASE_DIR / "test_images"
        imgs = sorted(test_img.glob("*.jpg")) if test_img.is_dir() else []
        if not imgs:
            self.skipTest("No test images available")
        frame = cv2.imread(str(imgs[0]))
        results = self.detector.detect(frame)
        for det in results:
            self.assertIn("label", det)
            self.assertIn("plastic_type", det)
            self.assertIn("confidence", det)
            self.assertIn("box", det)
            self.assertIsInstance(det["box"], tuple)
            self.assertEqual(len(det["box"]), 4)

    def test_draw_does_not_crash(self):
        """Drawing on a frame should not raise."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_dets = [
            {
                "label": "bottle",
                "plastic_type": "plastic_bottle",
                "confidence": 0.92,
                "box": (100, 100, 50, 120),
            }
        ]
        result = self.detector.draw(frame, fake_dets)
        self.assertEqual(result.shape, frame.shape)


class TestBackendReporter(unittest.TestCase):
    """Test the BackendReporter logic."""

    def test_rate_limiting(self):
        reporter = BackendReporter(cooldown=10.0)
        # First call should pass
        self.assertTrue(reporter._should_report("plastic_bottle"))
        # Immediate second call should be rate-limited
        self.assertFalse(reporter._should_report("plastic_bottle"))
        # Different type should still pass
        self.assertTrue(reporter._should_report("plastic_bag"))

    @patch("detector.requests")
    def test_flush_sends_single(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_requests.post.return_value = mock_resp

        reporter = BackendReporter(batch_size=1, cooldown=0)
        reporter.report([
            {"plastic_type": "plastic_bottle", "confidence": 0.9}
        ])
        # Should have flushed after 1 event
        self.assertEqual(len(reporter._buffer), 0)

    def test_buffer_accumulates(self):
        reporter = BackendReporter(batch_size=10, cooldown=0)
        reporter._buffer.append({"test": True})
        reporter._buffer.append({"test": True})
        self.assertEqual(len(reporter._buffer), 2)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND API TESTS (if FastAPI test client available)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from fastapi.testclient import TestClient
    from backend import app, detections as _store

    class TestBackendAPI(unittest.TestCase):
        """Test FastAPI endpoints."""

        @classmethod
        def setUpClass(cls):
            cls.client = TestClient(app)

        def setUp(self):
            _store.clear()

        def test_root(self):
            r = self.client.get("/")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "online")

        def test_health(self):
            r = self.client.get("/health")
            self.assertEqual(r.status_code, 200)
            self.assertIn("status", r.json())

        def test_report_single(self):
            event = {
                "zoneId": "Z-101",
                "plasticType": "plastic_bottle",
                "confidence": 0.85,
                "timestamp": "2026-02-14T12:00:00",
            }
            r = self.client.post("/report", json=event)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "accepted")
            self.assertEqual(r.json()["total"], 1)

        def test_report_batch(self):
            events = [
                {"zoneId": "Z-101", "plasticType": "plastic_bottle",
                 "confidence": 0.9, "timestamp": "2026-02-14T12:00:00"},
                {"zoneId": "Z-102", "plasticType": "plastic_bag",
                 "confidence": 0.7, "timestamp": "2026-02-14T12:01:00"},
            ]
            r = self.client.post("/report/batch", json=events)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["count"], 2)

        def test_get_detections_empty(self):
            r = self.client.get("/detections")
            self.assertEqual(r.json()["total"], 0)

        def test_get_detections_filtered(self):
            event = {
                "zoneId": "Z-101",
                "plasticType": "plastic_bottle",
                "confidence": 0.85,
                "timestamp": "2026-02-14T12:00:00",
            }
            self.client.post("/report", json=event)
            r = self.client.get("/detections?zone=Z-101")
            self.assertEqual(r.json()["total"], 1)
            r2 = self.client.get("/detections?zone=Z-999")
            self.assertEqual(r2.json()["total"], 0)

        def test_stats(self):
            event = {
                "zoneId": "Z-101",
                "plasticType": "plastic_wrapper",
                "confidence": 0.6,
                "timestamp": "2026-02-14T12:00:00",
            }
            self.client.post("/report", json=event)
            r = self.client.get("/stats")
            data = r.json()
            self.assertEqual(data["total_detections"], 1)
            self.assertIn("plastic_wrapper", data["by_type"])

        def test_clear(self):
            event = {
                "zoneId": "Z-101",
                "plasticType": "plastic_bottle",
                "confidence": 0.85,
                "timestamp": "2026-02-14T12:00:00",
            }
            self.client.post("/report", json=event)
            r = self.client.delete("/detections")
            self.assertEqual(r.json()["status"], "cleared")
            r2 = self.client.get("/detections")
            self.assertEqual(r2.json()["total"], 0)

        def test_invalid_event(self):
            bad = {"zoneId": "Z-101", "plasticType": "invalid_type",
                   "confidence": 0.5, "timestamp": "2026-02-14T12:00:00"}
            r = self.client.post("/report", json=bad)
            self.assertEqual(r.status_code, 422)

except ImportError:
    pass  # FastAPI not installed — skip API tests


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Plastic Detection — Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
