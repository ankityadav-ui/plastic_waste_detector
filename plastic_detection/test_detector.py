import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detector import (
    PlasticDetector,
    BackendReporter,
    PLASTIC_MAP,
    RESIN_CODES,
    COLORS,
    BASE_DIR,
    CFG_PATH,
    CFG_PATH_TINY,
    WEIGHTS_PATH,
    WEIGHTS_PATH_TINY,
    NAMES_PATH,
)


class TestPlasticMap(unittest.TestCase):

    def test_all_mapped_entries_have_required_fields(self):
        required = {"item_type", "is_plastic", "material", "resin_code", "recyclable", "description"}
        for coco_name, info in PLASTIC_MAP.items():
            for field in required:
                self.assertIn(field, info, f"'{coco_name}' missing field '{field}'")

    def test_all_materials_are_valid(self):
        valid_materials = set(RESIN_CODES.keys())
        for coco_name, info in PLASTIC_MAP.items():
            self.assertIn(info["material"], valid_materials,
                          f"'{coco_name}' has invalid material '{info['material']}'")

    def test_resin_codes_in_range(self):
        for coco_name, info in PLASTIC_MAP.items():
            self.assertIn(info["resin_code"], range(1, 8),
                          f"'{coco_name}' has invalid resin_code {info['resin_code']}")

    def test_key_classes_are_mapped(self):
        for name in ["bottle", "handbag", "cell phone", "fork", "cup"]:
            self.assertIn(name, PLASTIC_MAP, f"Expected '{name}' in PLASTIC_MAP")

    def test_bottle_is_pet(self):
        self.assertEqual(PLASTIC_MAP["bottle"]["material"], "PET")
        self.assertTrue(PLASTIC_MAP["bottle"]["recyclable"])
        self.assertEqual(PLASTIC_MAP["bottle"]["resin_code"], 1)


class TestModelFiles(unittest.TestCase):

    def test_cfg_exists(self):
        self.assertTrue(CFG_PATH.is_file(), f"Missing {CFG_PATH}")

    def test_cfg_tiny_exists(self):
        self.assertTrue(CFG_PATH_TINY.is_file(), f"Missing {CFG_PATH_TINY}")

    def test_weights_exists(self):
        self.assertTrue(WEIGHTS_PATH.is_file(), f"Missing {WEIGHTS_PATH}")

    def test_weights_tiny_exists(self):
        self.assertTrue(WEIGHTS_PATH_TINY.is_file(), f"Missing {WEIGHTS_PATH_TINY}")

    def test_names_exists(self):
        self.assertTrue(NAMES_PATH.is_file(), f"Missing {NAMES_PATH}")

    def test_names_has_80_classes(self):
        with open(NAMES_PATH) as f:
            names = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(names), 80, "COCO should have 80 classes")


class TestDetectorInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = PlasticDetector(
            cfg=CFG_PATH_TINY, weights=WEIGHTS_PATH_TINY, conf_threshold=0.3,
            multi_scale=False, temporal_smooth=False,
        )

    def test_class_names_loaded(self):
        self.assertEqual(len(self.detector.class_names), 80)

    def test_plastic_index_mapping(self):
        self.assertGreater(len(self.detector.index_to_plastic), 0)
        bottle_idx = self.detector.class_names.index("bottle")
        self.assertIn(bottle_idx, self.detector.index_to_plastic)
        info = self.detector.index_to_plastic[bottle_idx]
        self.assertEqual(info["material"], "PET")
        self.assertTrue(info["recyclable"])
        self.assertEqual(info["plastic_type"], "plastic_bottle")

    def test_output_layers(self):
        self.assertGreater(len(self.detector.output_layers), 0)


class TestDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.detector = PlasticDetector(
            cfg=CFG_PATH_TINY, weights=WEIGHTS_PATH_TINY, conf_threshold=0.3,
            multi_scale=False, temporal_smooth=False,
        )

    def test_detect_returns_list(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(blank)
        self.assertIsInstance(results, list)

    def test_detect_result_schema(self):
        test_img = BASE_DIR / "test_images"
        imgs = sorted(test_img.glob("*.jpg")) if test_img.is_dir() else []
        if not imgs:
            self.skipTest("No test images available")
        frame = cv2.imread(str(imgs[0]))
        results = self.detector.detect(frame)
        for det in results:
            self.assertIn("label", det)
            self.assertIn("plastic_type", det)
            self.assertIn("item_type", det)
            self.assertIn("material", det)
            self.assertIn("resin_code", det)
            self.assertIn("recyclable", det)
            self.assertIn("confidence", det)
            self.assertIn("box", det)
            self.assertIsInstance(det["box"], tuple)
            self.assertEqual(len(det["box"]), 4)
            self.assertIsInstance(det["recyclable"], bool)

    def test_draw_does_not_crash(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_dets = [{
            "label": "bottle",
            "plastic_type": "plastic_bottle",
            "item_type": "Plastic Bottle",
            "is_plastic": True,
            "material": "PET",
            "material_name": "Polyethylene Terephthalate",
            "resin_code": 1,
            "recyclable": True,
            "description": "PET bottle",
            "confidence": 0.92,
            "box": (100, 100, 50, 120),
        }]
        result = self.detector.draw(frame, fake_dets)
        self.assertEqual(result.shape, frame.shape)


class TestBackendReporter(unittest.TestCase):

    def test_rate_limiting(self):
        reporter = BackendReporter(cooldown=10.0)
        self.assertTrue(reporter._should_report("plastic_bottle"))
        self.assertFalse(reporter._should_report("plastic_bottle"))
        self.assertTrue(reporter._should_report("plastic_bag"))

    @patch("detector.requests")
    def test_flush_sends_single(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_requests.post.return_value = mock_resp

        reporter = BackendReporter(batch_size=1, cooldown=0)
        reporter.report([{"plastic_type": "plastic_bottle", "confidence": 0.9}])
        self.assertEqual(len(reporter._buffer), 0)

    def test_buffer_accumulates(self):
        reporter = BackendReporter(batch_size=10, cooldown=0)
        reporter._buffer.append({"test": True})
        reporter._buffer.append({"test": True})
        self.assertEqual(len(reporter._buffer), 2)


try:
    from fastapi.testclient import TestClient
    from backend import app, detections as _store

    class TestBackendAPI(unittest.TestCase):

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
                "itemType": "Plastic Bottle",
                "material": "PET",
                "materialName": "Polyethylene Terephthalate",
                "resinCode": 1,
                "recyclable": True,
            }
            r = self.client.post("/report", json=event)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["status"], "accepted")
            self.assertEqual(r.json()["total"], 1)

        def test_report_batch(self):
            events = [
                {"zoneId": "Z-101", "plasticType": "plastic_bottle",
                 "confidence": 0.9, "timestamp": "2026-02-14T12:00:00",
                 "material": "PET", "resinCode": 1, "recyclable": True},
                {"zoneId": "Z-102", "plasticType": "plastic_bag",
                 "confidence": 0.7, "timestamp": "2026-02-14T12:01:00",
                 "material": "LDPE", "resinCode": 4, "recyclable": True},
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
                "plasticType": "plastic_cutlery",
                "confidence": 0.6,
                "timestamp": "2026-02-14T12:00:00",
                "material": "PS",
                "resinCode": 6,
                "recyclable": False,
            }
            self.client.post("/report", json=event)
            r = self.client.get("/stats")
            data = r.json()
            self.assertEqual(data["total_detections"], 1)
            self.assertIn("plastic_cutlery", data["by_type"])
            self.assertIn("by_material", data)
            self.assertIn("recyclable", data)

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
            bad = {"zoneId": "Z-101", "plasticType": "plastic_bottle",
                   "confidence": 5.0, "timestamp": "2026-02-14T12:00:00"}
            r = self.client.post("/report", json=bad)
            self.assertEqual(r.status_code, 422)

except ImportError:
    pass


class TestHybridDetectorImports(unittest.TestCase):

    def test_imports(self):
        import hybrid_detector
        self.assertTrue(hasattr(hybrid_detector, "HybridPlasticDetector"))
        self.assertTrue(hasattr(hybrid_detector, "TACODetector"))
        self.assertTrue(hasattr(hybrid_detector, "ResinClassifier"))
        self.assertTrue(hasattr(hybrid_detector, "TrashNetClassifier"))

    def test_taco_material_mapping(self):
        from hybrid_detector import TACO_TO_MATERIAL
        required = {"item_type", "is_plastic", "material", "resin_code",
                     "recyclable", "description"}
        for cls_name, info in TACO_TO_MATERIAL.items():
            with self.subTest(cls=cls_name):
                self.assertTrue(required <= set(info.keys()),
                                f"Missing keys in {cls_name}: {required - set(info.keys())}")

    def test_taco_classes_length(self):
        from hybrid_detector import TACO_CLASSES
        self.assertEqual(len(TACO_CLASSES), 61)
        self.assertEqual(TACO_CLASSES[0], "__background__")

    def test_resin_labels_match_info(self):
        from hybrid_detector import RESIN_LABELS, RESIN_INFO
        for label in RESIN_LABELS:
            self.assertIn(label, RESIN_INFO)
            self.assertIn("code", RESIN_INFO[label])
            self.assertIn("name", RESIN_INFO[label])
            self.assertIn("recyclable", RESIN_INFO[label])

    def test_trashnet_classes(self):
        from hybrid_detector import TRASHNET_CLASSES
        self.assertEqual(len(TRASHNET_CLASSES), 6)
        self.assertIn("plastic", TRASHNET_CLASSES)
        self.assertIn("glass", TRASHNET_CLASSES)

    def test_has_torch_flag(self):
        from hybrid_detector import HAS_TORCH
        self.assertIsInstance(HAS_TORCH, bool)

    def test_plastic_category_mapping(self):
        from hybrid_detector import HybridPlasticDetector
        self.assertEqual(
            HybridPlasticDetector._get_plastic_category("Plastic Bottle", True),
            "plastic_bottle",
        )
        self.assertEqual(
            HybridPlasticDetector._get_plastic_category("Plastic Bag", True),
            "plastic_bag",
        )
        self.assertEqual(
            HybridPlasticDetector._get_plastic_category("Plastic Straw", True),
            "plastic_straw",
        )
        self.assertEqual(
            HybridPlasticDetector._get_plastic_category("Metal Can", False),
            "non_plastic_waste",
        )


class TestHybridYOLOFallback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.yolo = PlasticDetector(
            cfg=CFG_PATH_TINY, weights=WEIGHTS_PATH_TINY,
            conf_threshold=0.3, multi_scale=False, temporal_smooth=False,
        )

    def test_hybrid_init_without_torch(self):
        from hybrid_detector import HybridPlasticDetector
        hybrid = HybridPlasticDetector(
            use_taco=False, use_classifier=False,
            use_trashnet=False, yolo_fallback=self.yolo,
        )
        self.assertIsNotNone(hybrid)

    def test_hybrid_detect_blank_frame(self):
        from hybrid_detector import HybridPlasticDetector
        hybrid = HybridPlasticDetector(
            use_taco=False, use_classifier=False,
            use_trashnet=False, yolo_fallback=self.yolo,
        )
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        results = hybrid.detect(blank)
        self.assertIsInstance(results, list)

    def test_hybrid_pipeline_status(self):
        from hybrid_detector import HybridPlasticDetector
        hybrid = HybridPlasticDetector(
            use_taco=False, use_classifier=False,
            use_trashnet=False, yolo_fallback=self.yolo,
        )
        status = hybrid.pipeline_status
        self.assertEqual(status["taco_detector"], "disabled")
        self.assertEqual(status["resin_classifier"], "disabled")
        self.assertEqual(status["trashnet_fallback"], "disabled")
        self.assertEqual(status["yolo_fallback"], "active")

    def test_hybrid_draw_doesnt_crash(self):
        from hybrid_detector import HybridPlasticDetector
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = HybridPlasticDetector.draw(blank, [])
        self.assertIsNotNone(result)

    def test_hybrid_draw_with_detection(self):
        from hybrid_detector import HybridPlasticDetector
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = {
            "label": "Clear plastic bottle",
            "plastic_type": "plastic_bottle",
            "item_type": "Plastic Bottle",
            "is_plastic": True,
            "material": "PET",
            "material_name": "Polyethylene Terephthalate",
            "resin_code": 1,
            "recyclable": True,
            "description": "PET bottle",
            "confidence": 0.92,
            "box": (50, 50, 100, 200),
            "detection_source": "taco",
            "classifier_source": "mobilenet_v3",
        }
        result = HybridPlasticDetector.draw(frame, [det])
        self.assertIsNotNone(result)
        self.assertGreater(result.sum(), 0)


class TestDownloadModels(unittest.TestCase):

    def test_import(self):
        import download_models
        self.assertTrue(hasattr(download_models, "MODELS"))
        self.assertTrue(hasattr(download_models, "check_status"))

    def test_model_registry(self):
        import download_models
        for key, group in download_models.MODELS.items():
            with self.subTest(model=key):
                self.assertIn("display_name", group)
                self.assertIn("dir", group)
                self.assertIn("files", group)
                self.assertIsInstance(group["files"], list)
                for f in group["files"]:
                    self.assertIn("filename", f)
                    self.assertIn("description", f)

    def test_check_status(self):
        import download_models
        status = download_models.check_status()
        self.assertIsInstance(status, dict)
        self.assertIn("yolov4", status)
        self.assertIn("taco", status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
