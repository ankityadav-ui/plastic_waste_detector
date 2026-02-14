import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("hybrid_detector")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TACO_WEIGHTS_DIR = MODEL_DIR / "taco"
CLASSIFIER_WEIGHTS_DIR = MODEL_DIR / "classifier"
TRASHNET_WEIGHTS_DIR = MODEL_DIR / "trashnet"

TACO_SUPERCATEGORIES = [
    "__background__", "Plastic bag & wrapper", "Bottle", "Bottle cap",
    "Can", "Container", "Cup", "Lid", "Straw", "Cigarette", "Other litter",
]

TACO_CLASSES = [
    "__background__", "Aluminium foil", "Battery", "Aluminium blister pack",
    "Carded blister pack", "Other plastic bottle", "Clear plastic bottle",
    "Glass bottle", "Plastic bottle cap", "Metal bottle cap", "Broken glass",
    "Food Can", "Aerosol", "Drink can", "Toilet tube", "Other carton",
    "Egg carton", "Drink carton", "Corrugated carton", "Meal carton",
    "Pizza box", "Paper cup", "Disposable plastic cup", "Foam cup",
    "Glass cup", "Other plastic cup", "Food waste", "Glass jar",
    "Plastic lid", "Metal lid", "Other plastic", "Magazine paper",
    "Tissues", "Wrapping paper", "Normal paper", "Paper bag",
    "Plastified paper bag", "Plastic film", "Six pack rings", "Garbage bag",
    "Other plastic wrapper", "Single-use carrier bag", "Polypropylene bag",
    "Crisp packet", "Spread tub", "Tupperware", "Disposable food container",
    "Foam food container", "Other plastic container", "Plastic gloves",
    "Plastic utensils", "Pop tab", "Rope & strings", "Scrap metal",
    "Shoe", "Squeezable tube", "Plastic straw", "Paper straw",
    "Styrofoam piece", "Unlabeled litter", "Cigarette",
]

TACO_TO_MATERIAL: dict[str, dict] = {
    "Other plastic bottle": {
        "item_type": "Plastic Bottle", "is_plastic": True, "material": "HDPE",
        "resin_code": 2, "recyclable": True, "description": "Opaque plastic bottle — likely HDPE",
    },
    "Clear plastic bottle": {
        "item_type": "Plastic Bottle", "is_plastic": True, "material": "PET",
        "resin_code": 1, "recyclable": True, "description": "Clear PET bottle",
    },
    "Glass bottle": {
        "item_type": "Glass Bottle", "is_plastic": False, "material": "GLASS",
        "resin_code": 0, "recyclable": True, "description": "Glass bottle — not plastic",
    },
    "Plastic bottle cap": {
        "item_type": "Plastic Bottle Cap", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": True, "description": "Polypropylene bottle cap",
    },
    "Metal bottle cap": {
        "item_type": "Metal Bottle Cap", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Metal bottle cap — not plastic",
    },
    "Single-use carrier bag": {
        "item_type": "Plastic Carrier Bag", "is_plastic": True, "material": "LDPE",
        "resin_code": 4, "recyclable": True, "description": "LDPE single-use carrier bag",
    },
    "Polypropylene bag": {
        "item_type": "Polypropylene Bag", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": True, "description": "Polypropylene woven bag",
    },
    "Garbage bag": {
        "item_type": "Garbage Bag", "is_plastic": True, "material": "LDPE",
        "resin_code": 4, "recyclable": False, "description": "LDPE garbage bag — contaminated",
    },
    "Paper bag": {
        "item_type": "Paper Bag", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Paper bag — not plastic",
    },
    "Plastified paper bag": {
        "item_type": "Plastified Paper Bag", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": False, "description": "Paper bag with plastic lining",
    },
    "Plastic film": {
        "item_type": "Plastic Film", "is_plastic": True, "material": "LDPE",
        "resin_code": 4, "recyclable": True, "description": "LDPE plastic film/wrap",
    },
    "Other plastic wrapper": {
        "item_type": "Plastic Wrapper", "is_plastic": True, "material": "LDPE",
        "resin_code": 4, "recyclable": False, "description": "Mixed plastic wrapper",
    },
    "Crisp packet": {
        "item_type": "Crisp/Chip Packet", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": False, "description": "Multi-layer metallised film",
    },
    "Six pack rings": {
        "item_type": "Six Pack Rings", "is_plastic": True, "material": "LDPE",
        "resin_code": 4, "recyclable": True, "description": "LDPE six-pack rings",
    },
    "Disposable plastic cup": {
        "item_type": "Disposable Plastic Cup", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": True, "description": "Polypropylene disposable cup",
    },
    "Foam cup": {
        "item_type": "Foam/Styrofoam Cup", "is_plastic": True, "material": "PS",
        "resin_code": 6, "recyclable": False, "description": "Expanded polystyrene cup",
    },
    "Glass cup": {
        "item_type": "Glass Cup", "is_plastic": False, "material": "GLASS",
        "resin_code": 0, "recyclable": True, "description": "Glass cup — not plastic",
    },
    "Other plastic cup": {
        "item_type": "Plastic Cup", "is_plastic": True, "material": "PS",
        "resin_code": 6, "recyclable": False, "description": "Polystyrene cup",
    },
    "Paper cup": {
        "item_type": "Paper Cup", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": False, "description": "Paper cup with plastic lining",
    },
    "Disposable food container": {
        "item_type": "Disposable Food Container", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": True, "description": "Polypropylene food container",
    },
    "Foam food container": {
        "item_type": "Foam Food Container", "is_plastic": True, "material": "PS",
        "resin_code": 6, "recyclable": False, "description": "Expanded polystyrene container",
    },
    "Other plastic container": {
        "item_type": "Plastic Container", "is_plastic": True, "material": "HDPE",
        "resin_code": 2, "recyclable": True, "description": "HDPE container",
    },
    "Spread tub": {
        "item_type": "Spread Tub", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": True, "description": "Polypropylene tub",
    },
    "Tupperware": {
        "item_type": "Tupperware", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": True, "description": "Polypropylene reusable container",
    },
    "Plastic lid": {
        "item_type": "Plastic Lid", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": True, "description": "Polypropylene lid",
    },
    "Metal lid": {
        "item_type": "Metal Lid", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Metal lid — not plastic",
    },
    "Plastic utensils": {
        "item_type": "Plastic Cutlery", "is_plastic": True, "material": "PS",
        "resin_code": 6, "recyclable": False, "description": "Polystyrene single-use cutlery",
    },
    "Plastic straw": {
        "item_type": "Plastic Straw", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": False, "description": "Polypropylene straw",
    },
    "Paper straw": {
        "item_type": "Paper Straw", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Paper straw — not plastic",
    },
    "Squeezable tube": {
        "item_type": "Squeezable Tube", "is_plastic": True, "material": "LDPE",
        "resin_code": 4, "recyclable": False, "description": "LDPE squeeze tube — mixed materials",
    },
    "Plastic gloves": {
        "item_type": "Plastic Gloves", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": False, "description": "Latex/nitrile gloves",
    },
    "Styrofoam piece": {
        "item_type": "Styrofoam Piece", "is_plastic": True, "material": "PS",
        "resin_code": 6, "recyclable": False, "description": "Expanded polystyrene fragment",
    },
    "Other plastic": {
        "item_type": "Other Plastic", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": False, "description": "Unidentified plastic",
    },
    "Rope & strings": {
        "item_type": "Rope/String", "is_plastic": True, "material": "PP",
        "resin_code": 5, "recyclable": False, "description": "Polypropylene rope/string",
    },
    "Aluminium foil": {
        "item_type": "Aluminium Foil", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Aluminium foil",
    },
    "Battery": {
        "item_type": "Battery", "is_plastic": False, "material": "HAZARDOUS",
        "resin_code": 0, "recyclable": False, "description": "Battery — hazardous waste",
    },
    "Aluminium blister pack": {
        "item_type": "Blister Pack", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": False, "description": "Plastic/aluminium blister pack",
    },
    "Carded blister pack": {
        "item_type": "Carded Blister Pack", "is_plastic": True, "material": "PVC",
        "resin_code": 3, "recyclable": False, "description": "PVC blister pack",
    },
    "Broken glass": {
        "item_type": "Broken Glass", "is_plastic": False, "material": "GLASS",
        "resin_code": 0, "recyclable": False, "description": "Broken glass — safety hazard",
    },
    "Food Can": {
        "item_type": "Food Can", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Tin/steel food can",
    },
    "Aerosol": {
        "item_type": "Aerosol Can", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Aerosol can",
    },
    "Drink can": {
        "item_type": "Drink Can", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Aluminium drink can",
    },
    "Glass jar": {
        "item_type": "Glass Jar", "is_plastic": False, "material": "GLASS",
        "resin_code": 0, "recyclable": True, "description": "Glass jar",
    },
    "Pop tab": {
        "item_type": "Pop Tab", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Aluminium pop tab",
    },
    "Scrap metal": {
        "item_type": "Scrap Metal", "is_plastic": False, "material": "METAL",
        "resin_code": 0, "recyclable": True, "description": "Scrap metal",
    },
    "Toilet tube": {
        "item_type": "Toilet Tube", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Cardboard tube",
    },
    "Other carton": {
        "item_type": "Carton", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Carton/cardboard",
    },
    "Egg carton": {
        "item_type": "Egg Carton", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Paper/cardboard egg carton",
    },
    "Drink carton": {
        "item_type": "Drink Carton", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": True, "description": "Tetra Pak drink carton",
    },
    "Corrugated carton": {
        "item_type": "Corrugated Carton", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Corrugated cardboard",
    },
    "Meal carton": {
        "item_type": "Meal Carton", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Cardboard meal carton",
    },
    "Pizza box": {
        "item_type": "Pizza Box", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": False, "description": "Pizza box — greasy",
    },
    "Magazine paper": {
        "item_type": "Magazine Paper", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Magazine paper",
    },
    "Tissues": {
        "item_type": "Tissues", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": False, "description": "Tissues — contaminated",
    },
    "Wrapping paper": {
        "item_type": "Wrapping Paper", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": False, "description": "Wrapping paper — coated",
    },
    "Normal paper": {
        "item_type": "Paper", "is_plastic": False, "material": "PAPER",
        "resin_code": 0, "recyclable": True, "description": "Normal paper",
    },
    "Food waste": {
        "item_type": "Food Waste", "is_plastic": False, "material": "ORGANIC",
        "resin_code": 0, "recyclable": False, "description": "Food waste — compostable",
    },
    "Shoe": {
        "item_type": "Shoe", "is_plastic": False, "material": "OTHER",
        "resin_code": 0, "recyclable": False, "description": "Shoe — not recyclable",
    },
    "Cigarette": {
        "item_type": "Cigarette Butt", "is_plastic": True, "material": "OTHER",
        "resin_code": 7, "recyclable": False, "description": "Cigarette filter (cellulose acetate)",
    },
    "Unlabeled litter": {
        "item_type": "Unlabeled Litter", "is_plastic": False, "material": "OTHER",
        "resin_code": 0, "recyclable": False, "description": "Unidentified litter",
    },
}

TRASHNET_CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
RESIN_LABELS = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "OTHER"]

RESIN_INFO = {
    "PET":   {"code": 1, "name": "Polyethylene Terephthalate", "recyclable": True},
    "HDPE":  {"code": 2, "name": "High-Density Polyethylene",  "recyclable": True},
    "PVC":   {"code": 3, "name": "Polyvinyl Chloride",         "recyclable": False},
    "LDPE":  {"code": 4, "name": "Low-Density Polyethylene",   "recyclable": True},
    "PP":    {"code": 5, "name": "Polypropylene",              "recyclable": True},
    "PS":    {"code": 6, "name": "Polystyrene",                "recyclable": False},
    "OTHER": {"code": 7, "name": "Other Plastics",             "recyclable": False},
}

COCO_WASTE_CLASS_IDS = {
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 27: "backpack", 31: "handbag", 33: "suitcase",
    67: "cell phone", 72: "tv", 73: "laptop", 82: "refrigerator", 84: "book",
    86: "vase", 80: "toaster", 79: "oven",
}

HAS_TORCH = False
HAS_TORCHVISION = False
HAS_DETECTRON2 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import torchvision
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    pass

try:
    import detectron2
    HAS_DETECTRON2 = True
except ImportError:
    pass


def _no_grad_decorator(func):
    return func


def _torch_no_grad():
    if HAS_TORCH:
        return torch.no_grad()
    return _no_grad_decorator


def _get_device() -> "torch.device":
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for hybrid detection")
    if torch.cuda.is_available():
        log.info("Using CUDA GPU for inference")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log.info("Using Apple MPS for inference")
        return torch.device("mps")
    else:
        log.info("Using CPU for inference")
        return torch.device("cpu")


class TACODetector:
    """Waste detector using SSDLite320-MobileNetV3 (torchvision)."""

    MAX_INPUT_DIM = 320

    def __init__(
        self,
        weights_path: Optional[str] = None,
        num_classes: int = 61,
        confidence_threshold: float = 0.5,
        device: Optional["torch.device"] = None,
        max_input_dim: int = 320,
    ):
        if not HAS_TORCH or not HAS_TORCHVISION:
            raise ImportError("PyTorch and torchvision are required.")

        self.confidence_threshold = confidence_threshold
        self.device = device or _get_device()
        self._model = None
        self._weights_path = weights_path
        self._num_classes = num_classes
        self._loaded = False
        self.MAX_INPUT_DIM = max_input_dim

    def _lazy_load(self) -> None:
        if self._loaded:
            return

        from torchvision.models.detection import (
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights,
        )

        if self._weights_path and os.path.isfile(self._weights_path):
            log.info("Loading TACO weights from %s", self._weights_path)
            self._model = ssdlite320_mobilenet_v3_large(num_classes=self._num_classes)
            state_dict = torch.load(
                self._weights_path, map_location=self.device, weights_only=True
            )
            self._model.load_state_dict(state_dict, strict=False)
            self._use_taco_classes = True
        else:
            if self._weights_path:
                log.warning("TACO weights not found at %s — using COCO-pretrained", self._weights_path)
            else:
                log.info("Using COCO-pretrained SSDLite320-MobileNetV3")
            self._model = ssdlite320_mobilenet_v3_large(
                weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            )
            self._use_taco_classes = False

        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        log.info("SSDLite320-MobileNetV3 detector loaded on %s", self.device)

    @_torch_no_grad()
    def detect(self, frame: np.ndarray) -> list[dict]:
        self._lazy_load()

        orig_h, orig_w = frame.shape[:2]
        scale = 1.0
        if max(orig_h, orig_w) > self.MAX_INPUT_DIM:
            scale = self.MAX_INPUT_DIM / max(orig_h, orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        predictions = self._model([tensor])[0]

        results = []
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        inv_scale = 1.0 / scale if scale != 1.0 else 1.0

        for i in range(len(scores)):
            if scores[i] < self.confidence_threshold:
                continue

            bx1, by1, bx2, by2 = boxes[i]
            if inv_scale != 1.0:
                bx1 *= inv_scale
                by1 *= inv_scale
                bx2 *= inv_scale
                by2 *= inv_scale
            x1, y1 = int(bx1), int(by1)
            w, h = int(bx2 - bx1), int(by2 - by1)

            class_id = int(labels[i])

            if self._use_taco_classes:
                if class_id < len(TACO_CLASSES):
                    label = TACO_CLASSES[class_id]
                else:
                    label = f"class_{class_id}"
            else:
                if class_id not in COCO_WASTE_CLASS_IDS:
                    continue
                label = COCO_WASTE_CLASS_IDS[class_id]

            results.append({
                "label": label,
                "confidence": float(scores[i]),
                "box": (x1, y1, w, h),
                "class_id": class_id,
                "source": "taco" if self._use_taco_classes else "coco_maskrcnn",
            })

        return results

    @property
    def is_available(self) -> bool:
        return HAS_TORCH and HAS_TORCHVISION


class ResinClassifier:
    """Classifies cropped plastic waste images into 7 resin types using MobileNetV3-Small."""

    INPUT_SIZE = 224

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        device: Optional["torch.device"] = None,
    ):
        if not HAS_TORCH or not HAS_TORCHVISION:
            raise ImportError("PyTorch and torchvision are required.")

        self.confidence_threshold = confidence_threshold
        self.device = device or _get_device()
        self._weights_path = weights_path
        self._model = None
        self._loaded = False
        self._has_trained_weights = False

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _lazy_load(self) -> None:
        if self._loaded:
            return

        from torchvision.models import mobilenet_v3_small

        if self._weights_path and os.path.isfile(self._weights_path):
            log.info("Loading resin classifier weights from %s", self._weights_path)
            self._model = mobilenet_v3_small(num_classes=len(RESIN_LABELS))
            state_dict = torch.load(
                self._weights_path, map_location=self.device, weights_only=True
            )
            self._model.load_state_dict(state_dict, strict=False)
            self._has_trained_weights = True
        else:
            if self._weights_path:
                log.warning("Classifier weights not found at %s — using ImageNet backbone", self._weights_path)
            from torchvision.models import MobileNet_V3_Small_Weights
            self._model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_features = self._model.classifier[3].in_features
            self._model.classifier[3] = torch.nn.Linear(in_features, len(RESIN_LABELS))
            self._has_trained_weights = False

        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        log.info("Resin classifier (MobileNetV3-Small) loaded on %s (trained=%s)",
                 self.device, self._has_trained_weights)

    @_torch_no_grad()
    def classify(self, crop: np.ndarray) -> dict:
        self._lazy_load()

        if not self._has_trained_weights:
            return self._default_result()

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return self._default_result()

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)

        logits = self._model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        material = RESIN_LABELS[top_idx]

        if top_conf < self.confidence_threshold:
            return {
                "material": material,
                "material_name": RESIN_INFO[material]["name"],
                "resin_code": RESIN_INFO[material]["code"],
                "recyclable": RESIN_INFO[material]["recyclable"],
                "confidence": top_conf,
                "source": "mobilenet_v3",
                "reliable": False,
            }

        info = RESIN_INFO[material]
        return {
            "material": material,
            "material_name": info["name"],
            "resin_code": info["code"],
            "recyclable": info["recyclable"],
            "confidence": top_conf,
            "source": "mobilenet_v3",
            "reliable": True,
        }

    @staticmethod
    def _default_result() -> dict:
        return {
            "material": "OTHER", "material_name": "Other Plastics",
            "resin_code": 7, "recyclable": False, "confidence": 0.0,
            "source": "default", "reliable": False,
        }

    @property
    def is_available(self) -> bool:
        return HAS_TORCH and HAS_TORCHVISION


class TrashNetClassifier:
    """6-class waste classifier (glass, paper, cardboard, plastic, metal, trash) using MobileNetV3-Small."""

    INPUT_SIZE = 224

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.6,
        device: Optional["torch.device"] = None,
    ):
        if not HAS_TORCH or not HAS_TORCHVISION:
            raise ImportError("PyTorch and torchvision are required.")

        self.confidence_threshold = confidence_threshold
        self.device = device or _get_device()
        self._weights_path = weights_path
        self._model = None
        self._loaded = False
        self._has_trained_weights = False

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _lazy_load(self) -> None:
        if self._loaded:
            return

        from torchvision.models import mobilenet_v3_small

        if self._weights_path and os.path.isfile(self._weights_path):
            log.info("Loading TrashNet weights from %s", self._weights_path)
            self._model = mobilenet_v3_small(num_classes=len(TRASHNET_CLASSES))
            state_dict = torch.load(
                self._weights_path, map_location=self.device, weights_only=True
            )
            self._model.load_state_dict(state_dict, strict=False)
            self._has_trained_weights = True
        else:
            if self._weights_path:
                log.warning("TrashNet weights not found at %s — using ImageNet backbone", self._weights_path)
            from torchvision.models import MobileNet_V3_Small_Weights
            self._model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            in_features = self._model.classifier[3].in_features
            self._model.classifier[3] = torch.nn.Linear(in_features, len(TRASHNET_CLASSES))
            self._has_trained_weights = False

        self._model.to(self.device)
        self._model.eval()
        self._loaded = True
        log.info("TrashNet classifier (MobileNetV3-Small) loaded on %s (trained=%s)",
                 self.device, self._has_trained_weights)

    @_torch_no_grad()
    def classify(self, crop: np.ndarray) -> dict:
        self._lazy_load()

        if not self._has_trained_weights:
            return {"waste_type": "trash", "is_plastic": False, "confidence": 0.0,
                    "source": "trashnet", "reliable": False}

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return {"waste_type": "trash", "is_plastic": False, "confidence": 0.0,
                    "source": "trashnet", "reliable": False}

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(self.device)

        logits = self._model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        waste_type = TRASHNET_CLASSES[top_idx]

        is_plastic = waste_type == "plastic"
        reliable = self._has_trained_weights and top_conf >= self.confidence_threshold

        return {
            "waste_type": waste_type,
            "is_plastic": is_plastic,
            "confidence": top_conf if self._has_trained_weights else 0.0,
            "source": "trashnet",
            "reliable": reliable,
        }

    @property
    def is_available(self) -> bool:
        return HAS_TORCH and HAS_TORCHVISION


class HybridPlasticDetector:
    """Three-stage hybrid detection pipeline for plastic waste classification."""

    def __init__(
        self,
        use_taco: bool = True,
        taco_weights: Optional[str] = None,
        detection_threshold: float = 0.5,
        use_classifier: bool = True,
        classifier_weights: Optional[str] = None,
        classification_threshold: float = 0.7,
        use_trashnet: bool = True,
        trashnet_weights: Optional[str] = None,
        yolo_fallback=None,
        device: Optional["torch.device"] = None,
        frame_skip: int = 2,
    ):
        self.use_taco = use_taco
        self.use_classifier = use_classifier
        self.use_trashnet = use_trashnet
        self._yolo_fallback = yolo_fallback
        self._frame_skip = max(1, frame_skip)
        self._frame_counter = 0
        self._cached_results: list[dict] = []

        if taco_weights is None:
            taco_weights = str(TACO_WEIGHTS_DIR / "taco_maskrcnn.pth")
        if classifier_weights is None:
            classifier_weights = str(CLASSIFIER_WEIGHTS_DIR / "resin_classifier.pth")
        if trashnet_weights is None:
            trashnet_weights = str(TRASHNET_WEIGHTS_DIR / "trashnet_resnet50.pth")

        self._device = device
        if device is None and HAS_TORCH:
            self._device = _get_device()

        self._taco: Optional[TACODetector] = None
        self._classifier: Optional[ResinClassifier] = None
        self._trashnet: Optional[TrashNetClassifier] = None

        if self.use_taco and HAS_TORCH and HAS_TORCHVISION:
            try:
                self._taco = TACODetector(
                    weights_path=taco_weights,
                    confidence_threshold=detection_threshold,
                    device=self._device,
                )
                log.info("Stage 1: SSDLite320-MobileNetV3 detector enabled")
            except Exception as e:
                log.warning("Stage 1: Failed to initialize TACO detector: %s", e)
                self._taco = None
        else:
            log.info("Stage 1: TACO detector disabled (%s)",
                     "user choice" if not self.use_taco else "PyTorch not installed")

        if self.use_classifier and HAS_TORCH and HAS_TORCHVISION:
            try:
                self._classifier = ResinClassifier(
                    weights_path=classifier_weights,
                    confidence_threshold=classification_threshold,
                    device=self._device,
                )
                log.info("Stage 2: MobileNetV3-Small resin classifier enabled")
            except Exception as e:
                log.warning("Stage 2: Failed to initialize resin classifier: %s", e)
                self._classifier = None
        else:
            log.info("Stage 2: Resin classifier disabled (%s)",
                     "user choice" if not self.use_classifier else "PyTorch not installed")

        if self.use_trashnet and HAS_TORCH and HAS_TORCHVISION:
            try:
                self._trashnet = TrashNetClassifier(
                    weights_path=trashnet_weights,
                    confidence_threshold=0.6,
                    device=self._device,
                )
                log.info("Stage 2b: TrashNet fallback classifier enabled")
            except Exception as e:
                log.warning("Stage 2b: Failed to initialize TrashNet: %s", e)
                self._trashnet = None

        self._log_pipeline_status()

    def _log_pipeline_status(self) -> None:
        stages = []
        if self._taco:
            stages.append("SSDLite320-MobileNetV3")
        if self._classifier:
            stages.append("MobileNetV3-Classifier")
        if self._trashnet:
            stages.append("TrashNet-Fallback")
        if self._yolo_fallback:
            stages.append("YOLOv4-Fallback")

        if stages:
            log.info("Hybrid pipeline: %s", " → ".join(stages))
        else:
            log.warning("No hybrid models available — using YOLO-only mode")

    def detect(self, frame: np.ndarray) -> list[dict]:
        self._frame_counter += 1

        if self._frame_skip > 1 and (self._frame_counter % self._frame_skip != 1):
            return self._cached_results

        raw_detections = self._stage1_detect(frame)

        if not raw_detections:
            self._cached_results = []
            return []

        results = []
        for det in raw_detections:
            result = self._stage2_classify(frame, det)
            results.append(result)

        self._cached_results = results
        return results

    def _stage1_detect(self, frame: np.ndarray) -> list[dict]:
        if self._taco:
            try:
                detections = self._taco.detect(frame)
                if detections:
                    log.debug("Stage 1: TACO detected %d objects", len(detections))
                    return detections
            except Exception as e:
                log.warning("Stage 1: TACO detection failed: %s", e)

        if self._yolo_fallback:
            try:
                yolo_results = self._yolo_fallback.detect(frame)
                raw = []
                for det in yolo_results:
                    raw.append({
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "box": det["box"],
                        "class_id": -1,
                        "source": "yolo_fallback",
                        "_yolo_material": det.get("material"),
                        "_yolo_info": det,
                    })
                if raw:
                    log.debug("Stage 1: YOLO fallback detected %d objects", len(raw))
                    return raw
            except Exception as e:
                log.warning("Stage 1: YOLO fallback failed: %s", e)

        return []

    def _stage2_classify(self, frame: np.ndarray, detection: dict) -> dict:
        label = detection["label"]
        confidence = detection["confidence"]
        box = detection["box"]
        source = detection.get("source", "unknown")

        x, y, w, h = box
        fh, fw = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)
        crop = frame[y1:y2, x1:x2]

        base_info = self._get_base_material_info(label, source)

        classifier_result = None
        if self._classifier and base_info.get("is_plastic", True):
            if confidence >= 0.3 and crop.size > 0:
                try:
                    classifier_result = self._classifier.classify(crop)
                    if classifier_result.get("reliable"):
                        log.debug("Stage 2: Classified '%s' as %s (%.0f%%)",
                                  label, classifier_result["material"],
                                  classifier_result["confidence"] * 100)
                except Exception as e:
                    log.debug("Stage 2: Classifier failed for '%s': %s", label, e)

        trashnet_result = None
        if self._trashnet and crop.size > 0 and classifier_result is None:
            try:
                trashnet_result = self._trashnet.classify(crop)
                if trashnet_result.get("reliable"):
                    log.debug("Stage 2b: TrashNet classified '%s' as %s",
                              label, trashnet_result["waste_type"])
            except Exception as e:
                log.debug("Stage 2b: TrashNet failed for '%s': %s", label, e)

        return self._aggregate_result(detection, base_info, classifier_result, trashnet_result)

    def _get_base_material_info(self, label: str, source: str) -> dict:
        if label in TACO_TO_MATERIAL:
            return TACO_TO_MATERIAL[label].copy()

        return {
            "item_type": label.replace("_", " ").title(),
            "is_plastic": True,
            "material": "OTHER",
            "resin_code": 7,
            "recyclable": False,
            "description": f"Detected as '{label}' — material unknown",
        }

    def _aggregate_result(
        self,
        detection: dict,
        base_info: dict,
        classifier_result: Optional[dict],
        trashnet_result: Optional[dict],
    ) -> dict:
        label = detection["label"]
        confidence = detection["confidence"]
        box = detection["box"]
        source = detection.get("source", "unknown")

        material = base_info.get("material", "OTHER")
        material_name = RESIN_INFO.get(material, RESIN_INFO["OTHER"])["name"]
        resin_code = base_info.get("resin_code", 7)
        recyclable = base_info.get("recyclable", False)
        is_plastic = base_info.get("is_plastic", True)
        item_type = base_info.get("item_type", label)
        description = base_info.get("description", "")

        if classifier_result and classifier_result.get("reliable"):
            material = classifier_result["material"]
            material_name = classifier_result["material_name"]
            resin_code = classifier_result["resin_code"]
            recyclable = classifier_result["recyclable"]
            description = f"Classified by MobileNetV3: {material_name}"

        if trashnet_result and trashnet_result.get("reliable"):
            if not trashnet_result["is_plastic"]:
                is_plastic = False
                if trashnet_result["waste_type"] in ("glass", "metal"):
                    material = trashnet_result["waste_type"].upper()
                    material_name = trashnet_result["waste_type"].title()
                    resin_code = 0
                elif trashnet_result["waste_type"] in ("paper", "cardboard"):
                    material = "PAPER"
                    material_name = "Paper/Cardboard"
                    resin_code = 0

        yolo_info = detection.get("_yolo_info")
        if source == "yolo_fallback" and yolo_info:
            if not (classifier_result and classifier_result.get("reliable")):
                material = yolo_info.get("material", material)
                material_name = yolo_info.get("material_name", material_name)
                resin_code = yolo_info.get("resin_code", resin_code)
                recyclable = yolo_info.get("recyclable", recyclable)
                is_plastic = yolo_info.get("is_plastic", is_plastic)
                item_type = yolo_info.get("item_type", item_type)
                description = yolo_info.get("description", description)

        plastic_type = self._get_plastic_category(item_type, is_plastic)

        return {
            "label": label,
            "plastic_type": plastic_type,
            "item_type": item_type,
            "is_plastic": is_plastic,
            "material": material,
            "material_name": material_name,
            "resin_code": resin_code,
            "recyclable": recyclable,
            "description": description,
            "confidence": round(confidence, 4),
            "box": box,
            "detection_source": source,
            "classifier_source": (
                classifier_result.get("source", "none")
                if classifier_result else "rule_based"
            ),
        }

    @staticmethod
    def _get_plastic_category(item_type: str, is_plastic: bool) -> str:
        if not is_plastic:
            return "non_plastic_waste"
        item = item_type.lower()
        if "bottle" in item or "cup" in item or "glass" in item or "bowl" in item:
            return "plastic_bottle"
        elif "bag" in item or "carrier" in item or "film" in item or "wrapper" in item:
            return "plastic_bag"
        elif "cutlery" in item or "utensil" in item or "fork" in item or "spoon" in item:
            return "plastic_cutlery"
        elif "straw" in item:
            return "plastic_straw"
        elif "container" in item or "tub" in item or "tupperware" in item:
            return "plastic_container"
        elif "cap" in item or "lid" in item:
            return "plastic_cap"
        elif "casing" in item or "appliance" in item or "fixture" in item:
            return "plastic_casing"
        elif "foam" in item or "styrofoam" in item:
            return "plastic_foam"
        else:
            return "plastic_other"

    @staticmethod
    def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        CATEGORY_COLORS = {
            "plastic_bottle": (0, 255, 0),
            "plastic_bag": (0, 165, 255),
            "plastic_cutlery": (255, 0, 255),
            "plastic_straw": (255, 0, 128),
            "plastic_container": (0, 200, 200),
            "plastic_cap": (200, 200, 0),
            "plastic_casing": (255, 255, 0),
            "plastic_foam": (128, 128, 255),
            "plastic_other": (200, 200, 200),
            "non_plastic_waste": (128, 128, 128),
        }
        COLOR_RECYCLABLE = (0, 200, 0)
        COLOR_NOT_RECYCLABLE = (0, 0, 200)
        DEFAULT_COLOR = (255, 255, 0)

        for det in detections:
            x, y, bw, bh = det["box"]
            ptype = det.get("plastic_type", "plastic_other")
            material = det.get("material", "?")
            resin = det.get("resin_code", "?")
            recyclable = det.get("recyclable", False)
            item_type = det.get("item_type", det["label"])
            conf = det["confidence"]
            is_plastic = det.get("is_plastic", True)

            color = CATEGORY_COLORS.get(ptype, DEFAULT_COLOR)
            recycle_tag = "RECYCLABLE" if recyclable else "NOT RECYCLABLE"
            recycle_color = COLOR_RECYCLABLE if recyclable else COLOR_NOT_RECYCLABLE

            source_tag = ""
            det_source = det.get("detection_source", "")
            if det_source and det_source not in ("yolo_fallback",):
                source_tag = f" [{det_source.split('_')[0].upper()}]"
            line1 = f"{item_type} {conf:.0%}{source_tag}"

            if is_plastic and resin and resin != 0:
                line2 = f"{material} (#{resin}) | {recycle_tag}"
            elif is_plastic:
                line2 = f"{material} | {recycle_tag}"
            else:
                line2 = f"Non-plastic: {material} | {recycle_tag}"

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

    @property
    def pipeline_status(self) -> dict:
        return {
            "taco_detector": "active" if self._taco else "disabled",
            "resin_classifier": "active" if self._classifier else "disabled",
            "trashnet_fallback": "active" if self._trashnet else "disabled",
            "yolo_fallback": "active" if self._yolo_fallback else "disabled",
            "pytorch_available": HAS_TORCH,
            "torchvision_available": HAS_TORCHVISION,
            "detectron2_available": HAS_DETECTRON2,
        }
