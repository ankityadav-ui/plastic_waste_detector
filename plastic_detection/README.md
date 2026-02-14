# Plastic Detection Engine

Real-time plastic waste detection and resin type classification using YOLOv4 (OpenCV DNN) with an optional hybrid multi-model pipeline (PyTorch).

## Features

- Real-time detection from webcam, video files, or images
- COCO-to-plastic material mapping with resin identification codes (1-7)
- Recyclability classification for each detected item
- Multi-scale inference and temporal smoothing for stable detections
- FastAPI backend with REST endpoints for reporting and image upload
- Optional hybrid pipeline: SSDLite320-MobileNetV3 + MobileNetV3 resin classifier + TrashNet fallback

## Prerequisites

- Python 3.10+
- Webcam (for real-time mode)
- ~270 MB disk space for YOLOv4 model weights

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SSHRIHARI006/plastic_waste_detector.git
cd plastic_waste_detector/plastic_detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the hybrid detection pipeline (optional), also install PyTorch:

```bash
pip install torch torchvision
```

### 4. Download model weights

The YOLOv4 weights are required. Download them into the `model/` directory:

```bash
python download_models.py --yolov4
```

Or download manually:

| File | URL | Size |
|------|-----|------|
| `yolov4.weights` | [Download](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights) | 246 MB |
| `yolov4.cfg` | [Download](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg) | 12 KB |
| `yolov4-tiny.weights` | [Download](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights) | 22 MB |
| `yolov4-tiny.cfg` | [Download](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg) | 3 KB |
| `coco.names` | [Download](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names) | 1 KB |

Place all files in `plastic_detection/model/`.

## Usage

### Detection

```bash
# Webcam (real-time)
python detector.py --mode webcam

# Webcam with YOLOv4-tiny (faster, less accurate)
python detector.py --mode webcam --tiny

# Single image
python detector.py --mode image --source photo.jpg

# Video file
python detector.py --mode video --source clip.mp4

# Save annotated output
python detector.py --mode video --source clip.mp4 --save output.mp4

# Image directory
python detector.py --mode image --source test_images/

# Send detections to backend
python detector.py --mode webcam --send --zone Z-101

# GPU acceleration (requires OpenCV with CUDA)
python detector.py --mode webcam --gpu
```

### Hybrid Pipeline (requires PyTorch)

```bash
python detector.py --mode webcam --use-taco
python detector.py --mode webcam --use-taco --use-classifier
```

### Backend Server

```bash
python backend.py
# or
uvicorn backend:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `image`, `video`, or `webcam` | `webcam` |
| `--source` | Path to image/video file or directory | — |
| `--zone` | Monitoring zone ID | `Z-101` |
| `--send` | Send detections to backend API | off |
| `--backend-url` | Backend API URL | `http://localhost:8000` |
| `--camera` | Webcam device index | `0` |
| `--conf` | Confidence threshold | `0.25` |
| `--nms` | NMS threshold | `0.5` |
| `--size` | YOLO input size | `608` |
| `--tiny` | Use YOLOv4-tiny | off |
| `--gpu` | Use CUDA GPU acceleration | off |
| `--no-display` | Headless mode (no GUI window) | off |
| `--save` | Save annotated output to file | — |
| `--no-multiscale` | Disable multi-scale inference | off |
| `--no-smooth` | Disable temporal smoothing | off |
| `--use-taco` | Enable SSDLite320 waste detector | off |
| `--use-classifier` | Enable MobileNetV3 resin classifier | off |
| `--verbose` | Enable debug logging | off |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Server status |
| `GET` | `/health` | Health check |
| `POST` | `/report` | Submit single detection event |
| `POST` | `/report/batch` | Submit multiple events |
| `POST` | `/detect/image` | Upload image for detection |
| `GET` | `/detections` | List detections (with filters) |
| `GET` | `/stats` | Aggregated statistics |
| `DELETE` | `/detections` | Clear all detections |

## Detection Output Format

```json
{
  "zoneId": "Z-101",
  "plasticType": "plastic_bottle",
  "itemType": "Plastic Bottle",
  "material": "PET",
  "materialName": "Polyethylene Terephthalate",
  "resinCode": 1,
  "recyclable": true,
  "confidence": 0.87,
  "timestamp": "2026-02-14T12:00:00"
}
```

## Testing

```bash
python test_detector.py
```

## Project Structure

```
plastic_detection/
├── detector.py           # YOLOv4 detection engine
├── hybrid_detector.py    # Multi-model hybrid pipeline
├── backend.py            # FastAPI backend server
├── download_models.py    # Model weight downloader
├── test_detector.py      # Test suite
├── requirements.txt
├── README.md
└── model/
    ├── yolov4.cfg
    ├── yolov4.weights
    ├── yolov4-tiny.cfg
    ├── yolov4-tiny.weights
    └── coco.names
```
