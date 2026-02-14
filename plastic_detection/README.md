# Plastic Detection Engine

Real-time plastic waste detection using YOLOv4-tiny (OpenCV DNN backend).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on webcam (real-time)
python detector.py --mode webcam

# Run on image
python detector.py --mode image --source photo.jpg

# Run on video
python detector.py --mode video --source sample.mp4

# Run with backend integration
python detector.py --mode webcam --send --zone Z-101

# Run on a folder of images
python detector.py --mode image --source test_images/

# Save annotated output
python detector.py --mode video --source clip.mp4 --save output.mp4
```

## CLI Options

| Flag             | Description                              | Default              |
|------------------|------------------------------------------|----------------------|
| `--mode`         | `image`, `video`, or `webcam`            | `webcam`             |
| `--source`       | Path to image/video file or directory    | —                    |
| `--zone`         | Monitoring zone ID                       | `Z-101`              |
| `--send`         | Send detections to backend API           | off                  |
| `--backend-url`  | Backend API URL                          | `http://localhost:8000` |
| `--camera`       | Webcam device index                      | `0`                  |
| `--conf`         | Confidence threshold                     | `0.4`                |
| `--nms`          | NMS threshold                            | `0.4`                |
| `--size`         | YOLO input size                          | `416`                |
| `--no-display`   | Headless mode (no GUI window)            | off                  |
| `--save`         | Save annotated output to file            | —                    |
| `--gpu`          | Use CUDA GPU acceleration                | off                  |
| `--verbose`      | Enable debug logging                     | off                  |

## Detection Categories

| Category          | Mapped From (COCO)                           |
|-------------------|----------------------------------------------|
| `plastic_bottle`  | bottle, cup, wine glass, vase                |
| `plastic_bag`     | handbag, backpack, suitcase, umbrella        |
| `plastic_wrapper` | cell phone, remote, toothbrush, scissors, book, mouse, keyboard |

## Output Format

```json
{
  "zoneId": "Z-101",
  "plasticType": "plastic_bottle",
  "confidence": 0.84,
  "timestamp": "2026-02-14T12:00:00"
}
```

## Backend Server

```bash
# Start the API server
python backend.py
# or
uvicorn backend:app --reload --port 8000

# API docs at http://localhost:8000/docs
```

### Endpoints

| Method   | Path               | Description                        |
|----------|--------------------|------------------------------------|
| `GET`    | `/`                | Server status                      |
| `GET`    | `/health`          | Health check                       |
| `POST`   | `/report`          | Submit single detection event      |
| `POST`   | `/report/batch`    | Submit multiple events             |
| `POST`   | `/detect/image`    | Upload image for detection         |
| `GET`    | `/detections`      | List detections (with filters)     |
| `GET`    | `/stats`           | Aggregated statistics              |
| `DELETE` | `/detections`      | Clear all (testing)                |

## Testing

```bash
python test_detector.py
```

## Project Structure

```
plastic_detection/
├── detector.py          # Main detection engine (YOLOv4-tiny)
├── backend.py           # FastAPI backend server
├── test_detector.py     # Automated test suite
├── requirements.txt     # Python dependencies
├── README.md
└── model/
    ├── yolov4-tiny.cfg
    ├── yolov4-tiny.weights
    └── coco.names
```
