# Face & Hand Segmentation Project

This project automatically detects faces and hands using YOLOv8 and segments them using SAM2 (Segment Anything Model) from Replicate API.

## 🚀 How It Works
1. Uses YOLOv8 to detect faces/hands (bounding boxes)
2. Sends boxes to SAM2 API to get precise segmentation
3. Overlays masks on the original image

## 🛠 Requirements
```bash
pip install -r requirements.txt
