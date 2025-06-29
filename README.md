# 🧠 Face & Hand Segmentation Project

This project automatically detects **faces and hands** using **YOLOv8** and segments them with **SAM2 (Segment Anything Model)** via the **Replicate API**. It's built with Python, OpenCV, MediaPipe, and Streamlit for interactive UI.

---

## 🚀 How It Works

1. 🟨 **YOLOv8** detects bounding boxes for faces and hands
2. 🧠 **SAM2 (Segment Anything Model)** receives the boxes to perform pixel-wise segmentation
3. 🖼️ Segmentation masks are overlayed on the original image and displayed
4. 💻 A **Streamlit UI** allows users to upload images and get downloadable results

---

## 🛠 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
