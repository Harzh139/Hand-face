# 🧠 Face & Hand Detection with MediaPipe + Replicate SAM2

An interactive **Streamlit web app** that performs **face and hand detection** using **MediaPipe** and allows advanced segmentation with **SAM2** (Segment Anything Model) via the **Replicate API**.

---

## 🚀 How It Works

1. Upload an image using the Streamlit interface
2. MediaPipe detects:
   - Faces (bounding boxes)
   - Hands (landmarks)
3. (Optional) Detected face/hand regions are sent to **Replicate SAM2** for segmentation
4. Results are displayed and downloadable!

---

## 💻 Demo

> (Add screenshot or deployment link here)

---

## 📦 Requirements

Install all required packages:

```bash
pip install -r requirements.txt
