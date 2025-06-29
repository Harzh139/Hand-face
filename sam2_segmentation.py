import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import replicate
import requests
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Face & Hand Detection + Segmentation", layout="centered")
st.title("🖼️ Face & Hand Detection + SAM-2 Segmentation")
st.write("Upload an image to detect faces and hands using MediaPipe and segment them using SAM-2.")

# Replicate API from Streamlit secrets
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

def get_sam2_mask(image_path, boxes):
    output = client.run(
        "meta/sam-2",
        input={
            "image": open(image_path, "rb"),
            "boxes": boxes,
            "mask_format": "rgba"
        }
    )
    return output["masks"]

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    image_path = temp_file.name

    image = cv2.imread(image_path)
    if image is None:
        st.error("❌ Could not load image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        face_boxes = []

        with mp_hands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=0.6) as hands, \
             mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face:

            hand_result = hands.process(image_rgb)
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

            face_result = face.process(image_rgb)
            if face_result.detections:
                for detection in face_result.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)

                    # Draw face box
                    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)
                    cv2.putText(image, "Face", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    face_boxes.append({"x": x, "y": y, "width": width, "height": height})

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="🔍 MediaPipe Detection", use_column_width=True)

        if face_boxes:
            st.info("🧠 Running SAM-2 for segmentation...")

            try:
                mask_urls = get_sam2_mask(image_path, face_boxes)
                original = Image.open(image_path).convert("RGBA")
                response = requests.get(mask_urls[0])
                mask_img = Image.open(BytesIO(response.content)).convert("RGBA")
                overlay = Image.alpha_composite(original, mask_img)

                st.image(overlay, caption="🎭 Segmented Image", use_column_width=True)

                output_path = "segmented_output.png"
                overlay.save(output_path)

                with open(output_path, "rb") as f:
                    st.download_button("📥 Download Segmented Image", f, file_name="segmented_output.png")

                st.success("✅ Segmentation complete!")

            except Exception as e:
                st.error(f"❌ Segmentation failed: {e}")
        else:
            st.warning("⚠️ No face detected for segmentation.")
