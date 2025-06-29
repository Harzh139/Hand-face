import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile

st.set_page_config(page_title="Face & Hand Detection", layout="centered")
st.title("🖐️ Realistic Face & Hand Detection with MediaPipe")
st.write("Upload a clear image and see detected faces and hands marked beautifully.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    image_path = temp_file.name

    # Load and preprocess
    image = cv2.imread(image_path)
    if image is None:
        st.error("❌ Failed to load image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        with mp_hands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=0.5) as hands, \
             mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7) as face:

            # Detect hands
            hand_result = hands.process(image_rgb)
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

            # Detect faces
            face_result = face.process(image_rgb)
            if face_result.detections:
                for detection in face_result.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    w_box = int(bbox.width * w)
                    h_box = int(bbox.height * h)
                    cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
                    cv2.putText(image, "Face", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Convert BGR → RGB for display
        result_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(result_rgb, caption="🖼️ Detected Image", use_column_width=True)

        # Download button
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        result_path = "face_hand_output.jpg"
        cv2.imwrite(result_path, result_bgr)

        with open(result_path, "rb") as file:
            st.download_button("📥 Download Result", file, file_name="face_hand_output.jpg")

        st.success("✅ Detection complete!")
