import streamlit as st
import cv2
import mediapipe as mp
import time
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
grid_color = (0, 255, 0)  # Green color
drawing_spec = mp_drawing.DrawingSpec(color=grid_color, thickness=1, circle_radius=1)

st.title("MediaPipe Face Mesh Detection")

# Use Streamlit's webcam input
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            st.write("Ignoring empty camera frame.")
            continue

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

        end = time.time()
        fps = 1 / (end - start)
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Convert the image to an RGB format for display in Streamlit
        st.image(image, channels="BGR")

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
