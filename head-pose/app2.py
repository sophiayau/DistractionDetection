import base64
import cv2
import numpy as np
import requests
import streamlit as st
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Constants
IMG_PATH = "image.jpg"
API_KEY = os.getenv("ROBOFLOW-INFERENCE-API-KEY")
DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = (
    "http://localhost:9001/gaze/gaze_detection?api_key=" + API_KEY
)

MAX_YAW_LEFT = -0.5  # Maximum yaw for left (e.g., -45 degrees)
MAX_YAW_RIGHT = 0.5  # Maximum yaw for right (e.g., 45 degrees)
MAX_PITCH_UP = -0.5  # Maximum pitch for up (e.g., -30 degrees)
MAX_PITCH_DOWN = 0.5  # Maximum pitch for down (e.g., 30 degrees)

def check_for_distraction(gaze):
    """
    Check if the gaze is out of range (distraction) based on yaw (left-right) and pitch (top-bottom).
    """
    yaw = gaze["yaw"]
    pitch = gaze["pitch"]

    # Check if yaw or pitch exceeds defined distraction thresholds
    if yaw < MAX_YAW_LEFT or yaw > MAX_YAW_RIGHT or pitch < MAX_PITCH_UP or pitch > MAX_PITCH_DOWN:
        return True
    return False

MAX_YAW_LEFT = -0.5  # Maximum yaw for left (e.g., -45 degrees)
MAX_YAW_RIGHT = 0.5  # Maximum yaw for right (e.g., 45 degrees)
MAX_PITCH_UP = -0.5  # Maximum pitch for up (e.g., -30 degrees)
MAX_PITCH_DOWN = 0.5  # Maximum pitch for down (e.g., 30 degrees)

def check_for_distraction(gaze):
    """
    Check if the gaze is out of range (distraction) based on yaw (left-right) and pitch (top-bottom).
    """
    yaw = gaze["yaw"]
    pitch = gaze["pitch"]

    # Check if yaw or pitch exceeds defined distraction thresholds
    if yaw < MAX_YAW_LEFT or yaw > MAX_YAW_RIGHT or pitch < MAX_PITCH_UP or pitch > MAX_PITCH_DOWN:
        return True
    return False


def detect_gazes(frame: np.ndarray):
    """Detect gazes from the inference server."""
    if frame is None or frame.size == 0:
        print("Error: Empty or invalid frame passed to detect_gazes.")
        return []

    try:
        _, img_encode = cv2.imencode(".jpg", frame)
        img_base64 = base64.b64encode(img_encode)
        resp = requests.post(
            GAZE_DETECTION_URL,
            json={
                "api_key": API_KEY,
                "image": {"type": "base64", "value": img_base64.decode("utf-8")},
            },
        )
        resp.raise_for_status()
        gazes = resp.json()[0]["predictions"]
        return gazes
    except Exception as e:
        print(f"Error in detect_gazes: {e}")
        return []

def draw_gaze(img: np.ndarray, gaze: dict):
    """Draw gaze direction and keypoints on the image."""
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # Draw gaze arrow
    _, imgW = img.shape[:2]
    arrow_length = imgW / 2
    dx = -arrow_length * np.sin(gaze["yaw"]) * np.cos(gaze["pitch"])
    dy = -arrow_length * np.sin(gaze["pitch"])
    cv2.arrowedLine(
        img,
        (int(face["x"]), int(face["y"])),
        (int(face["x"] + dx), int(face["y"] + dy)),
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )

    # Draw keypoints
    for keypoint in face["landmarks"]:
        x, y = int(keypoint["x"]), int(keypoint["y"])
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    return img

def main():
    st.title("Distraction Detection App")
    
    video_src = st.sidebar.selectbox("Select Video Source", ("Webcam", "Video File"))

    if video_src == "Video File":
        video_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
        if video_file is not None:
            video_src = video_file
        else:
            st.warning("Please upload a video file.")
            return
    else:
        video_src = 0  # Webcam index

    cap = cv2.VideoCapture(video_src if video_src == 0 else video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    frame_placeholder = st.empty()
    yaw_pitch_placeholder = st.empty()
    frame_counter = 0    timer_placeholder = st.empty()
    summary_placeholder = st.empty()

    focused_time = 0
    distracted_time = 0
    start_time = None
    current_status = None
    timer_running = False
    timer_ended = False

    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("Start Timer")
    with col2:
        end_btn = st.button("End Timer")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if video_src == 0:
            frame = cv2.flip(frame, 2)

        faces, _ = face_detector.detect(frame, 0.7)
        if len(faces) > 0:
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]
            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            head_distraction, pose_vectors = pose_estimator.detect_distraction(marks)

            # getting distraction status from gaze functions
            gazes = detect_gazes(frame)
            if gazes:
                for gaze in gazes:
                    eye_distraction = check_for_distraction(gaze)
                    frame = draw_gaze(frame, gaze)
                    yaw_pitch_placeholder.write(f"Yaw: {gaze['yaw']:.2f}, Pitch: {gaze['pitch']:.2f}")

            combined_distraction = head_distraction or eye_distraction

            # displayed text --> combined results from head and eye gaze
            status_text = "Distracted" if combined_distraction else "Focused"

            cv2.putText(
                frame,
                f"Status: {status_text}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if not combined_distraction else (0, 0, 255),
                2,
            )

            # frame_counter+=1
            gazes = detect_gazes(frame)
            # if frame_counter % 15 == 0:
            if gazes:
                for gaze in gazes:
                    frame = draw_gaze(frame, gaze)
                    yaw_pitch_placeholder.write(f"Yaw: {gaze['yaw']:.2f}, Pitch: {gaze['pitch']:.2f}")

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")


            current_status = status_text

            # Timer Logic
            now = time.time()
            if start_btn and not timer_running:
                timer_running = True
                start_time = now

            if timer_running:
                elapsed_time = now - start_time
                if current_status == "Focused":
                    focused_time += elapsed_time
                elif current_status == "Distracted":
                    distracted_time += elapsed_time
                start_time = now  # Update for the next frame

                timer_placeholder.write(
                    f"Focused Time: {focused_time:.2f}s | Distracted Time: {distracted_time:.2f}s"
                )

            if end_btn and timer_running:
                timer_running = False
                timer_ended = True
                elapsed_time = now - start_time
                if current_status == "Focused":
                    focused_time += elapsed_time
                elif current_status == "Distracted":
                    distracted_time += elapsed_time
                break  

        if timer_ended:
            break

    cap.release()

if __name__ == "__main__":
    main()
