import base64
import cv2
import numpy as np
import requests
import streamlit as st
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

# Constants
IMG_PATH = "image.jpg"
API_KEY = "zPYon8OkapXPB5JFZ2h3"
DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = (
    "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY
)


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

    # print(gaze['yaw'], gaze["pitch"])
    
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

            distraction_status, pose_vectors = pose_estimator.detect_distraction(marks)
            status_text = "Distracted" if distraction_status else "Focused"

            cv2.putText(
                frame,
                f"Status: {status_text}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if not distraction_status else (0, 0, 255),
                2,
            )

            gazes = detect_gazes(frame)
            if gazes:
                for gaze in gazes:
                    frame = draw_gaze(frame, gaze)
                    yaw_pitch_placeholder.write(f"Yaw: {gaze['yaw']:.2f}, Pitch: {gaze['pitch']:.2f}")

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()


if __name__ == "__main__":
    main()
