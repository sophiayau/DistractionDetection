

"""
for the main part of the project
"""

import streamlit as st
from streamlit_navigation_bar import st_navbar
from datetime import date, timedelta
import time
import base64
import cv2
import numpy as np
import requests
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

def landing_page():
    st.markdown("<h1 style='text-align: center; color: black;'>Focus Point</h1>", unsafe_allow_html=True)
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


#     st.session_state.start_time = time.time()
#     elapsed_time = time.time() - st.session_state.start_time
#     elapsed_minutes = int(elapsed_time // 60)
#     elapsed_seconds = int(elapsed_time % 60)

    # st.title("Timer")
    # st.write(f"{elapsed_minutes}:{elapsed_seconds}")

    # while True:
    #     elasped_time = time.time() - st.session_state.start_time
    #     elapsed_minutes = int(elapsed_time // 60)
    #     elapsed_seconds = int(elapsed_time % 60)

    
landing_page()