
# module that encodes and decodes binary data into a textual representation
# converts an image into a base64 encoded string
import base64
# computer vision library
# captures frames from camera, processes images, displays video
import cv2
# numerical computing library
# can handle image data. in our case, we use it to calculate distance to object 
import numpy as np
# library that simplifies making HTTP requests
import requests
# library that provides a way to interact with the operating system
import os

IMG_PATH = "image.jpg"
API_KEY = "zPYon8OkapXPB5JFZ2h3"
DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = (
"http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY)

# takes in frame parameter of type np array, this is our image from the camera
# numpy arrays in terms of images are usually 3D arrays if they're colored images
# 3D --> height, width, color channels
def detect_gazes(frame: np.ndarray):

    if frame is None or frame.size == 0:
        print("Error: Empty or invalid frame passed to detect_gazes.")
        return []

    try:
        # converting the passed in frame to a .jpg file 
        _, img_encode = cv2.imencode(".jpg", frame)
        # converts binary data into a string for HTTP protocols
        img_base64 = base64.b64encode(img_encode)

        # sends a post request to gaze detection url
        resp = requests.post(
            GAZE_DETECTION_URL,
            json={
                "api_key": API_KEY,
                # decode converts base64 binary data to string format
                "image": {"type": "base64", "value": img_base64.decode("utf-8")},
            },
        )
        # checks for HTTP errors
        # in case there is a server issue
        resp.raise_for_status()  

        # Parse and return the predictions
        # resp is JSON formatted
        # .json is a python built in function to parse this json file into a dictionary 
        # gazes is a list of maps/dictionaries
        gazes = resp.json()[0]["predictions"]
        return gazes

    except Exception as e:
        print(f"Error in detect_gazes: {e}")
        return []


# takes in an img of type NumPy array and one dictionary from our list of dictionaries earlier
# draw face bounding box
def draw_gaze(img: np.ndarray, gaze: dict):
    
    # takes in face values from our dictionary
    # includes coordinates, height, width, etc
    face = gaze["face"]

    # calculation for face bounding box
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # draw gaze arrow
    # extracts the height and width of the image
    _, imgW = img.shape[:2]

    # length set to half the width so it doesn't go out of bounds
    arrow_length = imgW / 2

    # positive yaw value = face turned to right
    # negative yaw value = face turned to left
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

    # draw keypoints
    for keypoint in face["landmarks"]:
        color, thickness, radius = (0, 255, 0), 2, 2
        x, y = int(keypoint["x"]), int(keypoint["y"])
        cv2.circle(img, (x, y), thickness, color, radius)

    # draw label and score
    label = "yaw {:.2f}  pitch {:.2f}".format(
        gaze["yaw"] / np.pi * 180, gaze["pitch"] / np.pi * 180
    )
    cv2.putText(
        img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
    )

    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Could not read frame.")
            continue

        frame_count += 1
        gazes = []

        # Process frames every 5 iterations
        # processing every several frames instead of every frame should make video look smoother

        gazes = detect_gazes(frame)
        # if frame_count % 10 == 0:
        #     gazes = detect_gazes(frame)
        #     frame_count = 0

        if not gazes:  # If no gazes detected, skip drawing
            continue

        # Draw face & gaze for the first detected gaze
        gaze = gazes[0]
        draw_gaze(frame, gaze)

        image_height, image_width = frame.shape[:2]

        length_per_pixel = HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]

        dx = -DISTANCE_TO_OBJECT * np.tan(gaze["yaw"]) / length_per_pixel
        dx = dx if not np.isnan(dx) else 100000000
        dy = -DISTANCE_TO_OBJECT * np.tan(gaze["pitch"]) / length_per_pixel
        dy = dy if not np.isnan(dy) else 100000000
        gaze_point = int(image_width / 2 + dx), int(image_height / 2 + dy)

        # cv2.circle(frame, gaze_point, 25, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow("Gaze Tracker", frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
