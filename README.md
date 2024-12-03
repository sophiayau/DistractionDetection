![Header](./banner.jpg)

## Table of Contents
* [About](#about-)
* [Getting Started](#getting-started-)
* [Contacts](#contacts-)

## About 💡
**_Focus Track_ is a real-time distraction detection application that monitors and analyzes gaze direction and head movement using computer vision and machine learning. Our application allows users to understand their attention span and find ways to regain their focus back.**  
<img src="./demo.gif" width="350" />

### Problem Statement 🎯
In today’s digital age, technology, smartphones, and social media platforms like TikTok has negatively impacted our ability to concentrate on a single task for an extended period. These platforms captivate attention with endless streams of short-form content, making it especially difficult for younger generations to maintain focus. Recognizing the severity of this issue, our project aims to help users understand and analyze their patterns of focus and distraction. By becoming aware of these habits, individuals can take proactive steps to improve their focus and regain control over their time and productivity.

### Key Features
**Eye Gaze Detection**  
We implemented a gaze detection model to recognize when someones gaze is no longer at the screen. For each frame, it returns us the pitch and yaw and we use these numbers to set thresholds.

**Head Pose Estimation**  
Alongside the gaze detection, we also added a head pose estimation and once again set our thresholds to recognize when someone is not facing the screen. This model combined with the gaze model allows us to determine whether someone is distracted or focused.

**Timer**  
Users can start and end the program. We added a timer function that displays the total time that the user was distracted/focused. This allows users to gain a better understanding of their distracted to focused ratio. 

## Getting Started ✅
```bash
# Clone our repository
git clone https://github.com/sophiayau/DistractionDetection.git

# Navigate into the repo
cd DistractionDetection

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```


## Contacts 👩‍💻👨‍💻
**Sophia Yau**  
[![image](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)(https://github.com/sophiayau)]
[GitHub](https://github.com/sophiayau) - [LinkedIn](https://www.linkedin.com/in/sophiayau/)  
  
**Ye Htut Maung (Mike)**  
[GitHub](https://github.com/ye-htut-maung) - [LinkedIn](https://www.linkedin.com/in/ye-htut-maung/) 
  
**Afnan Ebrahim**  
[GitHub](https://github.com/Afnan214) - [LinkedIn](https://www.linkedin.com/in/afnan214/)

