![Header](./banner.jpg)

## Table of Contents
* [About](#about)
* Challenges
* Using Our Application
* Contacts

## About
**_Focus Track_ is a real-time distraction detection application that monitors and analyzes gaze direction and head movement using _computer vision_ and _machine learning_. Our application allows users to understand their attention span and find ways to regain their focus back.**

### Problem Statement
In today’s digital age, technology, smartphones, and social media platforms like TikTok has negatively impacted our ability to concentrate on a single task for an extended period. These platforms captivate attention with endless streams of short-form content, making it especially difficult for younger generations to maintain focus. Recognizing the severity of this issue, our project aims to help users understand and analyze their patterns of focus and distraction. By becoming aware of these habits, individuals can take proactive steps to improve their focus and regain control over their time and productivity.

### Key Features
**Eye Gaze Detection**  
We implemented a gaze detection model to recognize when someones gaze is no longer at the screen. For each frame, it returns us the pitch and yaw and we use these numbers to set thresholds.

**Head Pose Estimation**  
Alongside the gaze detection, we also added a head pose estimation and once again set our thresholds to recognize when someone is not facing the screen. This model combined with the gaze model allows us to determine whether someone is distracted or focused.

**Timer**  
Users can start and end the program. We added a timer function that displays the total time that the user was distracted/focused. This allows users to gain a better understanding of their distracted to focused ratio. 

