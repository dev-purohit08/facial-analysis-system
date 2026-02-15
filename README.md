# Real Time Facial Analysis System

A cloud-compatible AI web application built using FastAPI, MediaPipe and OpenCV.

This system performs real-time:

- Facial Landmark Detection
- Emotion Detection
- Drowsiness Detection (with Alarm)
- Face Recognition
- Face Enrollment

The application is fully browser-based and cloud deployable.

---

## 🚀 Live Demo

(Deploy link will go here after deployment)

---

## 🛠 Tech Stack

Backend:
- FastAPI
- MediaPipe Face Landmarker
- OpenCV
- NumPy

Frontend:
- HTML
- CSS
- JavaScript
- Web Camera API (getUserMedia)

Deployment:
- Render / AWS / Any Cloud Platform

---

## 🧠 Features

### 1. Facial Landmarks
Detects and visualizes 468 facial landmarks in real time.

### 2. Emotion Detection
Classifies facial expression into:
- Happy
- Sad
- Angry
- Surprised
- Neutral

Includes smoothing logic to prevent rapid flickering.

### 3. Drowsiness Detection
Uses Eye Aspect Ratio (EAR) to detect closed eyes.
Triggers browser alarm when drowsiness is detected.

### 4. Face Recognition
Matches detected face against enrolled faces using landmark vector similarity.

### 5. Face Enrollment
Allows capturing and saving new faces into the system.

---

## 📂 Project Structure

project/
│
├── main.py
├── requirements.txt
├── README.md
│
├── static/
│ ├── index.html
│ └── alarm.mp3
│
├── models/
│ └── face_landmarker.task
│
└── known_faces/


---

## ⚙️ Installation (Local Setup)

1. Clone the repository:

git clone https://github.com/dev-purohit08/facial-analysis-system

2. Create virtual environment:

python -m venv .venv


3. Activate environment:

Windows:
.venv\Scripts\activate


Mac/Linux:
source .venv/bin/activate


4. Install dependencies:

pip install -r requirements.txt


5. Run server:

uvicorn main:app --reload


6. Open in browser:

http://localhost:8000


---

## 🌐 Cloud Deployment

For Render:

Start command:

uvicorn main:app --host 0.0.0.0 --port 10000


Python version:
3.10.x


---

## ⚠️ Important Notes

- Application uses browser camera access.
- Backend does NOT use server webcam.
- Alarm sound is handled entirely in frontend.
- Mediapipe model file must be present in `/models` directory.

---

## 👨‍💻 Author

MCA Final Year Project  
Real Time Facial Analysis System
