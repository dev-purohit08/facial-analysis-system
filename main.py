from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import base64
import time

# ---------------- APP INIT ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_landmarker.task")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ---------------- LOAD MODEL ----------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------- GLOBAL STATE ----------------
current_emotion = "Neutral"
emotion_counter = 0
emotion_hold_frames = 5
is_drowsy = False

# ---------------- UTILITY FUNCTIONS ----------------
def get_face_vector(landmarks):
    vec = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return vec / np.linalg.norm(vec)

def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def decode_base64_image(data):
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# ---------------- LOAD KNOWN FACES ----------------
known_vectors = []
known_names = []
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def load_known_faces():
    known_vectors.clear()
    known_names.clear()

    for file in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, file)
        name = os.path.splitext(file)[0]

        img = cv2.imread(path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            vec = get_face_vector(result.face_landmarks[0])
            known_vectors.append(vec)
            known_names.append(name)

load_known_faces()

# ---------------- REQUEST MODELS ----------------
class FrameData(BaseModel):
    image: str
    mode: str

class CaptureData(BaseModel):
    image: str

# ---------------- PROCESS FRAME ----------------
@app.post("/process_frame")
def process_frame(data: FrameData):
    global current_emotion, emotion_counter, is_drowsy

    frame = decode_base64_image(data.image)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        points = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])
        face_vec = get_face_vector(landmarks)

        # -------- LANDMARK --------
        if data.mode == "landmark":
            for (x, y) in points:
                cv2.circle(frame, (x, y), 1, (0,255,0), -1)

        # -------- EMOTION --------
        elif data.mode == "emotion":
            detected = "Neutral"

            if result.face_blendshapes:
                scores = {b.category_name: b.score for b in result.face_blendshapes[0]}
                threshold = 0.20

                happy = scores.get("mouthSmileLeft",0) + scores.get("mouthSmileRight",0)
                sad = scores.get("mouthFrownLeft",0) + scores.get("mouthFrownRight",0)
                angry = scores.get("browDownLeft",0) + scores.get("browDownRight",0)
                surprise = scores.get("jawOpen",0)

                emotion_scores = {
                    "Happy": happy,
                    "Sad": sad,
                    "Angry": angry,
                    "Surprised": surprise
                }

                best = max(emotion_scores, key=emotion_scores.get)

                if emotion_scores[best] > threshold:
                    detected = best

            if detected == current_emotion:
                emotion_counter = 0
            else:
                emotion_counter += 1
                if emotion_counter >= emotion_hold_frames:
                    current_emotion = detected
                    emotion_counter = 0

            cv2.putText(frame, f"Emotion: {current_emotion}",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,0),
                        2)

        # -------- DROWSINESS --------
        elif data.mode == "drowsy":
            left_eye = points[LEFT_EYE]
            right_eye = points[RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            if ear < 0.20:
                is_drowsy = True
                cv2.putText(frame, "DROWSINESS ALERT!",
                            (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)
            else:
                is_drowsy = False
                cv2.putText(frame, "Eyes Open",
                            (30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

        # -------- RECOGNITION --------
        elif data.mode == "recognition":
            name = "Unknown"
            min_dist = 999

            for vec, person in zip(known_vectors, known_names):
                dist = np.linalg.norm(face_vec - vec)
                if dist < min_dist:
                    min_dist = dist
                    name = person

            if min_dist > 0.6:
                name = "Unknown"

            cv2.putText(frame, f"Person: {name}",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,255,0),
                        2)

    _, buffer = cv2.imencode(".jpg", frame)
    processed = base64.b64encode(buffer).decode("utf-8")

    return {
        "image": f"data:image/jpeg;base64,{processed}",
        "drowsy": is_drowsy
    }

# ---------------- CAPTURE FACE ----------------
@app.post("/capture/{person_name}")
def capture_photo(person_name: str, data: CaptureData):

    frame = decode_base64_image(data.image)
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.face_landmarks:
        return {"status": "no_face_detected"}

    filename = f"{person_name}_{int(time.time())}.jpg"
    path = os.path.join(KNOWN_FACES_DIR, filename)
    cv2.imwrite(path, frame)

    vec = get_face_vector(result.face_landmarks[0])
    known_vectors.append(vec)
    known_names.append(person_name)

    return {"status": "saved", "file": filename}
