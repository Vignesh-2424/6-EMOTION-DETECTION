import cv2
import numpy as np
import time
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ==== Configuration ====
MODEL_PATH = "model/emotion_model.keras"
CLASS_INDICES_PATH = "model/class_indices.json"
IMG_SIZE = 48
DETECTION_INTERVAL = 5  # seconds
CAMERA_INDEX = 1  # Change to your external camera index (e.g., 1, 2)

# ==== Load model and labels ====
model = load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping of class indices
labels = {v: k for k, v in class_indices.items()}

# Emoji mapping
emoji_map = {
    "angry": "😠",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

# ==== Load OpenCV face detector ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ==== Start video capture ====
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"❌ Could not open camera at index {CAMERA_INDEX}. Try a different index.")
    exit()

last_prediction_time = 0
print("🔴 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    current_time = time.time()
    if current_time - last_prediction_time >= DETECTION_INTERVAL:
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = roi.astype("float32") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = model.predict(roi, verbose=0)
            label_index = np.argmax(predictions)
            label = labels[label_index]
            emoji = emoji_map.get(label, "")
            print(f"🧠 You are {label} {emoji}")
            last_prediction_time = current_time

    cv2.imshow("Emotion Detection - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
