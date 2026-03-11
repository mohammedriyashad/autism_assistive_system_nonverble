import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd

model = joblib.load("gesture_model.pkl")

labels = ["call","dislike","fist","like","ok","palm","peace","stop"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)


def detect_gesture(frame):

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    gesture = "none"

    if results.multi_hand_landmarks:

        for hand in results.multi_hand_landmarks:

            landmarks = []

            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks = np.array(landmarks).reshape(1,-1)

            landmarks_df = pd.DataFrame(landmarks)

            prediction = model.predict(landmarks_df)

            gesture = labels[int(prediction)]

    return gesture