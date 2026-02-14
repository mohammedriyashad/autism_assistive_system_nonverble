import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("../models/gesture_model.pkl")
label_map = joblib.load("../models/label_map.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([landmarks])[0]
            gesture_name = label_map[prediction]

            cv2.putText(frame, gesture_name,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255,0,0), 2)

    cv2.imshow("Gesture Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()