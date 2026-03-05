import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands # type: ignore
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils # type: ignore

cap = cv2.VideoCapture(0)

labels = ["call","dislike","fist","like","ok","palm","peace","stop"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            landmarks = []
            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks = np.array(landmarks).reshape(1,-1)

            
            import pandas as pd
            landmarks_df = pd.DataFrame(landmarks)
            prediction = model.predict(landmarks_df) 
            gesture = labels[int(prediction)]

            cv2.putText(frame, gesture, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0),2)

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()