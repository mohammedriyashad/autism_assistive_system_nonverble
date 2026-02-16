import cv2
import mediapipe as mp
import csv
import os
import numpy as np

gesture_name = input("Enter gesture name: ")

save_dir = "../dataset/gesture"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, f"{gesture_name}.csv")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opening ❌")
    exit()

print("Press 's' to save sample")
print("Press ESC to exit")

with open(save_path, mode="a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not captured ❌")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Normalize landmarks
                landmarks = np.array(landmarks)
                landmarks = landmarks - landmarks.mean()

        cv2.putText(frame,
                    "S = Save | ESC = Exit",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Collect Gesture Data", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and results.multi_hand_landmarks:
            writer.writerow(landmarks.tolist())
            print("Sample saved ✔")

        elif key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()