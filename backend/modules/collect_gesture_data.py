import cv2
import mediapipe as mp
import csv
import os

gesture_name = input("Enter gesture name: ")

dataset_path = "../dataset/gesture"
os.makedirs(dataset_path, exist_ok=True)

file_path = os.path.join(dataset_path, f"{gesture_name}.csv")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

print("Press 's' to save sample, ESC to exit")

with open(file_path, mode='a', newline='') as f:
    writer = csv.writer(f)

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

                cv2.putText(frame, gesture_name,
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,0), 2)

                key = cv2.waitKey(1)

                if key == ord('s'):
                    writer.writerow(landmarks)
                    print("Saved")

        cv2.imshow("Collect Gesture Data", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()