import os
import cv2
import mediapipe as mp
import csv

# ==========================
# CONFIG
# ==========================
DATASET_PATH = "../dataset/hagrid"
OUTPUT_CSV = "../dataset/gesture_landmarks.csv"

GESTURES = [
    "call",
    "dislike",
    "fist",
    "like",
    "ok",
    "palm",
    "peace",
    "stop"
]

# ==========================
# MEDIAPIPE SETUP
# ==========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# ==========================
# CREATE CSV HEADER
# ==========================
header = []
for i in range(21):
    header.append(f"x{i}")
    header.append(f"y{i}")
header.append("label")

with open(OUTPUT_CSV, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

# ==========================
# PROCESS IMAGES
# ==========================
total_saved = 0

for gesture in GESTURES:
    gesture_folder = os.path.join(DATASET_PATH, gesture)

    print(f"\nProcessing: {gesture}")

    for img_name in os.listdir(gesture_folder):
        img_path = os.path.join(gesture_folder, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            row.append(gesture)

            with open(OUTPUT_CSV, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            total_saved += 1

    print(f"{gesture} done.")

print("\n==============================")
print("Dataset Generation Complete")
print("Total Samples Saved:", total_saved)
print("==============================")