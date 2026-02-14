import cv2
import csv
import os
from datetime import datetime
from deepface import DeepFace

# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "../dataset/emotion_data.csv"
CONFIDENCE_THRESHOLD = 60

# =====================================================
# BEHAVIORAL MAPPING
# =====================================================

def map_to_behavior(emotion, confidence):
    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain State"

    mapping = {
        "angry": "Distress / Frustration",
        "sad": "Low Mood / Withdrawal",
        "fear": "Anxiety / Overload",
        "happy": "Positive Engagement",
        "surprise": "Overstimulated",
        "neutral": "Calm / Baseline"
    }

    return mapping.get(emotion, "Unknown")

# =====================================================
# CSV LOGGER
# =====================================================

def save_to_csv(timestamp, emotion, confidence, behavior):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    file_exists = os.path.isfile(DATA_PATH)

    with open(DATA_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "emotion",
                "confidence",
                "behavior_state"
            ])

        writer.writerow([
            timestamp,
            emotion,
            round(confidence, 2),
            behavior
        ])

# =====================================================
# INITIALIZATION
# =====================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Press 'S' to save emotion sample")
print("Press ESC to exit")

# =====================================================
# MAIN LOOP
# =====================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion = "Unknown"
    confidence = 0
    behavior_state = "Unknown"

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            result = DeepFace.analyze(
                face,
                actions=["emotion"],
                enforce_detection=False
            )

            emotion = result[0]["dominant_emotion"]
            confidence = result[0]["emotion"][emotion]
            behavior_state = map_to_behavior(emotion, confidence)

        except:
            emotion = "Detecting..."
            confidence = 0
            behavior_state = "Analyzing..."

        # Draw face box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Text positions
        base_y = y - 70 if y > 80 else y + h + 20

        # Emotion
        cv2.putText(frame,
                    f"Emotion: {emotion}",
                    (x, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2)

        # Confidence
        cv2.putText(frame,
                    f"Confidence: {confidence:.1f}%",
                    (x, base_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

        # Behavioral State
        cv2.putText(frame,
                    f"State: {behavior_state}",
                    (x, base_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

    cv2.imshow("Face Emotion Behavioral Module", frame)

    key = cv2.waitKey(1)

    # Save sample when pressing S
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_to_csv(timestamp, emotion, confidence, behavior_state)
        print("Saved:", emotion, confidence, behavior_state)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()