import cv2
from deepface import DeepFace

CONFIDENCE_THRESHOLD = 60


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


def detect_emotion(frame):

    emotion = "none"
    behavior_state = "Unknown"

    try:

        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]
        confidence = result[0]["emotion"][emotion]

        behavior_state = map_to_behavior(emotion, confidence)

    except:
        pass

    return emotion, behavior_state