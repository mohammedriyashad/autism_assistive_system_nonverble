import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def detect_pose(frame):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    message = "none"

    if results.pose_landmarks:

        lm = results.pose_landmarks.landmark

        left_shoulder = lm[11]
        right_shoulder = lm[12]
        left_wrist = lm[15]
        right_wrist = lm[16]
        nose = lm[0]

        if right_wrist.y < right_shoulder.y:
            message = "Right Hand Raised - Need Help"

        elif left_wrist.y < left_shoulder.y:
            message = "Left Hand Raised - Attention"

        elif right_wrist.y < right_shoulder.y and left_wrist.y < left_shoulder.y:
            message = "Emergency Gesture"

        elif right_wrist.y < nose.y and left_wrist.y < nose.y:
            message = "Hands on Head - Stress"

        elif abs(right_wrist.x - left_shoulder.x) < 0.1 and abs(left_wrist.x - right_shoulder.x) < 0.1:
            message = "Arms Crossed - Discomfort"

    return message