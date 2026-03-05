import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        lm = results.pose_landmarks.landmark

        left_shoulder = lm[11]
        right_shoulder = lm[12]
        left_wrist = lm[15]
        right_wrist = lm[16]
        nose = lm[0]

        message = ""

        # Right hand raised
        if right_wrist.y < right_shoulder.y:
            message = "Right Hand Raised - Need Help"

        # Left hand raised
        elif left_wrist.y < left_shoulder.y:
            message = "Left Hand Raised - Attention"

        # Both hands raised
        elif right_wrist.y < right_shoulder.y and left_wrist.y < left_shoulder.y:
            message = "Emergency Gesture"

        # Hands on head
        elif right_wrist.y < nose.y and left_wrist.y < nose.y:
            message = "Hands on Head - Stress"

        # Arms crossed (approximation)
        elif abs(right_wrist.x - left_shoulder.x) < 0.1 and abs(left_wrist.x - right_shoulder.x) < 0.1:
            message = "Arms Crossed - Discomfort"

        if message:
            cv2.putText(frame,
                        message,
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,0),
                        2)

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()