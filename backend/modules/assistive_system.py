import cv2

from emotion.face_emotion import detect_emotion
from gesture.gesture_detection import detect_gesture
from pose.pose_detection import detect_pose
from behavior_interpreter import interpret_behavior


cap = cv2.VideoCapture(0)

print("Assistive Communication System Running...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # run perception modules
    emotion = detect_emotion(frame)
    gesture = detect_gesture(frame)
    pose = detect_pose(frame)

    # interpret behaviour
    result = interpret_behavior(
        emotion=emotion,
        gesture=gesture,
        pose=pose
    )

    message = result["message"]

    # display results
    cv2.putText(frame, f"Emotion: {emotion}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(frame, f"Gesture: {gesture}", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(frame, f"Pose: {pose}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(frame, f"Message: {message}", (20,140),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

    cv2.imshow("Assistive Communication System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()