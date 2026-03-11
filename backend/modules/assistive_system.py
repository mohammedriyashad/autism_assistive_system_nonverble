import cv2

from face_emotion import detect_emotion
from gesture_live_test import detect_gesture
from pose_detection import detect_pose
from behavior_interpreter import interpret_behavior
from audio_input import detect_audio


cap = cv2.VideoCapture(0)

print("Assistive Communication System Running...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # run perception modules
    emotion, behavior_state = detect_emotion(frame)
    gesture = detect_gesture(frame)
    pose = detect_pose(frame)
    audio_text = detect_audio()
    if audio_text is None:
        audio_text = "none"

    # interpret behaviour
    result = interpret_behavior(
        emotion=emotion,
        gesture=gesture,
        pose=pose,
        audio=audio_text
    )

    message = result["message"]

    # display results
    cv2.putText(frame, f"Emotion: {emotion}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(frame, f"Gesture: {gesture}", (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(frame, f"Pose: {pose}", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame, f"Audio: {audio_text}", (20,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(225,255,0),2)

    cv2.putText(frame, f"Message: {message}", (20,140),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

    cv2.imshow("Assistive Communication System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()