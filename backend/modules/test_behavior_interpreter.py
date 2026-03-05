from behavior_interpreter import interpret_behavior

# Example outputs coming from modules
emotion_output = "neutral"
gesture_output = "drink"
pose_output = "none"

result = interpret_behavior(
    emotion=emotion_output,
    gesture=gesture_output,
    pose=pose_output
)

print("---- SYSTEM OUTPUT ----")
print("Emotion :", emotion_output)
print("Gesture :", gesture_output)
print("Pose    :", pose_output)

print("\nIntent      :", result["intent"])
print("Message     :", result["message"])
print("Confidence  :", result["confidence"])
print("Reasoning   :", result["reasoning"])