def interpret_behavior(emotion=None, gesture=None, pose=None):
    """
    Multimodal Behavior Interpreter

    Combines outputs from:
    - Emotion Detection
    - Gesture Detection
    - Pose Detection

    Returns:
    intent, message, confidence, reasoning
    """

    reasoning = []
    confidence = 0.5
    intent = "UNKNOWN"
    message = "Unable to understand request"

    # -----------------------------
    # Emergency detection
    # -----------------------------
    if pose == "Both Hands Raised - Emergency":
        intent = "EMERGENCY"
        message = "Emergency assistance required"
        confidence = 0.95
        reasoning.append("Both hands raised detected")

    # -----------------------------
    # Help request
    # -----------------------------
    elif gesture == "help" or pose == "Right Hand Raised":
        intent = "REQUEST_HELP"
        message = "I need help"
        confidence = 0.90
        reasoning.append("Help gesture or hand raised")

    # -----------------------------
    # Drink request
    # -----------------------------
    elif gesture == "drink":
        intent = "REQUEST_DRINK"
        message = "I want water"
        confidence = 0.88
        reasoning.append("Drink gesture detected")

    # -----------------------------
    # Food request
    # -----------------------------
    elif gesture == "eat":
        intent = "REQUEST_FOOD"
        message = "I want food"
        confidence = 0.88
        reasoning.append("Eat gesture detected")

    # -----------------------------
    # Toilet request
    # -----------------------------
    elif gesture == "toilet":
        intent = "REQUEST_TOILET"
        message = "I need to go to the toilet"
        confidence = 0.90
        reasoning.append("Toilet gesture detected")

    # -----------------------------
    # Stress detection
    # -----------------------------
    elif pose == "Hands on Head - Stress":
        intent = "STRESS"
        message = "I feel stressed"
        confidence = 0.85
        reasoning.append("Hands on head pose")

    # -----------------------------
    # Discomfort
    # -----------------------------
    elif pose == "Arms Crossed - Discomfort":
        intent = "DISCOMFORT"
        message = "I feel uncomfortable"
        confidence = 0.80
        reasoning.append("Arms crossed pose")

    # -----------------------------
    # Emotion-based interpretation
    # -----------------------------
    elif emotion == "sad":
        intent = "SAD"
        message = "I feel sad"
        confidence = 0.75
        reasoning.append("Sad facial emotion")

    elif emotion == "angry":
        intent = "ANGER"
        message = "I feel angry"
        confidence = 0.75
        reasoning.append("Angry facial emotion")

    elif emotion == "happy":
        intent = "HAPPY"
        message = "I feel happy"
        confidence = 0.70
        reasoning.append("Happy facial emotion")

    # -----------------------------
    # Neutral fallback
    # -----------------------------
    elif emotion == "neutral":
        intent = "NEUTRAL"
        message = "User is calm"
        confidence = 0.60
        reasoning.append("Neutral emotion detected")

    # -----------------------------
    # Return result
    # -----------------------------
    return {
        "intent": intent,
        "message": message,
        "confidence": confidence,
        "reasoning": reasoning,
        "inputs": {
            "emotion": emotion,
            "gesture": gesture,
            "pose": pose
        }
    }