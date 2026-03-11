from phi_reasoner import generate_message

context = """
Emotion: sad
Gesture: palm
Pose: hand_raised
Speech: hello
"""

print(generate_message(context))