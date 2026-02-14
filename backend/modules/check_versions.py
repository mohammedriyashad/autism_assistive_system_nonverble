import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

print("cv2:", cv2.__version__)
print("numpy:", np.__version__)
print("mediapipe:", mp.__version__)
print("tensorflow:", tf.__version__)
print("solutions exists:", hasattr(mp, "solutions"))