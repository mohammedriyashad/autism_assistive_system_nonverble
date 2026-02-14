import cv2
import numpy as np

board = np.ones((400, 600, 3), dtype=np.uint8) * 255

items = ["Hungry", "Water", "Toilet", "Help", "Happy"]
x, y = 50, 100

for item in items:
    cv2.rectangle(board, (x,y), (x+200,y+50), (0,0,0), 2)
    cv2.putText(board, item, (x+20,y+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
    y += 60

cv2.imshow("Picture Board (AAC)", board)
cv2.waitKey(0)
cv2.destroyAllWindows()