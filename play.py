import cv2
import time
import numpy as np
cap = cv2.VideoCapture("./WT9.avi")
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))


x1 = 50
x2 = 150
y1 = 50
y2 = 200

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    # cv2.imwrite("frame%d.jpg" % count, frame)
    count = count + 1
    print(count)
    frame = frame[x1:x2, y1:y2]
    fgmask = fgbg.apply(frame)

    if count < 2200:
        continue
    time.sleep(0.10)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = np.stack([fgmask for _ in range(3)], axis=2)
    frame = cv2.hconcat([frame, fgmask])
    cv2.imshow('window-name', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  # destroy all opened windows
