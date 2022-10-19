import cv2
import os

#Human Face Detection using Haar Cascade Classifers

cap = cv2.VideoCapture(0)

cascadePath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

while cap.isOpened():
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Human Face Detection using Haar Cascade Classifers', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()