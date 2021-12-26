import os

import cv2
import sys
from deepface import DeepFace

abspath = os.path.abspath("haarcascade_frontalface_default.xml")
##cascPath= os.path.dirname(abspath)
faceCascade = cv2.CascadeClassifier(abspath)



video_capture = cv2.VideoCapture(0)
coun= 0
while True:
    coun= coun+1
    # Capture frame-by-frame
    ret, frame = video_capture.read()


    if coun > 10:
     result = DeepFace.analyze(frame, actions=['emotion'])




    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if coun > 10:
        cv2.putText(frame,
                    result['dominant_emotion'],
                    (50, 50),
                    font, 1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)



        # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
