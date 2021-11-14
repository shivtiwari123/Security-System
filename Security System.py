''' SECURITY SYSTEM '''

import numpy as np
import cv2
import time
import datetime

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Face Detection link.
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") # Eye detection link.
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml") # Body detection link.
detection = False
timer_started = False
detection_stopped_time = None
SECONDS_TO_RECORD_AFTER_DETECTION = 5
frame_size = (int(cap.get(3)), int(cap.get(4))) # Recording window frame should be same as diplayong frame.
fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Setting video recording format as MP4.
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Starts detecting faces.
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5) # Starts detecting bodies. 


    if len(faces) + len(bodies) > 0: # Recroding if and when faces and bodies are detected.
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S") # Stores date and time of current recording.
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size) # Creating a output stream.
            print("Started recording")
    elif detection: # To keep recording for a while after a body has gone out of frame.
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stopped recording")  
        else:
            timer_started = True
            detection_stopped_time = time.time()
    if detection:
        out.write(frame)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 5) # Draws rectangle around faces.
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5) # Starts detecting eyes.
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5) # Draws rectangle around eyes.

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("x"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()