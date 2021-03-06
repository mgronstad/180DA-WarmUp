import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/Users/megangronstad/Documents/GitHub/180DA-WarmUp/face.xml')

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #faces multiscale detector
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
    #loop throughout the faces detected and place a box around it
    for (x,y,w,h) in faces:
        gray = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(gray,'FACE',(x, y-10), font, 0.5, (11,255,255), 2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
  
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
