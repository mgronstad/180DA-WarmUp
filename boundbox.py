# references include:
# https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_canny/py_canny.html#canny
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # take in each frame
    ret, frame = cap.read()
    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of red in HSV
    lower_red = np.array([0,230,170])
    upper_red = np.array([255,255,220])
    
    # threshold hsv image to take in only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    if (cv2.waitKey(5) & 0xFF) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
