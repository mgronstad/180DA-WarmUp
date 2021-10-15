# references include:
#  primary source for bones of object tracking code development:  https://www.analyticsvidhya.com/blog/2021/08/getting-started-with-object-tracking-using-opencv/
#  segments of the code found here was used to develop the bounded box section of the code: https://github.com/opencv/opencv/blob/master/samples/python/contours.py
#  used as reference for changing colorspace https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
#  code for video capture was developed from here: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # take in each frame
    ret, frame = cap.read()
    # convert BGR to HSV
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue in HSV
    #lower_blue = np.array([30,150,50])
    #upper_blue = np.array([255,255,180])
    
    # threshold hsv image to take in only red colors
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # define range of blue in RBG
    lower_blue = np.array([0,0,0])
    # (0, 0, 255) dark blue
    upper_blue = np.array([50,50,255])
    
    # threshold hsv image to take in only red colors
    mask = cv2.inRange(rgb, lower_blue, upper_blue)
    
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
