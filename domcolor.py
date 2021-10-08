# code primarily adapted from https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# as well as the boundbox.py in my own repository and its references

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

cap = cv2.VideoCapture(0)

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
    
# modified to only print dom color

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
        break

    # return the bar chart
    return bar
    
while(True):
    # take in each frame
    ret, frame = cap.read()
    
    # capturing the size of the frame took a bit of trial and error
    
    img = frame[200:400, 500:700]
    # convert BGR to HSV
    #img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    img = img.reshape((img.shape[0] * img.shape[1],3))
    # set cluster to one because we want dominant color
    clust = KMeans(n_clusters=1)
    clust.fit(img)
    
    histog = find_histogram(clust)
    bar = plot_colors(histog,clust.cluster_centers_)
    cv2.rectangle(frame, (500,200),(700,400),(255,0,0))
    
    cv2.imshow('dom color',bar)
    cv2.imshow('video',frame)
    #cv2.imshow('rect',img)
    if (cv2.waitKey(5) & 0xFF) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
