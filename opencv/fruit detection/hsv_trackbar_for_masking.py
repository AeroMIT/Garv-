import cv2 as cv
import numpy as np

def nothing(x):
    pass

cv.namedWindow('BARS')
cv.resizeWindow('BARS',500,500)

cv.createTrackbar('H','BARS',0,179,nothing)
cv.createTrackbar('S','BARS',0,255,nothing)
cv.createTrackbar('V','BARS',0,255,nothing)
cv.createTrackbar('dil_iter','BARS',0,100,nothing)

img = cv.imread(r"D:\workspace\python programs\openCV\application\given_img.jpeg")

while True:
    
    roi = img.copy()
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

    H = cv.getTrackbarPos('H','BARS')
    S = cv.getTrackbarPos('S','BARS')
    V = cv.getTrackbarPos('V','BARS')
    i = cv.getTrackbarPos('dil_iter','BARS')

    lower = np.array([H,S,V])
    upper = np.array([179,255,255])

    mask = cv.inRange(hsv,lower,upper)
    mask = cv.dilate(mask,(3,3), iterations = i)
    result = cv.bitwise_and(roi,roi,mask = mask)
    
    cv.imshow('res',result)

    if cv.waitKey(10) & 0xFF == ord('d'):
        break
    
cv.destroyAllWindows()
