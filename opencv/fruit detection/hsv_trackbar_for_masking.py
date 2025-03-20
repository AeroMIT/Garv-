import cv2 as cv
import numpy as np

def nothing(x):
    pass

cv.namedWindow('BARS')
cv.resizeWindow('BARS',500,500)

cv.createTrackbar('H1','BARS',0,180,nothing)
cv.createTrackbar('S1','BARS',0,255,nothing)
cv.createTrackbar('V1','BARS',0,255,nothing)
cv.createTrackbar('H2','BARS',0,180,nothing)
cv.createTrackbar('S2','BARS',0,255,nothing)
cv.createTrackbar('V2','BARS',0,255,nothing)
cv.createTrackbar('dil_iter','BARS',0,100,nothing)
cv.createTrackbar('ero_iter','BARS',0,100,nothing)

img = cv.imread(r"D:\workspace\python programs\openCV\application\fruit detection\given_img.jpeg")

while True:
    
    roi = img.copy()
    hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
    H1 = cv.getTrackbarPos('H1','BARS')
    S1 = cv.getTrackbarPos('S1','BARS')
    V1 = cv.getTrackbarPos('V1','BARS')
    H2 = cv.getTrackbarPos('H2','BARS')
    S2 = cv.getTrackbarPos('S2','BARS')
    V2 = cv.getTrackbarPos('V2','BARS')
    i = cv.getTrackbarPos('dil_iter','BARS')
    j = cv.getTrackbarPos('ero_iter','BARS')

    lower = np.array([H1,S1,V1])
    upper = np.array([H2,S2,V2])

    mask = cv.inRange(hsv,lower,upper)
    mask = cv.dilate(mask,(3,3), iterations = i)
    mask = cv.erode(mask, (3,3), iterations = j)
    result = cv.bitwise_and(roi,roi,mask = mask)
    
    cv.imshow('res',result)

    if cv.waitKey(10) & 0xFF == ord('d'):
        break
    
cv.destroyAllWindows()
