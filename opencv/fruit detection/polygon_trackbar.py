#NUMBER OF POINTS CAN BE MODIFIED

import cv2 as cv
import numpy as np

def nothing(x):
    pass

cv.namedWindow('POLY')
cv.resizeWindow('POLY',750,750)
img = cv.imread(r"D:\workspace\python programs\openCV\application\sample_img.jpg")

cv.createTrackbar('X1','POLY',0,1280,nothing)
cv.createTrackbar('X2','POLY',0,1280,nothing)
cv.createTrackbar('X3','POLY',0,1280,nothing)
cv.createTrackbar('X4','POLY',0,1280,nothing)


cv.createTrackbar('Y1','POLY',0,720,nothing)
cv.createTrackbar('Y2','POLY',0,720,nothing)
cv.createTrackbar('Y3','POLY',0,720,nothing)
cv.createTrackbar('Y4','POLY',0,720,nothing)

while True:
    copy = img.copy()
    x1 = cv.getTrackbarPos('X1','POLY')
    x2 = cv.getTrackbarPos('X2','POLY')
    x3 = cv.getTrackbarPos('X3','POLY')
    x4 = cv.getTrackbarPos('X4','POLY')

    y1 = cv.getTrackbarPos('Y1','POLY')
    y2 = cv.getTrackbarPos('Y2','POLY')
    y3 = cv.getTrackbarPos('Y3','POLY')
    y4 = cv.getTrackbarPos('Y4','POLY')

    points = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    
    cv.polylines(copy,[points],isClosed = True, color = (0,255,0),thickness = 2)
    cv.putText(copy,f'({x1},{y1})',(x1-30,y1-30),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv.putText(copy,f'({x2},{y2})',(x2-30,y2-30),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv.putText(copy,f'({x3},{y3})',(x3-30,y3-30),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv.putText(copy,f'({x4},{y4})',(x4-30,y4-30),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv.imshow('window',copy)

    if cv.waitKey(10) & 0xFF == ord('d'):
        break

cv.imwrite('redzone.jpg',copy)
