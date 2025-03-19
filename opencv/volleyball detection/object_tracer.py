import pickle
import cv2 as cv
import numpy as np


vid = cv.VideoCapture('volleyball_match.mp4')
img = cv.imread(r"D:\workspace\python programs\openCV\application\sample_img.jpg")
all_rect_coordinates = []
"""file = open('coordinates.dat','wb')"""

while True:
    
    isTrue,frame = vid.read()
    if not isTrue:
        break
    
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    LA = 150
    HA = 1100

    lower = np.array([6,130,120])
    upper = np.array([30,255,255])

    kernel = np.ones((3,3),dtype=np.uint8)
    
    mask = cv.inRange(hsv_frame,lower,upper)
    result = cv.bitwise_and(hsv_frame,hsv_frame,mask = mask)

    result = cv.erode(mask,kernel,iterations = 15)
    result = cv.dilate(mask,kernel,iterations = 6)
    
    contours,_ = cv.findContours(result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)           
    filtered_contours = [contour for contour in contours if LA < cv.contourArea(contour) < HA]
    
    for contour in filtered_contours:
        cv.drawContours(result,[contour],-1,(0,255,0))
        x, y, w, h = cv.boundingRect(contour)
        
    if not (427 < x < 527  and 92 < y < 166):
        cv.rectangle((img),(x,y),(x+w,y+h),(0,255,255),thickness = 2)
        all_rect_coordinates.append((x,y,x+w,y+h))
        pickle.dump((x,y,x+w,y+h),file)
        print((x,y,x+w,y+h))
    cv.imshow('traces',img)
    
        
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    
vid.release()
print(all_rect_coordinates)
cv.destroyAllWindows()
"""file.close()"""
 


