import cv2 as cv
import numpy as np

img = cv.imread(r"D:\workspace\python programs\openCV\application\fruit detection\given_img.jpeg")
final_img, red_copy, purple_copy = img.copy(), img.copy(), img.copy()


purple_hsv = cv.cvtColor(purple_copy,cv.COLOR_BGR2HSV)
purple_L = np.array([121, 34, 59])
purple_H = np.array([180, 255, 255])
purple_mask = cv.inRange(purple_hsv, purple_L, purple_H)
purple_mask = cv.dilate(purple_mask,(3,3), iterations = 0)
purple_mask = cv.erode(purple_mask,(3,3), iterations = 4)

purple_contours, _ = cv.findContours(purple_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
purple_filtered_contours = [cont for cont in purple_contours if cv.contourArea(cont) > 40]

purple_count = 0
for contour in purple_filtered_contours:
    purple_count += 1 
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(final_img, (x-10, y-10), (x+w+10, y+h+10), (255, 0, 255), 2)
    cv.putText(final_img,f'PURPLE FRUIT {purple_count}',(x-20,y-20),cv.FONT_HERSHEY_TRIPLEX,0.6,(255, 0, 255),thickness = 2)


red_hsv = cv.cvtColor(red_copy, cv.COLOR_BGR2HSV)
red_L = np.array([0, 75, 0])
red_H = np.array([14, 204, 255])
red_mask = cv.inRange(red_hsv, red_L, red_H)
red_mask = cv.dilate(red_mask,(3,3), iterations = 2)
red_mask = cv.erode(red_mask,(3,3), iterations = 3)

red_contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
red_filtered_contours = [cont for cont in red_contours if cv.contourArea(cont) > 7]

red_count = 0
for contour in red_filtered_contours:
    red_count += 1 
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(final_img, (x-10, y-10), (x+w+10, y+h+10), (0, 0, 255), 2)
    cv.putText(final_img,f'RED FRUIT {red_count}',(x-20,y-20),cv.FONT_HERSHEY_TRIPLEX,0.6,(0,0,255),thickness = 2)

cv.imshow('detected_fruits', final_img)
cv.imwrite('detected_fruits_2.jpg', final_img)
