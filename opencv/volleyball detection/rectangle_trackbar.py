import cv2 as cv

def nothing(x):
    pass

cv.namedWindow('RECT')
cv.resizeWindow('RECT',400,400)

cv.createTrackbar('MoveX','RECT',0,1280,nothing)
cv.createTrackbar('MoveY','RECT',0,720,nothing)
cv.createTrackbar('width','RECT',1,1280,nothing)
cv.createTrackbar('height','RECT',1,720,nothing)

webcam = cv.VideoCapture(r"D:\workspace\python programs\openCV\application\traces.mp4")

while True:
    isTrue,frame = webcam.read()
    if not isTrue:
        break

    if cv.waitKey(60) & 0xFF == ord('d'):
        break

    x = cv.getTrackbarPos('MoveX','RECT')
    y = cv.getTrackbarPos('MoveY','RECT')
    w = cv.getTrackbarPos('width','RECT')
    h = cv.getTrackbarPos('height','RECT')

    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
    cv.imshow('frame',frame)

while True:
    copy = frame.copy()
    x = cv.getTrackbarPos('MoveX','RECT')
    y = cv.getTrackbarPos('MoveY','RECT')
    w = cv.getTrackbarPos('width','RECT')
    h = cv.getTrackbarPos('height','RECT')

    
    cv.rectangle(copy,(x,y),(x+w,y+h),(0,255,0),thickness = 2)
    cv.putText(copy,f'{w*h}',(x-15,y-15),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv.putText(copy,f'({x},{y})',(x-35,y-35),cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)
    cv.imshow('frame',copy)
    if cv.waitKey(60) & 0xFF == ord('d'):
        break
    

webcam.release()


