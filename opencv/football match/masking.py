import cv2 as cv
from trackbars import Trackbar
import numpy as np
def resize(frame, scale = 0.75):
    interpolation = cv.INTER_LINEAR if scale > 1 else cv.INTER_AREA
    
    width = int(frame.shape[1] * scale)
    length = int(frame.shape[0] * scale)
    dimensions = (width, length)
    return cv.resize(frame, dimensions, interpolation)

path = r"C:\Users\GARV OFFLINE\Desktop\python programs\openCV\application\football match\football1.mov"

vid = cv.VideoCapture(path)
k = 1
tb = Trackbar(mode = 'mask')
while True:
    
    isTrue, frame = vid.read()
    msk, res = tb.get(frame)
    
    cv.imshow('res', resize(res,0.4))
    
    if cv.waitKey(k) & 0xFF == ord('a'):
        k = k*10
        print('reduced-playback-speed')
    if cv.waitKey(k) & 0xFF == ord('d'):
        k = max(1, k//10)
        print('increased-playback-speed')
    if cv.waitKey(k) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
