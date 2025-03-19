import pickle
import time
import cv2 as cv
import numpy as np
import statistics as stats




#Modify redzones into finer regions
#Handle the faint red zones
#Create a function for green region above net increasing square like character and area threshold to stricter levels
#Start being more specific, divide the zones into smaller rectangles, if you see any false positives in limited numbers,
#take them out specifically.
#Set free zones for obvious ball paths that you see but tailor their coordinates down very particularly. the function achieving this should
#have an extremely specific area threshold that has very close min and max values as similar as to the squares around the ball
#Control these features using the frame count. get timestamps of when the ball is in these regions, and then only these conditions be relaxed
#Restrict horizontal movement of the ball around the net
rects = {}
f_count = 0

lowH, lowS, lowV = 10, 60, 80 
upH, upS, upV = 30, 255, 240

def rect_updater(x,y,w,h):
    global rects
    
    if (x,y,w,h) in rects:
        rects[(x,y,w,h)] += 1
    else:
        rects[(x,y,w,h)] = 1
        
    if rects[(x,y,w,h)] > 2:
        return False
    else:
        return True
    
def in_redzones(x,y,w,h):
    if (200 < x < 927 and 454 < y < 525) or (211 < x < 1111 and 359 < y < 454) or (1064 < x < 1111 and 495 < y < 525):
        return False
    else:
        return True
    
def in_faint_red_upper(x,y,w,h):
    pass

def in_faint_red_lower(x,y,w,h):
    pass

def in_above_net(x,y,w,h):
    if (195 < x < 1112 and 200 < y < 228) and (0 < (h*w) < 1000):
        return False
    else:
        return True
    

def true_rect(x,y,w,h):    
    if h > w:
        ratio = w / h
    else:
        ratio = h / w
        
    if  not (442 < x < 527 and 92 < y < 166):
        if not (1148 < x < 1280 and 98 < y < 317):   
            if not (558 < y < 669 and 172 < y < 216):
                if ratio > 0.75:
                    if 340 < (h*w) < 2400:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False
                                          
    #LA chnaged from 400 to 200 to 320 to 340 and RA 1100 to 1500 to 1600 to 2000
            

vid = cv.VideoCapture('volleyball_match.mp4')
img = cv.imread('sample_img.jpg')

#codec = cv.VideoWriter_fourcc(*'MP4V')
#output = cv.VideoWriter('traces.mp4', codec,30, (1280, 720))

while True:
    
    isTrue,frame = vid.read()
    if not isTrue:
        break
    
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lower = np.array([lowH, lowS, lowV])
    upper = np.array([upH, upS, upV])

    kernel = np.ones((3,3),dtype=np.uint8)
    mask = cv.inRange(hsv_frame,lower,upper)
    result = cv.bitwise_and(hsv_frame,hsv_frame,mask = mask)
    result = cv.erode(mask,kernel,iterations = 18)
    result = cv.dilate(mask,kernel,iterations = 7)

    cv.putText(frame,f'{f_count}',(40,680),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),thickness = 2)
    
    contours,_ = cv.findContours(result, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    
    for contour in contours:
        cv.drawContours(result,[contour],-1,(0,255,0))
        x, y, w, h = cv.boundingRect(contour)
        rectang = x,y,w,h

        if true_rect(x,y,w,h) and rect_updater(x,y,w,h) and in_redzones(x,y,w,h):
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),thickness = 2)
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),thickness = 2)
            print(true_rect(x,y,w,h) and rect_updater(x,y,w,h))
            
            
    cv.imshow('video',frame)
    cv.imshow('traces',img)
                       
    if cv.waitKey(100) & 0xFF == ord('d'):
        break

    f_count += 1

vid.release()
