import cv2 as cv
import numpy as np
from trackbars import Trackbar

def resize(frame, scale = 0.75):
    interpolation = cv.INTER_LINEAR if scale > 1 else cv.INTER_AREA
    
    width = int(frame.shape[1] * scale)
    length = int(frame.shape[0] * scale)
    dimensions = (width, length)
    return cv.resize(frame, dimensions, interpolation)

def _audience_remover(hsv_frame):
    
    hsv_frame = cv.blur(hsv_frame, (15,15))
    frame = cv.cvtColor(hsv_frame, cv.COLOR_HSV2BGR)
    kernel = np.ones((3,3), dtype = np.uint8)
    aud_lower = np.array([16/2, 0*255/100, 0*255/100], dtype = np.uint8)
    aud_upper = np.array([200/2, 100*255/100, 100*255/100], dtype = np.uint8)
    audience_mask = cv.inRange(hsv_frame, aud_lower, aud_upper)
    
    audience_mask = cv.erode(audience_mask, kernel, iterations = 4)
    audience_mask = cv.dilate(audience_mask, kernel, iterations = 9)

    return np.uint8(audience_mask)


def _blue_goalie_blocker(hsv_frame):

    kernel = np.ones((3,3), dtype = np.uint8)
    b_lower = np.array([210/2, 54*255/100, 39*255/100], dtype = np.uint8)
    b_upper = np.array([230/2, 100*255/100, 100*255/100], dtype = np.uint8)
    b_mask = cv.inRange(hsv_frame, b_lower, b_upper)
    b_mask = cv.dilate(b_mask, kernel, iterations = 12)
    inverted_b_mask = cv.bitwise_not(b_mask)

    return np.uint8(inverted_b_mask)


def _left_billboard(hsv_frame):
    
    hsv_frame = cv.medianBlur(hsv_frame, 9)
    kernel1 = np.ones((5,5), dtype = np.uint8)
    kernel2 = np.ones((5,5), dtype = np.uint8)
    
    bill_lower = np.array([0, 0, 0], dtype = np.uint8)
    bill_upper = np.array([10/2, 100*255/100, 100*255/100], dtype = np.uint8)
    billboard_mask = cv.inRange(hsv_frame, bill_lower, bill_upper)
    billboard_mask = cv.dilate(billboard_mask, kernel1, iterations = 25)
    billboard_mask = cv.erode(billboard_mask, kernel1, iterations = 4)
    inverted_billboard_mask = cv.bitwise_not(billboard_mask)
    
    bb_text_lower = np.array([0/2, 0*255/100, 80*255/100], dtype = np.uint8)
    bb_text_upper = np.array([360/2, 20*255/100, 100*255/100], dtype = np.uint8)
    bb_text_mask = cv.inRange(hsv_frame, bb_text_lower, bb_text_upper)
    bb_text_mask = cv.erode(bb_text_mask, kernel2, iterations = 3)
    bb_text_mask = cv.dilate(bb_text_mask, kernel2, iterations = 100)
    inverted_bb_text_mask = cv.bitwise_not(bb_text_mask)

    net_billboard_mask = cv.bitwise_and(inverted_billboard_mask, inverted_bb_text_mask)

    return np.uint8(net_billboard_mask)


def _blue_billboard(hsv_frame):

    kernel = np.ones((21,21), dtype = np.uint8)
    blue_lower = np.array([220/2, 80*255/100, 40*255/100], dtype = np.uint8)
    blue_upper = np.array([222/2, 90*255/100, 55*255/100], dtype = np.uint8)
    blue_mask = cv.inRange(hsv_frame, blue_lower, blue_upper)
    blue_mask = cv.erode(blue_mask, np.ones((3,3), dtype = np.uint8), iterations = 2)
    blue_mask = cv.dilate(blue_mask, kernel, iterations = 50)

    inverted_blue_mask = cv.bitwise_not(blue_mask)
    
    return np.uint8(inverted_blue_mask)


def _skin(hsv_frame):

    skin_lower = np.array([10/2, 20*255/100, 10*255/100], dtype = np.uint8)
    skin_upper = np.array([30/2, 60*255/100, 40*255/100], dtype = np.uint8)
    skin_mask = cv.inRange(hsv_frame, skin_lower, skin_upper)
    skin_mask = cv.erode(skin_mask, kernel = np.ones((5,5), dtype = np.uint8), iterations = 2)
    skin_mask = cv.dilate(skin_mask, kernel = np.ones((21,21), dtype = np.uint8), iterations = 100)
    
    inverted_skin_mask = cv.bitwise_not(skin_mask)

    return np.uint8(inverted_skin_mask)


def _ball_blocker(hsv_frame, mask):
    
    hsv_frame = cv.bitwise_and(hsv_frame, hsv_frame, mask = mask)
    
    kernel = np.ones((21,21), dtype = np.uint8)

    ball_lower = np.array([280/2, 42*255/100, 40*255/100], dtype = np.uint8)
    ball_upper = np.array([315/2, 75*255/100, 70*255/100], dtype = np.uint8)
    ball_mask = cv.inRange(hsv_frame, ball_lower, ball_upper)
    ball_mask = cv.dilate(ball_mask, kernel, iterations = 80)

    green_goalie_lower = np.array([78/2, 90*255/100, 60*255/100], dtype = np.uint8)
    green_goalie_upper = np.array([90/2, 100*255/100, 100*255/100], dtype = np.uint8)
    green_goalie_mask = cv.inRange(hsv_frame, green_goalie_lower, green_goalie_upper)
    green_goalie_mask = cv.erode(green_goalie_mask, kernel, iterations = 0)
    green_goalie_mask = cv.dilate(green_goalie_mask, kernel, iterations = 40)

    inverted_ball_mask = cv.bitwise_not(ball_mask)
    inverted_goalie_mask = cv.bitwise_not(green_goalie_mask)
   
    combined_ball_mask = cv.bitwise_and(inverted_goalie_mask, inverted_ball_mask)

    frame = cv.cvtColor(hsv_frame, cv.COLOR_HSV2BGR)
    result = cv.bitwise_and(frame, frame, mask = combined_ball_mask)
    
    return np.uint8(combined_ball_mask)

    
def preprocessed(frame):
    
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    audience_mask = _audience_remover(hsv_frame)
    bb_mask = _left_billboard(hsv_frame)
    inverted_blue_mask = _blue_billboard(hsv_frame)
    blue_goalie_mask = _blue_goalie_blocker(hsv_frame)
    skin_mask = _skin(hsv_frame)
    
    hsv_frame_green = cv.medianBlur(hsv_frame, 5)

    blank_mask = np.ones(frame.shape, dtype = np.uint8)*255
    cv.rectangle(blank_mask, (0,0), (576,100), (0,0,0), thickness = -1)
    cv.rectangle(blank_mask, (0,325), (576,325+35), (0,0,0), thickness = -1)
    blank_mask = cv.cvtColor(blank_mask, cv.COLOR_BGR2GRAY)

    combined_ball_mask = _ball_blocker(hsv_frame, blank_mask)
    
    kernel = np.ones((3,3), dtype = np.uint8)
    green_lower = np.array([76/2, 0*255/100, 0*255/100], dtype = np.uint8)
    green_upper = np.array([200/2, 100*255/100, 100*255/100], dtype = np.uint8)
    green_mask = cv.inRange(hsv_frame_green, green_lower, green_upper)
    green_mask = cv.dilate(green_mask, kernel, iterations = 0)
    green_mask = cv.erode(green_mask, kernel, iterations = 10)
    '''
    #maybe useful for big red mask
    add_red_lower = np.array([0/2, 50*255/100, 60*255/100], dtype = np.uint8)
    add_red_upper = np.array([12/2, 90*255/100, 80*255/100], dtype = np.uint8)
    add_red_mask = cv.inRange(hsv_frame, add_red_lower, add_red_upper)
    add_red_mask = cv.dilate(green_mask, kernel, iterations = 0)
    add_red_mask = cv.erode(green_mask, kernel, iterations = 1)
    cv.imshow('additional', add_red_mask)
    '''
    inverted_green_mask = cv.bitwise_not(green_mask)
    
    combined_green_mask = cv.bitwise_and(inverted_green_mask, blank_mask)
    combined_green_mask = cv.bitwise_and(combined_green_mask, audience_mask)
    combined_green_mask = cv.bitwise_and(combined_green_mask, bb_mask)
    combined_green_mask = cv.bitwise_and(combined_green_mask, combined_ball_mask)
    combined_green_mask = cv.bitwise_and(combined_green_mask, inverted_blue_mask)
    combined_green_mask = cv.bitwise_and(combined_green_mask, blue_goalie_mask)
    combined_green_mask = cv.bitwise_and(combined_green_mask, skin_mask)

    cv.imshow('mask', combined_green_mask)
                           
    return np.uint8(combined_green_mask)


def red_aerial_zoom(frame, mask):
    
    kernel = np.ones((3,3), dtype = np.uint8)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    red_lower = np.array([350/2, 50*255/100, 50*255/100], dtype = np.uint8)
    red_upper = np.array([360/2, 100*255/100, 100*255/100], dtype = np.uint8)
    red_mask = cv.inRange(hsv_frame, red_lower, red_upper)
    red_mask = cv.erode(red_mask, kernel, iterations = 0)
    red_mask = cv.dilate(red_mask, kernel, iterations = 6)
    
    final_red_mask = cv.bitwise_and(red_mask, mask)
    edited_frame = cv.bitwise_and(frame, frame, mask = final_red_mask)

    return np.uint8(final_red_mask)


def black_aerial_zoom(frame, mask):
    
    kernel = np.ones((3,3), dtype = np.uint8)
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_frame = cv.blur(hsv_frame, (3,3))
    
    black_lower = np.array([90/2, 0*255/100, 0*255/100], dtype = np.uint8)
    black_upper = np.array([360/2, 40*255/100, 36*255/100], dtype = np.uint8)
    black_mask = cv.inRange(hsv_frame, black_lower, black_upper)
    black_mask = cv.erode(black_mask, kernel, iterations = 0)
    black_mask = cv.dilate(black_mask, kernel, iterations = 5)
    
    final_black_mask = cv.bitwise_and(black_mask, mask)

    return np.uint8(final_black_mask)


def blue_goalkeeper(frame):

    blank_mask = np.ones(frame.shape, dtype = np.uint8)
    cv.rectangle(blank_mask, (0,0), (576,80), (0,0,0), thickness = -1)
    blank_mask = cv.cvtColor(blank_mask, cv.COLOR_BGR2GRAY)

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    skin_mask = _skin(hsv_frame)

    blue_lower = np.array([212/2, 70*255/100, 60*255/100], dtype = np.uint8)
    blue_upper = np.array([220/2, 100*255/100, 80*255/100], dtype = np.uint8)
    blue_mask = cv.inRange(hsv_frame, blue_lower, blue_upper)
    blue_mask = cv.dilate(blue_mask, kernel = np.ones((3,3), dtype = np.uint8), iterations = 8)

    blue_mask = cv.bitwise_and(blue_mask, blank_mask)
    blue_mask = cv.bitwise_and(blue_mask, skin_mask)

    return np.uint8(blue_mask)

def green_goalkeeper(frame):
    #not perfect... increased noise. this section needs its own masks
    blank_mask = np.ones(frame.shape, dtype = np.uint8)
    cv.rectangle(blank_mask, (0,0), (576,80), (0,0,0), thickness = -1)
    blank_mask = cv.cvtColor(blank_mask, cv.COLOR_BGR2GRAY)

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    skin_mask = _skin(hsv_frame)
    
    lime_lower = np.array([80/2, 60*255/100, 60*255/100], dtype = np.uint8)
    lime_upper = np.array([84/2, 100*255/100, 100*255/100], dtype = np.uint8)
    lime_mask = cv.inRange(hsv_frame, lime_lower, lime_upper)
    lime_mask = cv.erode(lime_mask, kernel = np.ones((3,3),dtype = np.uint8), iterations = 1)
    lime_mask = cv.dilate(lime_mask, kernel = np.ones((5,5),dtype = np.uint8), iterations = 4)

    lime_mask = cv.bitwise_and(lime_mask, blank_mask)
    lime_mask = cv.bitwise_and(lime_mask, skin_mask)
    cv.imshow('lime', lime_mask)

    return np.uint8(lime_mask)


def preprocessed2(frame):

    frame = cv.blur(frame, (15, 15))
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blank = np.ones(frame.shape, dtype = np.uint8)*255
    cv.rectangle(blank, (25, 31), (25+167, 31+18), (0,0,0), -1)
    cv.rectangle(blank, (534, 26), (534+31, 26+34), (0,0,0), -1)
    cv.rectangle(blank, (547, 331), (547+23, 331+21), (0,0,0), -1)
    blank = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)

    green_lower = np.array([0/2, 0*255/100,0*255/100], dtype = np.uint8)
    green_upper = np.array([190/2, 100*255/100, 100*255/100], dtype = np.uint8)
    green_mask = cv.inRange(hsv_frame, green_lower, green_upper)
    green_mask = cv.erode(green_mask, kernel = np.ones((5,5), dtype = np.uint8), iterations = 2)
    green_mask = cv.erode(green_mask, kernel = np.ones((5,5), dtype = np.uint8), iterations = 12)

    return np.uint8(green_mask)

    
path = "football1.mov"
vid = cv.VideoCapture(path)
resize_factor = 0.2

fourcc = cv.VideoWriter_fourcc(*'H264')
op = cv.VideoWriter('aerial_shots.mp4', fourcc, 30.0, (576, 360))

k = 1
while True:
    
    isTrue, frame = vid.read()

    if not isTrue:
        break

    frame = resize(frame, resize_factor)
    output = frame.copy()
    frame = cv.blur(frame, (3,3))

    if cv.waitKey(k) & 0xFF == ord('a'):
        k = k*10
        print(f'playback speed decreased to {k}.')

    if cv.waitKey(k) & 0xFF == ord('d'):
        k = max(1, k//10)
        print(f'playback speed increased to {k}.')
        
    if cv.waitKey(k) & 0xFF == ord('q'):
        break

    mask1 = preprocessed(frame)
    
    red_mask = red_aerial_zoom(frame, mask1)
    blue_mask = blue_goalkeeper(frame)
    red_mask_1 = cv.bitwise_or(red_mask, blue_mask)
    
    black_mask = black_aerial_zoom(frame, mask1)
    lime_mask = green_goalkeeper(frame)
    black_mask_1 = cv.bitwise_or(black_mask, lime_mask)

    mask2 = preprocessed2(frame)
    

    red_contours_1, _ = cv.findContours(red_mask_1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    red_rects = []
    for contour in red_contours_1:
        cv.drawContours(red_mask_1, [contour], 0, (0,255,0))
        x, y, w, h = cv.boundingRect(contour)
        if  160 < (w*h) < 2000 and (w/h < 2 and h/w < 2):
            cv.rectangle(output, (x,y), (x+w,y+h), (0,0,255), thickness = 2)
            red_rects.append((x,y,w,h))

            
    cv.putText(output, f'{len(red_rects)}', (500,80), cv.FONT_HERSHEY_SIMPLEX, 1, color = (0,0,255), thickness = 2)

    
    black_contours_1, _ = cv.findContours(black_mask_1, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    black_rects = []
    for contour in black_contours_1:
        cv.drawContours(black_mask_1, [contour], 0, (0,255,0))
        x, y, w, h = cv.boundingRect(contour)
        if  160 < (w*h) < 2000 and w/h < 2:
            cv.rectangle(output,(x,y),(x+w,y+h),(0,0,0),thickness = 2)
            black_rects.append((x,y,w,h))

    cv.putText(output, f'{len(black_rects)}', (500,120), cv.FONT_HERSHEY_SIMPLEX, 1, color = (0,0,0), thickness = 2)

    cv.imshow('final', output)
    op.write(output)

vid.release()
op.release()
cv.destroyAllWindows()
