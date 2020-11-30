import cv2
import numpy as np
import time 


def set_upper(x):
    global colorUpper
    colorUpper[0] = x


def set_lower(x):
    global colorLower
    colorLower[0] = x


cam = cv2.VideoCapture(0)
cv2.namedWindow('camera', cv2.WINDOW_KEEPRATIO)
cv2.namedWindow('mask', cv2.WINDOW_KEEPRATIO)
cv2.createTrackbar("U", 'Mask', 0, 255, set_upper)
cv2.createTrackbar("L", 'Mask', 0, 255, set_lower)

colorLower = np.array([0, 100, 100], dtype='uint8')
colorUpper = np.array([255, 255, 255], dtype='uint8')


def draw(cnts, frame):
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (curr_x, curr_y), radii = cv2.minEnclosingCircle(c)
        if radii > 20:
            cv2.circle(frame, (int(curr_x), int(curr_y)),
                       int(radii), (0, 255, 255), 2)
            cv2.circle(frame, (int(curr_x), int(curr_y)), 5, (0,0,255), -1)
    


def get_cnts(hsv, colorLower, colorUpper):
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # cv2.imshow('mask', mask)

    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cnts

prev_time = time.time()
curr_time = time.time()

prev_x = 0
prev_y = 0 
curr_x = 0
curr_y = 0


while cam.isOpened():
    ret, frame = cam.read()
    
    blurred=cv2.GaussianBlur(frame,(11,11),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv, colorLower, colorUpper)
    mask=cv2.erode(mask, None, iterations=2)
    mask=cv2.dilate(mask, None, iterations=2)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    cnts = get_cnts(hsv, np.array([64, 100, 100]), np.array([86,255,255]))
    draw(cnts, frame)

    cnts = get_cnts(hsv, np.array([13, 100, 100]), np.array([30,255,255]))
    draw(cnts, frame)
    
    cnts = get_cnts(hsv, np.array([95, 100, 100]), np.array([110,255,255]))
    draw(cnts, frame)
    
    time_diff = curr_time - prev_time
    prev_time = curr_time
    
    dist = ((prev_x - curr_x) ** 2 + (prev_y - curr_y) ** 2) ** 0.5
    prev_x = curr_x
    prev_y = curr_y
    
    velocity = dist/time_diff
    
    cv2.putText(frame, f"Dist = {dist:1f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
    cv2.imshow('mask', mask)
    cv2.imshow('camera', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
