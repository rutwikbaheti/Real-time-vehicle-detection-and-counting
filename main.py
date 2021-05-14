import cv2
import numpy as np
from time import sleep

min_width=80 #Minimum rectangle width
min_height=80 #Minimum rectangle height

offset=6 #Allowable error between pixel  

count_line=250#550 #Count line position 

delay= 60 #Video FPS

detect = []
cars= 0

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('6.mp4')
subtraction = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    time = float(1/delay)
    sleep(time) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtraction.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (10, count_line), (1200, count_line), (255,127,0), 3) 
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= min_width) and (h >= min_height)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        source_img=frame1[y:y+h, x:x+w]
        cv2.imwrite(r'C:\Users\Kundan\Desktop\minipro\detected_vehicles\vehicle' + str(cars)+ '.png',source_img)
        centro = get_center(x, y, w, h)
        detect.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)

        for (x,y) in detect:
            if y<(count_line+offset) and y>(count_line-offset):
                cars+=1
               # cv2.line(frame1, (25, count_line), (1200, count_line), (0,127,255), 3)  
                cv2.line(frame1, (10, count_line), (1200, count_line), (0,127,255), 3)
                detect.remove((x,y))
                print("car is detected : "+str(cars))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(cars), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    #cv2.imshow("detectar",img_sub)
    #cv2.imshow("detectar",dilated)
    #cv2.imshow("detectar",dilat)
    #cv2.imshow("detectar",grey)
    #cv2.imshow("detectar",blur)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()