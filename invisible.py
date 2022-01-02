import cv2
import numpy as np
cap=cv2.VideoCapture(0)
back=cv2.imread('image.jpg')

while True:
    ret,frame=cap.read()
    if ret:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv',hsv)
        red=np.uint8([[[0,255,255]]])
        hsv_red=cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        # print(hsv_red)
        l_red=np.array([20,100,100])
        h_red=np.array([30,255,255])
        mask=cv2.inRange(hsv,l_red,h_red)
        # cv2.imshow('mask',mask)
        part1=cv2.bitwise_and(back,back,mask=mask)
        # cv2.imshow('part1',part1)
        mask=cv2.bitwise_not(mask)
        # cv2.imshow('not mask',mask)
        part2=cv2.bitwise_and(frame,frame,mask=mask)
        # cv2.imshow('part2',part2)

        cv2.imshow('cloak',part1+part2)



        cv2.waitKey(10)