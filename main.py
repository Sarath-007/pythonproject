#background
import cv2
cap=cv2.VideoCapture(0)
while True:
    ret,back=cap.read()
    if ret:
        cv2.imshow('background',back)
        if cv2.waitKey(5)==ord('q'):
            cv2.imwrite('image.jpg',back)

