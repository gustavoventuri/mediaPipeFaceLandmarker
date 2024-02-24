import mediapipe as mp
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from PIL import Image
from mss import mss

captura = 'webcam'
bouning_box = {'top':460, 'left':1980, 'width': 800, 'height': 400}
#video da webcam
if captura == 'webcam':
    video = cv2.VideoCapture(0)
else:
    video = mss()
#video de captura de tela


fc = mp.solutions.face_detection
fr = fc.FaceDetection()
dr = mp.solutions.drawing_utils

while True:
    if captura == 'webcam':
        r, img = video.read()

        if not r:
            break        
    else:
        sct = video.grab(bouning_box)
        img = sct.bgra

    rostos = fr.process(img)

    if rostos.detections:
        for rosto in rostos.detections:
            dr.draw_detection(img,rosto)

    cv2.imshow(captura,img)

    if cv2.waitKey(5) == 27:
        break

video.release()
cv2.destroyAllWindows()
