import cv2
import time

#algoritmo para captura de imagens para treinar a IA

video = cv2.VideoCapture(0)

amostra = 1

while True:
    check,img = video.read()

    if cv2.waitKey(1) & 0xFF == ord('c'):
        for i in range (0, 500):
            imgR = cv2.resize(img,(32,32))
            cv2.imwrite(f'Imagens/0/Neutro{amostra}.jpg',imgR)
            print(f'imagem salva {amostra}')
            amostra +=1
    cv2.imshow('Captura', img)
    cv2.waitKey(1)