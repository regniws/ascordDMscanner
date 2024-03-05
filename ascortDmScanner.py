# Данный модуль позволяет сканировать ДМ коды с помощью веб-камеры
# Используются библиотека libdmtx в py обёртке
# Для работы необходимо установить связаные модули
# pip install pylibdmtx
# pip install opencv-python
# pip install pyperclip
#
# Распространяется без поддержки и каких-либо гарантий. Доступно
# коммерческое и не коммерческое использование и вообще любое 
# использование при условии сохранения авторства. 
#
# Myagkov Anton @ wow1c
# a@wow1c.com, 2023

from pylibdmtx.pylibdmtx import decode
import argparse
import cv2
import pyperclip
import sys
import dottedDataMatrix as ddm
import numpy as np

import matplotlib.pyplot as plt
import PIL


#constans
windowName = "Ascort:DM scanner"
version = 1

def renderPlain(image):

    image = cv2.putText(
        image,
        'Поиск DM',
        (30, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 55, 108),
        1
    )

    image = cv2.putText(
        image,
        'press any key',
        (30, 450),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 55, 108),
        1
    )

    image = cv2.putText(
        image,
        'Версия ' + str(version),
        (520, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        0.5,
        (255, 55, 108),
        1
    )    

def renderWithUIN(image, plain, points, UIN):

    # рамка датаматрикса, для повёрнутых кодов pylibdmtx даёт некорректную ширину
    image = cv2.rectangle(
        image, 
        points[0],
        points[1],
        (132,255,56),
        5
    )
    
    # определённый уин
    image = cv2.putText(
        image,
        UIN,
        (30, 30),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (132,255,56),
        2
    )

def render(plain, image, points, UIN):
    if plain:
        renderPlain(image)
    else:
        renderWithUIN(image, plain, points, UIN)
    cv2.imshow(windowName, image)
    

def proccessDMCode(image):

    alg = ddm.DottedDataMatrix()
    data = alg.detect_datamatrix(image, True, True)
    if data is not None:
        points = [(0,0), (0,0)]
        return (True, points, data[1])
    # data = decode(image, accuracy, shape=2)

    # for decodedObject in data:
    #
    #     points = [
    #         (decodedObject.rect.left,  image.shape[0] - decodedObject.rect.top),
    #         (decodedObject.rect.left + decodedObject.rect.width,  image.shape[0] - decodedObject.rect.top - decodedObject.rect.height)
    #     ]
    #     print(decodedObject.data.decode("utf-8"))
    #     return (True,  points, decodedObject.data.decode("utf-8"))

    return (False, None, None)
       
#main

parser=argparse.ArgumentParser(
    prog ='Ascort:DM scanner ' + str(version),
    description ='datamatrix codes scanners with USB-cameras',
    epilog = '''Usages examples: \n
    ascortDmScanner // open the program \n
    ascortDmScanner --camID=1 --clipboard=True // open and scan with camerID = 1 \n
    ascortDmScanner --readFromFile='qr.jpg' --resultFile='result.txt'  // open file with image and save result to  result.txt'''
)

parser.add_argument(
    "--resultFile", "-r",
    help="file with result",
    type = str,
    default = ''
)

parser.add_argument(
    "--camID", "-c",
    help="USB camera ID",
    dest="camID",
    type = int,
    default = 0
)

parser.add_argument(
    "--accuracy",
    "-a", help="accuracy in milliseconds",
    type = int,
    default = 70
)

parser.add_argument(
    "--clipboard", "-C",
    help="copy resul to clipboard",
    type = bool,
    default = False
)

parser.add_argument(
    "--readFromFile", "-f",
    help="the image file to read",
    type = str,
    default = ''
)

args = parser.parse_args()

resultFile = args.resultFile
camID = args.camID
accuracy = args.accuracy
copyToClipboard = args.clipboard
readFromFile = args.readFromFile

cap = cv2.VideoCapture(camID)

#event loop
it1 = 0

while True:
    if readFromFile != '':
        image = cv2.imread(readFromFile)
    else:
        rez, image = cap.read()
        if not rez:
            print('can\'t attach to camera')
            sys.exit(-1)

    # Если требуется прочитал 
    #

    (found, points, UIN) = proccessDMCode(image)

    render(not found, image, points, UIN)

    if found:
        if resultFile != '':
            with open(resultFile, 'w') as f:
                f.write(UIN)
                f.close()
                break
        elif copyToClipboard:
            pyperclip.copy(UIN)
            break
    break

    k = cv2.waitKey(33)
    if k == -1:
        continue
    else:
        break

cv2.destroyWindow(windowName)
cap.release()
sys.exit(0)
