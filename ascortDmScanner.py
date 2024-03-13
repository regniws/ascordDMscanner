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


#constans
windowName = "Ascort:DM scanner"
version = 2

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
        'для завершения нажмите любую кнопку',
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

def renderWithUIN(image, UIN):
    
    overlay = image.copy() 
    
    # рамка датаматрикса, для повёрнутых кодов pylibdmtx даёт некорректную ширину
    if rectWork:
        cv2.rectangle(
            overlay, 
            (int(image.shape[1] / 2  - aimSize), int(image.shape[0] / 2  - aimSize / 4)),
            (int(image.shape[1] / 2  + aimSize), int(image.shape[0] / 2  + aimSize / 4)),
            (135,255,169),
            -1
        )
    else:
       cv2.rectangle(
            overlay, 
            (int(image.shape[1] / 2  - aimSize / 2), int(image.shape[0] / 2  - aimSize / 2)),
            (int(image.shape[1] / 2  + aimSize / 2), int(image.shape[0] / 2  + aimSize / 2)),
            (135,255,169),
            -1
       )
    
    image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    
    # определённый уин
    image = cv2.putText(
        image,
        UIN,
        (30, 30),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1,
        (132,255,56),
        2
    )
    
    return image

def render(plain, image, points, UIN):
    if plain:
        renderPlain(image)
    else:
        image = renderWithUIN(image, UIN)
        
    if rectWork:
       image = cv2.rectangle(
            image, 
            (int(image.shape[1] / 2  - aimSize), int(image.shape[0] / 2  - aimSize / 4)),
            (int(image.shape[1] / 2  + aimSize), int(image.shape[0] / 2  + aimSize / 4)),
            (135,255,169),
            3
       )
    else:
        image = cv2.rectangle(
            image, 
            (int(image.shape[1] / 2  - aimSize / 2), int(image.shape[0] / 2  - aimSize / 2)),
            (int(image.shape[1] / 2  + aimSize / 2), int(image.shape[0] / 2  + aimSize / 2)),
            (135,255,169),
            3
    )
    cv2.imshow(windowName, image)
    
def morphology(binary):
    rez = []
    
    for i in [2,3]:
        for j in [2,3]:
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (i, i))
            open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element, iterations=1)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (j, j))
            closed = cv2.morphologyEx(open, cv2.MORPH_CLOSE, element, iterations=1)
            rez.append(closed)
    return rez

def prepareImage(image):
        
    imagesForDebug = []
    if rectWork:
        cropAim = image[int(image.shape[0] / 2  - aimSize / 3.5): int(image.shape[0] / 2  + aimSize / 3.5), int(image.shape[1] / 2  - aimSize ): int(image.shape[1] / 2  + aimSize )]
    else:
        cropAim = image[ int(image.shape[0] / 2  - aimSize / 2): int(image.shape[0] / 2  + aimSize / 2), int(image.shape[1] / 2  - aimSize / 2): int(image.shape[1] / 2  + aimSize / 2)]
    
    imagesForDebug.append(cropAim)
    gray = cv2.cvtColor(cropAim, cv2.COLOR_BGR2GRAY)
    imagesForDebug.append(gray)
    
    blur = cv2.GaussianBlur(gray,(5,5),0)
    imagesForDebug.append(blur)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    imagesForDebug.append(thresh)
    morph = morphology(thresh)
    
    for t in morph:
        imagesForDebug.append(t)
    
    if debug:
        cnt = 0
        for i in imagesForDebug:
            cv2.imshow('debug '+ str(cnt), i)
            cnt = cnt + 1
    return morph

def proccessDMCode(image):
    
    imagesForScan = prepareImage(image)
    
    for image in imagesForScan:
        if rectWork:
            data = decode(image, accuracy)
        else:
            data = decode(image, accuracy, shape=2)
        
        for decodedObject in data:
            
            points = [
                (decodedObject.rect.left,  image.shape[0] - decodedObject.rect.top), 
                (decodedObject.rect.left + decodedObject.rect.width,  image.shape[0] - decodedObject.rect.top - decodedObject.rect.height)
            ]
            print(decodedObject.data.decode("utf-8"))
            return (True,  points, decodedObject.data.decode("utf-8"))
    
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

parser.add_argument(
    "--aimSize", "-as",
    help="the aim box size in pixelx",
    type = int,
    default = 140
)

parser.add_argument(
    "--debug", "-d",
    help="show debugging view",
    type = bool,
    default = False
)

parser.add_argument(
    "--rect", "-rt",
    help="scan for rects",
    type = bool,
    default = False
)

args = parser.parse_args()

resultFile = args.resultFile
camID = args.camID
accuracy = args.accuracy
copyToClipboard = args.clipboard
readFromFile = args.readFromFile
debug = args.debug
aimSize = args.aimSize
rectWork = args.rect

cap = cv2.VideoCapture(camID)

#event loop

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
    
    k = cv2.waitKey(33)
    if k == -1:
        continue
    else:
        break

cv2.destroyWindow(windowName)
cap.release()
sys.exit(0)
