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
        'Нажмите любую клавишу для завершения',
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
    prog ='Аскорт:Сканнер ДМ кодов. Версия ' + str(version),
    description ='Программа для распознавания ДМ кодов с ювелирных изделий с помощью USB-микроскопа',
    epilog = '''Примеры использования: \n
    ascortDmScanner // открыть программу с параметрами по умолчанию и показывать на экране режим сканирования \n
    ascortDmScanner --camID=1 --clipboard=True // Использовать камеру с номером 1 и скопировать в буфер обмена, как только будет найден УИН \n
    ascortDmScanner --readFromFile='qr.jpg' --resultFile='result.txt'  // прочитать файл и сохранить УИН в файл result.txt'''
)

parser.add_argument(
    "--resultFile", "-r",
    help="Файл результата. Если не передаваь, сохранение не будет произведено",
    type = str,
    default = ''
)

parser.add_argument(
    "--camID", "-c",
    help="Идентификатор usb микроскопа или камеры",
    dest="camID",
    type = int,
    default = 0
)

parser.add_argument(
    "--accuracy",
    "-a", help="Точность чтения в миллисекундах",
    type = int,
    default = 70
)

parser.add_argument(
    "--clipboard", "-C",
    help="Записать результат в буфер обмена",
    type = bool,
    default = False
)

parser.add_argument(
    "--readFromFile", "-f",
    help="Имя файла для чтения и распознавания, в случае установки данного параметра, подключение камеры не производится",
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

while True:
    if readFromFile != '':
        image = cv2.imread(readFromFile)
    else:
        rez, image = cap.read()
        if not rez:
            print('не удалось подключить веб-камеру')
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

    k = cv2.waitKey(33)
    if k == -1:
        continue
    else:
        break

cv2.destroyWindow(windowName)
cap.release()
sys.exit(0)