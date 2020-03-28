import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "1.jpg"
number_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
number = number_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(10, 10))

for (x, y, w, h) in number:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    crop = image[y:y+h, x:x+w]
    print(pytesseract.image_to_string(crop))
    cv2.imshow("cropped", crop)
viewImage(image,"Number detected")

