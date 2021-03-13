import numpy as np
import cv2

# number plate detection
######################################################
frameWidth = 640
frameHeight = 480
# numbPlate = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
numbPlate = cv2.CascadeClassifier("resources/haarcascade_russian_plate_number.xml")
minArea = 500
color = (255, 0, 255)
count = 4
#######################################################

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    numberPlate = numbPlate.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in numberPlate:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)
    cv2.imshow("result", img)

    keyboard_k = cv2.waitKey(1)
    if keyboard_k & 0xff == ord('s'):
        cv2.imwrite("resources/scanned/NoPlate_" + str(count) + ".jpg", imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scanned Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("result", img)
        cv2.waitKey(500)
        count += 1
    if keyboard_k & 0xff == ord('q'):
        break

# img = cv2.imread("./resources/p3.jpg")
# cv2.resize(img, (frameWidth, frameHeight))
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# numberPlate = numbPlate.detectMultiScale(imgGray, 1.1, 4)
# for (x, y, w, h) in numberPlate:
#     area = w * h
#     if area > minArea:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
#         imgRoi = img[y:y + h, x:x + w]
#         cv2.imshow("ROI", imgRoi)
# cv2.imshow("result", img)
# if cv2.waitKey(0) & 0xff == ord("s"):
#     cv2.imwrite("resources/scanned/Noplate_3.jpg", imgRoi)
#     cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
#     cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
#     cv2.imshow("Result", img)
#     cv2.waitKey(500)
