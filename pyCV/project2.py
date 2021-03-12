import cv2
import numpy as np
from chapter6 import stackImages

# Document Sacnner

############################
widthImg = 480
heightImg = 640
cv2.namedWindow("WorkFlow", 0)

#############################

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 150)


def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel=kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    return imgThreshold


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgContours, cnt, -1, (0, 0, 255), 3)
            peri = cv2.arcLength(cnt, True)

            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContours, biggest, -1, (0, 250, 0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print("newPoints", myPointsNew)
    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


# img = cv2.imread("./resources/paper.jpg")
# cv2.resize(img, (widthImg, heightImg))
# imgContours = img.copy()
#
# imgThres = preProcessing(img)
# # getContours(imgThres)
# biggest = getContours(imgThres)
# # print(biggest)
# if biggest.size != 0:
#     imgWarped = getWarp(img, biggest)
#     imageArray = ([img, imgThres], [imgContours, imgWarped])
# else:
#     imageArray = ([img, imgThres], [img, img])
#
# stackImages = stackImages(0.6, imageArray)
#
# # cv2.imshow("result", imgContours)
# cv2.imshow("result", stackImages)
# cv2.imshow("ImageWarped", imgWarped)
# cv2.waitKey(0)


while True:
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContours = img.copy()

    imgThres = preProcessing(img)
    # getContours(imgThres)
    biggest = getContours(imgThres)
    # print(biggest)
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        imageArray = ([img, imgThres],
                      [imgContours, imgWarped])
        # imageArray = ([imgContours, imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        # imageArray = ([imgContours, img])
        imageArray = ([img, imgThres],
                      [img, img])

    stackedImages = stackImages(0.6, imageArray)
    #
    # cv2.imshow("result", imgContours)
    cv2.imshow("WorkFlow", stackedImages)

    if cv2.waitKey(1) & 0xff == ord("q"):
        break
