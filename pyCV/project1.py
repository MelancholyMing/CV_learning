import numpy as np
import cv2


# Virtual Paint
frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

myColors = [[164, 148, 136, 179, 255, 255],
            [96, 42, 90, 105, 255, 255]]

myColorValues = [[51, 153, 255],
                 [255, 0, 255]]  # BGR

myPoints = []  # [x,y,colorID]


def findColor(img, myColor, myColorValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints_ = []
    for color in myColor:
        lower = np.array(color[:3])
        upper = np.array(color[3:6])
        # cv2.inRange函数设阈值，去除背景部分
        mask = cv2.inRange(imgHSV, lower, upper)
        # getContours(mask)
        x, y = getContours(mask)
        print('=============')
        cv2.circle(imgResult, (x, y), 10, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints_.append([x, y, count])
        count += 1
        # cv2.imshow(str(color[0]), mask)
    return newPoints_


def getContours(img):
    # https://blog.csdn.net/hjxu2016/article/details/77833336
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)  # 计算轮廓的面积
        if area > 500:
            # 画出图片中的轮廓值，也可以用来画轮廓的近似值 参数说明:img 表示输入的需要画的图片， contours 表示轮廓值，-1 表示轮廓的索引, 如果是 - 1，则绘制其中的所有轮廓，(0, 0, 255) 表示颜色， 2 表示线条粗细
            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)  # 计算轮廓的周长
            # 用于获得轮廓的近似值，把一个连续光滑曲线折线化.使用 cv2.drawCountors 进行画图操作.cnt 为输入的轮廓值， epsilon 为阈值 T，通常使用轮廓的周长作为阈值，True 表示的是轮廓是闭合的
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(approx)
            objCor = len(approx)
            # 获得外接矩形. x，y, w, h 分别表示外接矩形的 x 轴和 y 轴的坐标，以及矩形的宽和高， cnt 表示输入的轮廓值
            x, y, w, h = cv2.boundingRect(approx)

    return x + w // 2, y


def drawOncanvas(points, ColorValues):
    for point in points:
        cv2.circle(imgResult, (point[0], point[1]), 10, ColorValues[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    # findColor(img, myColors, myColorValues)
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints) != 0:
        drawOncanvas(myPoints, myColorValues)

    cv2.imshow("result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
