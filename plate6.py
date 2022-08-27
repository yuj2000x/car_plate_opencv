# locate car plate
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

import authmecv as acv

# car_plate : (561, 640, 3)
# 讀取圖片 縮放至固定大小
img = cv2.imread('./cars/car8.jpeg')
img = imutils.resize(img, 500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 雙邊模糊去噪點 sobel邊緣偵測 Otsu二值化
blur = cv2.bilateralFilter(gray, 11, 17, 17)  # 雙邊
blur = cv2.GaussianBlur(blur, (1, 3), 0)  # 高斯 先寬再高
sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=1)
ret, thr = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 閉運算 開運算 canny偵測
kernel1 = np.ones((5, 31), np.uint8)  # kernel 先高再寬
closeimg = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel1)
openimg1 = cv2.morphologyEx(closeimg, cv2.MORPH_OPEN, kernel1)

kernel2 = np.ones((15, 3), np.uint8)
openimg2 = cv2.morphologyEx(openimg1, cv2.MORPH_CLOSE, kernel2)

kernel4 = np.ones((25, 25), np.uint8)
openimg3 = cv2.morphologyEx(openimg2, cv2.MORPH_OPEN, kernel4)

kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 先寬再高
dilated = cv2.dilate(openimg3, kernel3)

# 圖片 2 edged = dilate
# 圖片 3 4 edged = openimg2
edged = cv2.Canny(dilated, 170, 200)

# 提取輪廓，找出最長周長的輪廓
cnts, new = cv2.findContours(
    edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img_cnts = img.copy()
cv2.drawContours(img_cnts, cnts, -1, (0, 0, 255), 2)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
PlateCount = None

img_cnts_sort = img.copy()
cv2.drawContours(img_cnts_sort, cnts, -1, (0, 0, 255), 2)

# ----------------------------------------------------
# # 將輪廓規整為長方形
# rectangles = []
# for c in cnts:
#     x = []
#     y = []
#     for point in c:
#         y.append(point[0][0])
#         x.append(point[0][1])
#     r = [min(y), min(x), max(y), max(x)]
#     rectangles.append(r)
# rectangles = rectangles[0]
# # 把兩個角轉成四個
# corner = []
# x = rectangles[2] - rectangles[0]
# y = rectangles[3] - rectangles[1]
# for num in range(4):
#     i = int(num / 2)
#     j = num % 2
#     corner.append(rectangles[0] + x*i)
#     corner.append(rectangles[1] + y*j)
# corner = np.array(corner)
# corner = corner.reshape((4, 2))
# print(corner)
# ----------------------------------------------------

# 角座標提取
rect = cv2.minAreaRect(cnts[0])
box = cv2.boxPoints(rect)
box = box.astype(int)

imgcnt = img.copy()
for i in range(4):
    cv2.circle(imgcnt, (box[i, 0], box[i, 1]), 4, (0, 255, 0), -1)
cv2.drawContours(imgcnt, [box], -1, (0, 0, 255), 2)

# 裁切圖片
x = []
y = []
for i in range(len(box)):
    x.append(box[i][0])
    y.append(box[i][1])
corner = [min(y), max(y), min(x), max(x)]

img_crop = img.copy()
crop = img_crop[corner[0]: corner[1], corner[2]: corner[3]]

# 顯示過程處理的圖
res = np.hstack((closeimg, openimg1, openimg2, openimg3, dilated))

cv2.imshow("img", crop)
cv2.imshow("car8_cnt", imgcnt)
# cv2.imshow("car8_crop", res)
key = cv2.waitKey()
if key == ord("q"):
    print("exit")
cv2.destroyWindow("windows")
