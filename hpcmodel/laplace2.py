import cv2
import sys
import numpy as np
import os

#laplace
kernel_sharpen = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]])
#图像锐化
kernel_sharpen1 = np.array([
    [0, -1, 0],
    [0, 5, 0],
    [0, -1, 0]])
#beisoer
kernel_sharpen2 = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]])

kernel_sharpen3 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])

kernel_sharpen4 = np.array([
    [0, -1, 0],
    [0, 5, 0],
    [0, -1, 0]])

kernel_sharpen5 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])

kernel_sharpen6 = np.array([
    [1, 1, 1],
    [1, -7, 1],
    [1, 1, 1]])

kernel_sharpen7 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [-1, -1, -1, -1, -1]])

# dataroot=r"C:\Users\zxpan\Desktop\daily_report\data\badcase2"
# wpath=r"C:\Users\zxpan\Desktop\daily_report\data\badcase2labeled"


img_path=os.path.join(r"C:\Users\zxpan\Desktop\test\ori4\1640567376916708_4.jpg")
roi = cv2.imread(img_path, cv2.IMREAD_COLOR)
blur = cv2.bilateralFilter(roi, 5, 10, 10)
result1 = cv2.filter2D(blur, -1, kernel_sharpen1)
result2 = cv2.filter2D(blur, -1, kernel_sharpen2)
result3 = cv2.filter2D(blur, -1, kernel_sharpen3)
result4 = cv2.filter2D(blur, -1, kernel_sharpen4)
result5 = cv2.filter2D(blur, -1, kernel_sharpen5)
result6 = cv2.filter2D(blur, -1, kernel_sharpen6)
result7 = cv2.filter2D(blur, -1, kernel_sharpen7)
result = cv2.filter2D(blur, -1, kernel_sharpen)
# cv2.imwrite(os.path.join(wpath1,"result1_"+path),result1)
# cv2.imwrite(os.path.join(wpath1,"result2_"+path),result2)
# cv2.imwrite(os.path.join(wpath1,"result3_"+path),result3)
# cv2.imwrite(os.path.join(wpath1,"result4_"+path),result4)
# cv2.imwrite(os.path.join(wpath1,"result5_"+path),result5)
# cv2.imwrite(os.path.join(wpath1,"result6_"+path),result6)
# cv2.imwrite(os.path.join(wpath1,"result7_"+path),result7)
# cv2.imwrite(os.path.join(wpath1,"result_"+path),result)


cv2.imshow("original", roi)
cv2.imshow("blur", blur)
cv2.imshow("result", result)
cv2.imshow("result1", result1)
cv2.imshow("result2", result2)
cv2.imshow("result3", result3)
cv2.imshow("result4", result4)
cv2.imshow("result5", result5)
cv2.imshow("result6", result6)
cv2.imshow("result7", result7)

cv2.waitKey(0)
cv2.destroyAllWindows()
