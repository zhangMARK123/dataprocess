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

dataroot=r"C:\Users\zxpan\Desktop\daily_report\data\badcase2"
wpath=r"C:\Users\zxpan\Desktop\result7"
for datacard_id in os.listdir(dataroot):
    imgdir=os.path.join(dataroot,datacard_id,"images")
    wpath1=os.path.join(wpath,datacard_id,"images")
    if not os.path.exists(wpath1):
        os.makedirs(wpath1)
    for path in os.listdir(imgdir):
        img_path=os.path.join(imgdir,path)
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
        cv2.imwrite(os.path.join(wpath1,"result7_"+path),result7)
        # cv2.imwrite(os.path.join(wpath1,"result_"+path),result)


# cv2.imshow("original Image", roi)
# cv2.imshow("blur Image", blur)
# cv2.imshow("result Image", result)
# cv2.imshow("result Image1", result1)
# cv2.imshow("result Image2", result2)
# cv2.imshow("result Image3", result3)
# cv2.imshow("result Image4", result4)
# cv2.imshow("result Image5", result5)
# cv2.imshow("result Image6", result6)
# cv2.imshow("result Image7", result7)

# cv2.waitKey(120000)
# cv2.destroyAllWindows()
