import cv2
import os
import numpy as np

imgpath=r"C:\Users\zxpan\Desktop\daily_report\data\badcase2\select\images"
wpath=r"C:\Users\zxpan\Desktop\daily_report\data\imageprocess\ori"


kernel_sharpen4 = np.array([
    [0, -1, 0],
    [0, 5, 0],
    [0, -1, 0]])


for path in os.listdir(imgpath):
    img=cv2.imread(os.path.join(imgpath,path))
    ###转换成HSV颜色空间进行分割，自定义阈值下，效果较差
    # hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    # (thresh, im_bw) = cv2.threshold(hsv[:,:,0], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # blur = cv2.bilateralFilter(gray, 5, 10, 10)
    # result1 = cv2.filter2D(blur, -1, kernel_sharpen4)
    
    # ###进行Gabor滤波
    # # retval = cv2.getGaborKernel(ksize=(20,20), sigma=5, theta=45, lambd=10, gamma=1.2)
    # # result = cv2.filter2D(img,-1,retval)

    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) 
    # ori=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    wimg=os.path.join(wpath,path)

    cv2.imwrite(wimg,hsv)

