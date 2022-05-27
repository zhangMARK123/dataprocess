import cv2
import os
import numpy as np
import random
imgpath=r"C:\Users\zxpan\Desktop\daily_report\data\processimg\ori\images"
wpath=r"C:\Users\zxpan\Desktop\daily_report\data\processimg\process1"
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])


for path in os.listdir(imgpath):
    img=cv2.imread(os.path.join(imgpath,path))
    blue,green,red=cv2.split(img)
    # ret2, th2 = cv2.threshold(red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # sharpened_img = cv2.filter2D(red, -1, kernel_sharpening)
    # red=cv2.medianBlur(red,3)
    # res=cv2.Canny(red)
    b = random.sample([blue,green,red], 3) 
    img2=cv2.merge((b[0],b[1],b[2]))
    # img2=cv2.merge((blue,red,green))
    # h,w,c=img.shape
    # # center=(w//2,h//2)
    # # M=cv2.getRotationMatrix2D(center, 90,1.0)
    # # img= cv2.warpAffine(img, M, (w,h)) 
    # img=cv2.transpose(img)
    # img=cv2.flip(img,1)
    cv2.imwrite(os.path.join(wpath,path),img2)
