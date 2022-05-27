import json
import os
import cv2
imgpath=r"C:\Users\zxpan\Desktop\daily_report\data\ICU30_crop"
labelpath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30_train_review.json"
wpath=r"C:\Users\zxpan\Desktop\daily_report\data\ICU30_crop_labeled\color\others\front"

SIMPE_LIGHT = ["simple","complex"]
COLOR = ["red","yellow","green", "black","others","unknown"]
SHAPE=["circle","up arrow","down arrow","left arrow","right arrow","return arrow","bicycle","others"]                       
TWOARD = ["front", "side","backside","unknown"]
CHARACTER=["pass","president","number","word","others","unknow"]

labels=json.loads(open(labelpath,'r').read())
for obj in labels["objects"]:
    filename=os.path.join(imgpath,obj["data_card_id"],obj["img_info"]["filename"])  
    if not os.path.exists(filename):
        continue
    img=cv2.imread(filename)  
    if img is None:
        continue
    bbox=obj["bbox"]
    x1=int(bbox[0])-2
    y1=int(bbox[1])-2
    x2=int(bbox[0]+bbox[2])+2
    y2=int(bbox[1]+bbox[3])+2
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
    
    if obj["ext_occlusion"]==1 or obj["truncation"]==1:
        continue
    
    if obj["simplelight"]==1:
        continue
    if bbox[2]<10 or bbox[3]<10:
        continue
    
    if obj["lightboxcolor_head"]==False:
        continue
    
    if obj["boxcolor"] not in [4]:
        continue
    color=(255,0,255)
    print("True1")
    cv2.putText(img, COLOR[obj["boxcolor"]], (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
    if not os.path.exists(os.path.join(wpath,obj["data_card_id"],"images")):
        os.makedirs(os.path.join(wpath,obj["data_card_id"],"images"))
    # shape=obj["characteristic"]
    temppath="images/"+obj["img_info"]["filename"].split("/")[-1]
    # print(temppath)
    cv2.imwrite(os.path.join(wpath,obj["data_card_id"],temppath),img)


        
