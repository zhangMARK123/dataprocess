from genericpath import exists
import os
import json
import shutil
import cv2
data_root=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30_crop_labeled\color\greenside"
savepath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30_crop_labeled\color\greensidecat\withshape"
labelpath=r'C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30_train_review_color_0303.json'

SIMPE_LIGHT = ["simple","complex"]
COLOR = ["red","yellow","green", "black","others","unknown"]
SHAPE=["circle","up arrow","down arrow","left arrow","right arrow","return arrow","bicycle","others"]                       
TWOARD = ["front", "side","backside","unknown"]
CHARACTER=["pass","president","number","word","others","unknow"]

with open(labelpath) as f:
    sample_json = json.load(f)
for obj in sample_json["objects"]:
    data_card=obj["data_card_id"]
    imgname=obj["img_info"]["filename"].split("/")[-1]
    if not os.path.exists(os.path.join(savepath,data_card)):
        os.makedirs(os.path.join(savepath,data_card))
    oriimg=os.path.join(data_root,data_card,imgname)
   
    if not exists(oriimg):
        continue
    
    # if obj["ext_occlusion"] == 1 or obj["truncation"] == 1:
    #             continue 
    # if max(obj["bbox"][2], obj["bbox"][3]) / min(obj["bbox"][2], obj["bbox"][3]) < 2.5:
    #     continue 
    # if obj["bbox"][2] <10 or obj["bbox"][3] <10:
    #     continue
    # if obj["lightboxcolor_head"]==False:
    #     continue
    # if obj["boxshape"]!=7 or obj["toward_orientation"]!=0 or obj["boxcolor"] not in [0,1,2,4]:
    #     continue
    if obj["boxshape"] ==7:
        continue
    # unknow=0
    # for sub in obj["sub_lights"]:
    #     if sub["color"]==5:
    #         unknow+=1
    # if unknow>=1:
    #     continue

    
    bbox=obj["bbox"]
    x1=int(bbox[0])-2
    y1=int(bbox[1])-2
    x2=int(bbox[0]+bbox[2])+2
    y2=int(bbox[1]+bbox[3])+2
    color=(255,0,255)
    # oriimg=os.path.join(data_root,data_card,obj["img_info"]["filename"])
    img=cv2.imread(oriimg)

    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
    cv2.putText(img, COLOR[obj["boxcolor"]], (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
    movedpath=os.path.join(savepath,data_card,imgname)
    cv2.imwrite(movedpath,img)
    # shutil.copy(oriimg,movedpath) 