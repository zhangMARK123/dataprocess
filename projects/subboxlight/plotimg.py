import cv2 
import numpy as np
import os
import json


def plotcrop(imgpath,labelpath,wpath):
    SHAPE_CLASSES = [
        'circle',
        'uparrow',
        'downarrow',
        'leftarrow',
        'rightarrow',
        'returnarrow',
        'bicycle',   
        'others',
    ]

    COLOR_CLASSES = [
        'red',
        'yellow',
        'green',
        'black',
        'others',
        'unknow'
    
    ]

    TOWARD_CLASSES = [
        'front',
        'side',
        'backside',
        'unknow'
    ]
    CHARACTER_CLASSES=[
        'pass',
        'president',
        'number',
        'word',
        'others',
        'unknow'
    ]
    labels=json.loads(open(labelpath,'r').read())
    for obj in labels["objects"]:
        filename=os.path.join(imgpath,obj["data_card_id"],obj["img_info"]["filename"])   
        
        if obj["ext_occlusion"]==1 or obj["truncation"]==1:
            continue
        if obj["toward"]!=0:
            continue
        if obj["bbox"][2]<0 or obj["bbox"][3]<0:
            continue
        if obj["shape_head"]==False and obj["color_head"]==False:
            continue
        imgname=os.path.join(imgpath,obj["data_card_id"],obj["img_info"]["filename"])
       
        if not os.path.exists(imgname):
            continue
        # print(imgname)
        img=cv2.imread(imgname)  
        # if not img:
        #     continue
        bbox=obj["bbox"]
        bbox1=obj["allbbox"]
        x1=int(bbox1[0])
        y1=int(bbox1[1])
        x2=int(bbox1[0]+bbox1[2])
        y2=int(bbox1[1]+bbox1[3]) 
        x3=int(bbox[0])
        y3=int(bbox[1])
        x4=int(bbox[0]+bbox[2])
        y4=int(bbox[1]+bbox[3])       
        color=(0,0,200)
        color1=(200,0,0)
        # cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        cv2.rectangle(img,(x3,y3),(x4,y4),color1,1)
        colorlabel=COLOR_CLASSES[obj["color"]]
        shapelabel=SHAPE_CLASSES[obj["shape"]]
        characterlabel=CHARACTER_CLASSES[obj["characteristic"]]
        towardlabel=TOWARD_CLASSES[obj["toward"]]

        if obj["color_head"]:
             cv2.putText(img, colorlabel, (5, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["shape_head"]:
             cv2.putText(img, shapelabel, (5, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        # if obj["toward_head"]:
        #      cv2.putText(img, towardlabel, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        # if obj["character_head"]:
        #    cv2.putText(img, characterlabel, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if not os.path.exists(os.path.join(wpath,obj["data_card_id"],"images")):
            os.makedirs(os.path.join(wpath,obj["data_card_id"],"images"))   
        # if obj["toward_orientation"]==0 or obj["toward_orientation"]==1:
        # if obj["simplelight_head"] and obj["lightboxcolor_head"] and obj["lightboxshape_head"] and  obj["character_head"]:
        #     continue
        
        if min(bbox1[2],bbox1[3])>10:
            cv2.imwrite(os.path.join(wpath,obj["data_card_id"],obj["img_info"]["filename"]),img)
    

imgpath="/disk3/zbh/Datasets/2022_Q1_icu30_crop/"
labelpath="/disk3/zbh/Datasets/2022_Q1_icu30_train_splitbox.json"
wpath="/disk3/zs1/Datasets/plotdata/2022_Q1_icu30_splitbox/"

plotcrop(imgpath,labelpath,wpath)