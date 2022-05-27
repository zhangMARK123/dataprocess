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
    SIMPLE_CLASSES=['simple',
    'complex']
    labels=json.loads(open(labelpath,'r').read())
    for obj in labels["objects"]:
        filename=os.path.join(imgpath,obj["data_card_id"],obj["img_info"]["filename"])   
        
        if obj["ext_occlusion"]==1 or obj["truncation"]==1:
            continue
        if obj["toward_orientation"]!=0:
            continue   
        img=cv2.imread(filename)  
        bbox=obj["bbox"]
        x1=int(bbox[0])
        y1=int(bbox[1])
        x2=int(bbox[0]+bbox[2])
        y2=int(bbox[1]+bbox[3])       
        color=(0,123,200)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        colorlabel=COLOR_CLASSES[obj["boxcolor"]]
        shapelabel=SHAPE_CLASSES[obj["boxshape"]]
        
        
        if obj["lightboxcolor_head"]:
             cv2.putText(img, colorlabel, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["lightboxshape_head"]:
             cv2.putText(img, shapelabel, (15, 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        # if obj["toward_head"]:
        #      cv2.putText(img, towardlabel, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
       

        
        if not os.path.exists(os.path.join(wpath,obj["data_card_id"],"images")):
            os.makedirs(os.path.join(wpath,obj["data_card_id"],"images")) 
        
        if min(bbox[2],bbox[3])>10:
            cv2.imwrite(os.path.join(wpath,obj["data_card_id"],obj["img_info"]["filename"]),img)
    

imgpath="/disk3/zbh/Datasets/2022_Q1_icu30_crop/"
labelpath="/disk3/zbh/Datasets/2022_Q1_icu30_moni.json"
wpath="/disk3/zs1/Datasets/plotdata/"

plotcrop(imgpath,labelpath,wpath)