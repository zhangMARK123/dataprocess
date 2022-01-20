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
        
        img=cv2.imread(filename)  
        bbox=obj["bbox"]
        x1=int(bbox[0])
        y1=int(bbox[1])
        x2=int(bbox[0]+bbox[2])
        y2=int(bbox[1]+bbox[3])       
        color=(255,255,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        colorlabel=COLOR_CLASSES[obj["boxcolor"]]
        shapelabel=SHAPE_CLASSES[obj["boxshape"]]
        characterlabel=CHARACTER_CLASSES[obj["characteristic"]]
        towardlabel=TOWARD_CLASSES[obj["toward_orientation"]]
        simplelightlabel=SIMPLE_CLASSES[obj["simplelight"]]

        if obj["simplelight_head"]:
             cv2.putText(img, simplelightlabel, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["lightboxcolor_head"]:
             cv2.putText(img, colorlabel, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["lightboxshape_head"]:
             cv2.putText(img, shapelabel, (15, 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["toward_head"]:
             cv2.putText(img, towardlabel, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["character_head"]:
           cv2.putText(img, characterlabel, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if not os.path.exists(os.path.join(wpath,obj["data_card_id"],"images")):
            os.makedirs(os.path.join(wpath,obj["data_card_id"],"images"))   
        # if obj["toward_orientation"]==0 or obj["toward_orientation"]==1:
        if obj["simplelight_head"] and obj["lightboxcolor_head"] and obj["lightboxshape_head"] and  obj["character_head"]:
            continue
        if min(bbox[2],bbox[3])<10:
            cv2.imwrite(os.path.join(wpath,obj["data_card_id"],obj["img_info"]["filename"]),img)
    

imgpath="/share/zbh/Datasets/2022_Q1_icu30_new_crop2_correct/"
labelpath="/share/zbh/Datasets/2022_Q1_icu30_new_crop_correct4_zs_test.json"
wpath="/share/zbh/Datasets/wpath/simplelight"

plotcrop(imgpath,labelpath,wpath)