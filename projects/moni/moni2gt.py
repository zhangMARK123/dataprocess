import json
import os
import cv2

spath="/disk3/zbh/Datasets/moni/crop/crop/"
dpath="/disk3/zbh/Datasets/2022_Q1_icu30_crop/"
labelpath="/disk3/zs1/mmclassification/projects/moni/1645429690016553.json"

for l in os.listdir(spath):
    if l in ['CN_TL_YellowSolid_H',"CN_TL_YellowArrowLeft_H","CN_TL_YellowArrowStraight_H","CN_TL_YellowArrowRight_H","CN_TL_YellowUTurn_H"]:
            continue
    if not os.path.exists(os.path.join(dpath,l,'labels')):
        os.makedirs(os.path.join(dpath,l,'labels'))
        os.makedirs(os.path.join(dpath,l,'images'))
        
    for path in os.listdir(os.path.join(spath,l)):
        img=cv2.imread(os.path.join(spath,l,path))
        h=img.shape[0]
        w=img.shape[1]
        img=cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(128,128,128))
        labels=json.loads(open(labelpath,'r').read())
        labels["objects"][0]["bbox"]=[200,200,w,h]
        labels['width']=img.shape[1]
        labels['height']=img.shape[0]
        if l in ['CN_TL_YellowSolid_H','CN_TL_YellowSolid_H']:
            labels["objects"][0]["sub_lights"][1]["shape"]=0
        elif l in["CN_TL_YellowArrowLeft","CN_TL_YellowArrowLeft_H"]:
            labels["objects"][0]["sub_lights"][1]["shape"]=2
            labels["objects"][0]["sub_lights"][1]["arrow_orientation"]["left_arrow"]=1
        elif l in ["CN_TL_YellowArrowStraight","CN_TL_YellowArrowStraight_H"] :
            labels["objects"][0]["sub_lights"][1]["shape"]=2
            labels["objects"][0]["sub_lights"][1]["arrow_orientation"]["up_arrow"]=1
        elif l in ["CN_TL_YellowArrowRight","CN_TL_YellowArrowRight_H"]:
            labels["objects"][0]["sub_lights"][1]["shape"]=2
            labels["objects"][0]["sub_lights"][1]["arrow_orientation"]["right_arrow"]=1
        elif l in ["CN_TL_YellowUTurn","CN_TL_YellowUTurn_H"]:
            labels["objects"][0]["sub_lights"][1]["shape"]=2
            labels["objects"][0]["sub_lights"][1]["arrow_orientation"]["u_turn"]=1
        # cv2.imwrite(os.path.join(dpath,l,'images',path[:-3]+"jpg"),img)
        with open(os.path.join(dpath,l,'labels',path[:-3]+"json"),'w+') as f:
            f.write(json.dumps(labels, ensure_ascii=False, indent=4))


    