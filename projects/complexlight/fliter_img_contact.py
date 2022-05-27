import json 
import os
import cv2
import shutil
data_root="/disk3/zbh/Datasets/2022_Q1_icu30_crop/"
labelpath="/disk3/zs1/mmclassification/projects/complexlight/littledata_color.json"
savepath="/disk3/zs1/Datasets/traindata_badcase/littledata/wrongdata/"
with open(labelpath) as f:
    sample_json = json.load(f)

for obj in sample_json["objects"]:
    data_card=obj["data_card_id"]
    imgname=obj["img_info"]["filename"]
    srcpath=os.path.join(data_root,data_card,imgname)
    img=cv2.imread(srcpath)  
    bbox=obj["bbox"]
    x1=int(bbox[0])
    y1=int(bbox[1])
    x2=int(bbox[0]+bbox[2])
    y2=int(bbox[1]+bbox[3])       
    color=(0,123,200)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
    cv2.putText(img, str(obj["boxcolor"]), (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    dstpath=os.path.join(savepath,imgname.split('/')[-1])
    # shutil.copy(srcpath,dstpath)
    cv2.imwrite(dstpath,img)