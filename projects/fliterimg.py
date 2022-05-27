from cgitb import small
import os
import json
import shutil
import cv2
data_root="/disk3/zbh/Datasets/2022_Q1_icu30_test_crop/"
savepath="/disk3/zs1/Datasets/2022_Q1_icu30_crop_labeled2/comlpex/num/0"
labelpath='/disk3/zbh/Datasets/2022_Q1_icu30_testsublight0326.json'

SIMPE_LIGHT = ["simple","complex"]
COLOR = ["red","yellow","green", "black","others","unknown"]
SHAPE=["circle","up arrow","down arrow","left arrow","right arrow","return arrow","bicycle","others"]                       
TWOARD = ["front", "side","backside","unknown"]
CHARACTER=["pass","president","number","word","others","unknow"]

with open(labelpath) as f:
    sample_json = json.load(f)

count_small=0
count_middle=0
count_large=0
count_large1=0
count_large2=0
max_length=0
num=[]
count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
for obj in sample_json["objects"]:
    data_card=obj["data_card_id"]
    
    imgname=obj["img_info"]["filename"].split("/")[-1]
    if not os.path.exists(os.path.join(savepath,data_card)):
        os.makedirs(os.path.join(savepath,data_card))
    if obj["ext_occlusion"] == 1 or obj["truncation"] == 1:
                continue 
    if min(obj["bbox"][2], obj["bbox"][3])<=0:
                continue
    if max(obj["bbox"][2], obj["bbox"][3]) / min(obj["bbox"][2], obj["bbox"][3]) < 2.5:
        continue 
    if obj["bbox"][2] <10 or obj["bbox"][3] <10:
        continue

    ####
    # if obj["numsublight_head"]==False:
    #     continue
    # if obj["numcolorlight"]!=0:
    #     continue
    ##night light
    # if obj['toward_orientation']==2:
    #     obj["toward_head"]=False
    # if obj['toward_orientation']!=0:
    #     obj["lightboxshape_head"]=False
    #     obj["lightboxcolor_head"]=False

    # if obj["boxshape"] in [4,5]:
    #     obj["lightboxshape_head"]=False
    #     # obj["lightboxcolor_head"]=False
    # if obj["boxcolor"] in [3,4]:   
    #     obj["lightboxcolor_head"]=False


    # if obj["lightboxcolor_head"]==False:
    #     continue
    # if obj["toward_orientation"]!=0:
    #     continue

    # if obj["boxshape"]!=5:
    #     continue
    # count=obj["boxcolor"]
    # if count==0:
    #     count0+=1
    # elif count==1:
    #     count1+=1
    # elif count==2:
    #     count2+=1
    # elif count==3:
    #     count3+=1
    # elif count==4:
    #     count4+=1
    # elif count==5:
    #     count5+=1


    # if count!=3:
    #     continue

    bbox=obj["bbox"]

    
    ###统计各类别数量
    # length=min(bbox[2],bbox[3])
    # max_length=max(max_length,length)
    # if 90>length>80:
    #     print(obj["img_info"])
    # elif 15<=length<20:
    #     count_middle+=1
    # elif 20<=length<=25:
    #     count_large+=1
    # elif 26<length<33:
    #     count_large1+=1
    # elif length>=140:
    #     count_large2+=1


    

    x1=int(bbox[0])-2
    y1=int(bbox[1])-2
    x2=int(bbox[0]+bbox[2])+2
    y2=int(bbox[1]+bbox[3])+2
    color=(255,0,255)
    oriimg=os.path.join(data_root,data_card,obj["img_info"]["filename"])
    img=cv2.imread(oriimg)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)
    cv2.putText(img, str(obj["boxshape"]), (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
    movedpath=os.path.join(savepath,data_card,imgname)
    cv2.imwrite(movedpath,img)
  

# print(count_small)
# print(count_middle)
# print(count_large)
# print(count_large1)
# print(count_large2)
# print(num)

# print(count0)
# print(count1)
# print(count2)
# print(count3)
# print(count4)
# print(count5)