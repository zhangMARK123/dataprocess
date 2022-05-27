import json
import os
import cv2
import shutil


labelpath="/disk3/zbh/Datasets/2022_Q1_icu30_train0316.json"

labels=json.loads(open(labelpath,'r').read())

all=[]
name=[]
count0=0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
for obj in labels["objects"]:
    # if obj["pose_orientation"]!=0:
    #     continue
    if obj["truncation"]==1 or obj["ext_occlusion"]==1:
        continue
    if not obj["simplelight_head"]:
        continue
    if obj["simplelight"]==0:
        continue
    if obj["num_sub_lights"]!=5:
        continue
    arrow=""
    numsub=0
    for sub in obj["sub_lights"]:
        subshape="不知"
        if sub["shape"]==2:
            numsub+=1
            if sub["arrow_orientation"]["up_arrow"]==1:
                subshape="上"
            elif sub["arrow_orientation"]["down_arrow"]==1:
                subshape="下"
            elif sub["arrow_orientation"]["left_arrow"]==1:
                subshape="左"
            elif sub["arrow_orientation"]["right_arrow"]==1:
                subshape="右"
            elif sub["arrow_orientation"]["u_turn"]==1:
                subshape="掉头"
            arrow+=str(subshape)
    if numsub!=5:
        continue
    cat=['左上上上上', '左上上上右', '左左上上右', '上上上上右', '左左上上上', '左上上右掉头', '左左上上左']
    # if arrow not in all:
    #     all.append(arrow)
    #     name.append(obj["data_card_id"]+"/"+obj["img_info"]["filename"])
    if arrow==cat[0]:
        count0+=1
    elif arrow==cat[1]:
        count1+=1
    elif arrow==cat[2]:
        count2+=1
    elif arrow==cat[3]:
        count3+=1
    elif arrow==cat[4]:
        count4+=1
    elif arrow==cat[5]:
        count5+=1
    elif arrow==cat[6]:
        count6+=1
print(all)
print(count0, count1, count2, count3, count4, count5, count6)
print(name)

#labelpath='/disk3/zbh/Datasets/2022_Q1_icu30_crop'
#name=['61f3c73f761998edc27300be/images/1642038437746932_0.jpg', '620a0077d3572380d57d7b1e/images/1640920481016591_0.jpg', '6207d426d3572380d596ea29/images/1642057164388122_1.jpg', '6219df488ec597d64c599cb9/images/1639441420316732_0.jpg', '61e5171c910d7200e711bf5c/images/1640918956216489_0.jpg', '6207d426d3572380d596ea29/images/1642057157936296_3.jpg', '6207d426d3572380d596ea29/images/1642052725896383_2.jpg', '6219df488ec597d64c599cb9/images/1639460340389712_2.jpg', '6219df488ec597d64c599cb9/images/1639448095516812_0.jpg', '6207c1bbd3572380d596ab26/images/1642043983005759_1.jpg', '620cab537504aaa4086e6ede/images/1642217453431341_5.jpg', '6219df488ec597d64c599cb9/images/1639462596303324_1.jpg', '6207d426d3572380d596ea29/images/1642052079628656_0.jpg', '61f200eb761998edc22ab096/images/1640929184016637_5.jpg']
# name=['61f3c73f761998edc27300be/images/1642038437746932_0.jpg', '620a0077d3572380d57d7b1e/images/1640920481016591_0.jpg', '6207d426d3572380d596ea29/images/1642057164388122_1.jpg', '6219df488ec597d64c599cb9/images/1639441420316732_0.jpg', '61e5171c910d7200e711bf5c/images/1640918956216489_0.jpg', '6207d426d3572380d596ea29/images/1642057157936296_3.jpg', '6219df488ec597d64c599cb9/images/1639460340389712_2.jpg', '6219df488ec597d64c599cb9/images/1639448095516812_0.jpg', '6219df488ec597d64c599cb9/images/1639462596303324_1.jpg', '6207d426d3572380d596ea29/images/1642052079628656_0.jpg', '6219df488ec597d64c599cb9/images/1639460604059038_0.jpg']
# count=0
# for label in name:
#     imgname=os.path.join(labelpath,label)
    
#     shutil.copy(imgname,'/disk3/zbh/Datasets/'+str(count)+".jpg")
#     count+=1




