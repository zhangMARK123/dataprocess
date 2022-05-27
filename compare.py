import os
import json
import cv2
oripath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30_crop"
clearpath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30_crop_labeled2"
wname=r"C:\Users\zxpan\Desktop\daily_report\data\names.txt"
labelpath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30_arrow.json"
labels=json.loads(open(labelpath,'r').read())
exe=[]
for obj in labels["objects"]:
    filename=os.path.join(oripath,obj["data_card_id"],obj["img_info"]["filename"])  
    if not os.path.exists(filename):
        continue
    # img=cv2.imread(filename)  
    # if img is None:
    #     continue
    if obj["ext_occlusion"]==1 or obj["truncation"]==1:
        exe.append(os.path.join(obj["data_card_id"],"images",obj["img_info"]["filename"].split("/")[-1]))
noname=[]
def fun(x):
    return x[2:]
for data_card_id in os.listdir(oripath):
    allname=os.listdir(os.path.join(clearpath,data_card_id,"images"))
    allname=list(map(fun,allname)) 
    for idx in os.listdir(os.path.join(oripath, data_card_id, "images")):      
        if idx not in allname:
            noname.append(os.path.join(data_card_id,"images",idx))
f=open(wname,"w")
print(len(noname))
print(len(exe))
count=0
for i in noname:
    if i not in exe:
        count+=1
        f.write(i)
        f.write("\n")
f.close()
print(count)