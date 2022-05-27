import json
import pandas as pd
import numpy as np
import os
import cv2
####传为pandas参数，分别统计各类别的precious和recall

def plotcrop(imgpath,labelpath,wpath):
   
    COLOR_CLASSES = [
        'red',
        'yellow',
        'green',
        'black',
        'others',
        'unknow'
    ]
    labels=json.loads(open(labelpath,'r').read())
    # print(len(labels["objects"]))
    count1=0
    count=0
    for obj in labels["objects"]:
        filename=os.path.join(imgpath,obj["imgname"])   
        showflag=False
       
        # if obj["sublightcolor_head"]=="True" and obj["label_numsublight_gt"]=="0":
        #         obj["sublightcolor_head"]="False"    
     
        if obj["sublightcolor_head"].strip()=="False":
            continue  
        count1+=1    
        if obj["sublightcolor_head"]=="True" and obj["label_subcolor1_pred"]!=obj["label_subcolor1_gt"]:
            showflag=True
        if obj["sublightcolor_head"]=="True" and obj["label_subcolor2_pred"]!=obj["label_subcolor2_gt"]:
            showflag=True

        if obj["sublightcolor_head"]=="True" and obj["label_subcolor3_pred"]!=obj["label_subcolor3_gt"]:
            showflag=True

        if obj["sublightcolor_head"]=="True" and obj["label_subcolor4_pred"]!=obj["label_subcolor4_gt"]:
            showflag=True

        if obj["sublightcolor_head"]=="True" and obj["label_subcolor5_pred"]!=obj["label_subcolor5_gt"]:
            showflag=True     
        if showflag==False:
            continue
        img=cv2.imread(filename)  
        count+=1    
        color=(26,26,139)
        bbox=obj["bbox"].split(",")
        
        x1=int(float(bbox[0][1:-1]))
        y1=int(float(bbox[1]))
        x2=int(float(bbox[0][1:-1])+float(bbox[2]))
        y2=int(float(bbox[1])+float(bbox[3][:-1]))   
        # color=(26,139,26)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
        color1label_pred=COLOR_CLASSES[int(obj["label_subcolor1_pred"])]
        color1label_gt=COLOR_CLASSES[int(obj["label_subcolor1_gt"])]
        color2label_pred=COLOR_CLASSES[int(obj["label_subcolor2_pred"])]
        color2label_gt=COLOR_CLASSES[int(obj["label_subcolor2_gt"])]
        color3label_pred=COLOR_CLASSES[int(obj["label_subcolor3_pred"])]
        color3label_gt=COLOR_CLASSES[int(obj["label_subcolor3_gt"])]
        color4label_pred=COLOR_CLASSES[int(obj["label_subcolor4_pred"])]
        color4label_gt=COLOR_CLASSES[int(obj["label_subcolor4_gt"])]
        color5label_pred=COLOR_CLASSES[int(obj["label_subcolor5_pred"])]
        color5label_gt=COLOR_CLASSES[int(obj["label_subcolor5_gt"])]
       
        if obj["sublightcolor_head"]==str(True):
             cv2.putText(img, color1label_pred, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
            #  cv2.putText(img, color1label_gt, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["sublightcolor_head"]==str(True):
             cv2.putText(img, color2label_pred, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
            #  cv2.putText(img, color2label_gt, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["sublightcolor_head"]==str(True):
             cv2.putText(img, color3label_pred, (15, 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
            #  cv2.putText(img, color3label_gt, (15, 90), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["sublightcolor_head"]==str(True):
             cv2.putText(img, color4label_pred, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
            #  cv2.putText(img, color4label_gt, (15, 1), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["sublightcolor_head"]==str(True):
             cv2.putText(img, color5label_pred, (60, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
            #  cv2.putText(img, color5label_gt, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        datacard=obj["imgname"].split('/')[0]
        if not os.path.exists(os.path.join(wpath,datacard,"images")):
            os.makedirs(os.path.join(wpath,datacard,"images"))   
        
        cv2.imwrite(os.path.join(wpath,obj["imgname"]),img)
    print(count1)
    print(count)

def evalresult(labelpath,wpath):

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
    subcolor1pred=[]
    subcolor1get=[]
    subcolorhead=[]
    subcolor2pred=[]
    subcolor2get=[]
    subcolor3pred=[]
    subcolor3get=[]
    subcolor4pred=[]
    subcolor4get=[]
    subcolor5pred=[]
    subcolor5get=[]
    name=[]
    for result_output in labels["objects"]:
        
        
        # if result_output["sublightcolor_head"]=="True" and result_output["label_numsublight_gt"]=="0":
        #         result_output["sublightcolor_head"]="False"     
        if result_output["sublightcolor_head"]=="False":
            continue  
        subcolor1pred.append(result_output["label_subcolor1_pred"])
        subcolor1get.append(result_output["label_subcolor1_gt"])
        subcolorhead.append(result_output["sublightcolor_head"])
        subcolor2pred.append(result_output["label_subcolor2_pred"])
        subcolor2get.append(result_output["label_subcolor2_gt"])        
        subcolor3pred.append(result_output["label_subcolor3_pred"])
        subcolor3get.append(result_output["label_subcolor3_gt"])       
        subcolor4pred.append(result_output["label_subcolor4_pred"])
        subcolor4get.append(result_output["label_subcolor4_gt"])
        subcolor5pred.append(result_output["label_subcolor5_pred"])
        subcolor5get.append(result_output["label_subcolor5_gt"])    
        name.append(result_output["imgname"])
        
    df=pd.DataFrame({"label_subcolor1_pred":subcolor1pred,"label_subcolor1_gt":subcolor1get,"sublightcolor_head":subcolorhead,
    "label_subcolor2_pred":subcolor2pred,"label_subcolor2_gt":subcolor2get,"label_subcolor3_pred":subcolor3pred,"label_subcolor3_gt":subcolor3get,
    "label_subcolor4_pred":subcolor4pred,"label_subcolor4_gt":subcolor4get,"label_subcolor5_pred":subcolor5pred,"label_subcolor5_gt":subcolor5get,
    "imgname":name})
    
   
   ####颜色
    color_tp=((df["sublightcolor_head"]=='True' ) & (df["label_subcolor1_pred"]==df["label_subcolor1_gt"])).sum()
    color_all=(df["sublightcolor_head"]=='True').sum() 
    color_precious=color_tp/color_all
    
    print("#############color############################")
    print("color precious is ",color_precious)
    for i in range(len(COLOR_CLASSES)):
        tp=((df["sublightcolor_head"]=='True') & (df["label_subcolor1_pred"]==str(i)) & (df["label_subcolor1_gt"]==str(i))).sum()
        full=((df["sublightcolor_head"]=='True') & (df["label_subcolor1_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", COLOR_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(COLOR_CLASSES[i]," pricious is :",precious)  

    ####颜色
    color_tp=((df["sublightcolor_head"]=='True' ) & (df["label_subcolor2_pred"]==df["label_subcolor2_gt"])).sum()
    color_all=(df["sublightcolor_head"]=='True').sum() 
    color_precious=color_tp/color_all
    
    print("#############color############################")
    print("color precious is ",color_precious)
    for i in range(len(COLOR_CLASSES)):
        tp=((df["sublightcolor_head"]=='True') & (df["label_subcolor2_pred"]==str(i)) & (df["label_subcolor2_gt"]==str(i))).sum()
        full=((df["sublightcolor_head"]=='True') & (df["label_subcolor2_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", COLOR_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(COLOR_CLASSES[i]," pricious is :",precious)  

    ####颜色
    color_tp=((df["sublightcolor_head"]=='True' ) & (df["label_subcolor3_pred"]==df["label_subcolor3_gt"])).sum()
    color_all=(df["sublightcolor_head"]=='True').sum() 
    color_precious=color_tp/color_all
    
    print("#############color############################")
    print("color precious is ",color_precious)
    for i in range(len(COLOR_CLASSES)):
        tp=((df["sublightcolor_head"]=='True') & (df["label_subcolor3_pred"]==str(i)) & (df["label_subcolor3_gt"]==str(i))).sum()
        full=((df["sublightcolor_head"]=='True') & (df["label_subcolor3_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", COLOR_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(COLOR_CLASSES[i]," pricious is :",precious)  
        
    ####颜色
    color_tp=((df["sublightcolor_head"]=='True' ) & (df["label_subcolor4_pred"]==df["label_subcolor4_gt"])).sum()
    color_all=(df["sublightcolor_head"]=='True').sum() 
    color_precious=color_tp/color_all
    
    print("#############color############################")
    print("color precious is ",color_precious)
    for i in range(len(COLOR_CLASSES)):
        tp=((df["sublightcolor_head"]=='True') & (df["label_subcolor4_pred"]==str(i)) & (df["label_subcolor4_gt"]==str(i))).sum()
        full=((df["sublightcolor_head"]=='True') & (df["label_subcolor4_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", COLOR_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(COLOR_CLASSES[i]," pricious is :",precious)  

    ####颜色
    color_tp=((df["sublightcolor_head"]=='True' ) & (df["label_subcolor5_pred"]==df["label_subcolor5_gt"])).sum()
    color_all=(df["sublightcolor_head"]=='True').sum() 
    color_precious=color_tp/color_all
    
    print("#############color############################")
    print("color precious is ",color_precious)
    for i in range(len(COLOR_CLASSES)):
        tp=((df["sublightcolor_head"]=='True') & (df["label_subcolor5_pred"]==str(i)) & (df["label_subcolor5_gt"]==str(i))).sum()
        full=((df["sublightcolor_head"]=='True') & (df["label_subcolor5_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", COLOR_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(COLOR_CLASSES[i]," pricious is :",precious)  
    

    


labelpath="../../work_dirs/complexlight_img/little_data_complex/12_result.json"
wpath="../../work_dirs/complexlight_img/sublightcolorwith3/12_result_eval.json"
evalresult(labelpath,wpath)


# imgpath="/disk3/zbh/Datasets/2022_Q1_icu30_test_crop/"
# wimg="../../work_dirs/complexlight_img/sublightcolorwith3/testcolor"

# plotcrop(imgpath,labelpath,wimg)