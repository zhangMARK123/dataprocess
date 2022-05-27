import json
import pandas as pd
import numpy as np
import os
import cv2
####传为pandas参数，分别统计各类别的precious和recall

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
    print(len(labels["objects"]))
    count=0
    for obj in labels["objects"]:
        filename=os.path.join(imgpath,obj["imgname"])   
        showflag=False

        if obj["label_color_gt"] in ['0','1','2','4'] and obj["label_shape_gt"]=='7':
            obj["lightboxshape_head"]="False"
        if obj["label_color_gt"] in ['0','1','2','4'] and obj["label_shape_gt"]=='7' and obj["label_toward_gt"]=='1':
            obj["lightboxcolor_head"]="False"
        if obj["label_color_gt"]=='4':
            obj["lightboxcolor_head"]=False
        if obj["label_character_gt"]=='1' and obj["label_toward_gt"]=='1':
            obj["character_head"]="False"
        # if obj["label_color_gt"] in ['0','1','2'] and obj["label_shape_gt"]=='7' and obj["label_toward_gt"]=='0':
        #     obj["lightboxshape_head"]="False"
        # if obj["label_color_gt"] in ['0','1','2'] and obj["label_shape_gt"]=='7' and obj["label_toward_gt"]=='1':
        #     obj["lightboxcolor_head"]="False"
        # if obj["label_character_gt"]=='1' and obj["label_toward_gt"]=='1':
        #     obj["character_head"]="False"
        # print("===zs====")
        # print(obj["lightboxcolor_head"]==str(True))
        # print(obj["lightboxcolor_head"])
        # if obj["lightboxcolor_head"]==str(True) and obj["score_color_pred"]<0.5:
        #     showflag=True
        # if obj["lightboxcolor_head"]==str(True) and obj["label_color_pred"]!=obj["label_color_gt"]:
        #     showflag=True
        # if obj["lightboxshape_head"]==str(True) and obj["score_shape_pred"]<0.5:
        #     showflag=True
        if obj["lightboxshape_head"]==str(True) and obj["label_shape_pred"]!=obj["label_shape_gt"]:
            showflag=True
        # if obj["toward_head"] and obj["label_toward_pred"]!=obj["label_toward_gt"] and obj["label_toward_gt"] in [0,1]:
        #     showflag=True
        # if obj["simplelight_head"] and obj["label_simplelight_pred"] != obj["label_simplelight_gt"] and obj["label_simplelight_gt"]==0:
        #     showflag=True
        # if obj["character_head"]==str(True) and  obj["label_character_pred"]!=obj["label_character_gt"]:
        #     showflag=True
        # if obj["lightboxshape_head"]==str(True) and obj["label_shape_gt"]=="7":
        #     showflag=True
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
        colorlabel_pred=COLOR_CLASSES[int(obj["label_color_pred"])]
        colorlabel_gt=COLOR_CLASSES[int(obj["label_color_gt"])]
        shapelabel_pred=SHAPE_CLASSES[int(obj["label_shape_pred"])]
        shapelabel_gt=SHAPE_CLASSES[int(obj["label_shape_gt"])]
        characterlabel_pred=CHARACTER_CLASSES[int(obj["label_character_pred"])]
        characterlabel_gt=CHARACTER_CLASSES[int(obj["label_character_gt"])]
        towardlabel_pred=TOWARD_CLASSES[int(obj["label_toward_pred"])]
        towardlabel_gt=TOWARD_CLASSES[int(obj["label_toward_gt"])]
        simplelightlabel_pred=SIMPLE_CLASSES[int(obj["label_simplelight_pred"])]
        simplelightlabel_gt=SIMPLE_CLASSES[int(obj["label_simplelight_gt"])]

        # if obj["simplelight_head"]:
        #      cv2.putText(img, simplelightlabel_pred, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        #     #  cv2.putText(img, simplelightlabel_gt, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        # if obj["lightboxcolor_head"]:
        #      cv2.putText(img, colorlabel_pred, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        #      cv2.putText(img, colorlabel_gt, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        if obj["lightboxshape_head"]:
             cv2.putText(img, shapelabel_pred, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
             cv2.putText(img, shapelabel_gt, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        # if obj["toward_head"]:
        #      cv2.putText(img, towardlabel_pred, (15, 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        #      cv2.putText(img, towardlabel_gt, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        # if obj["character_head"]:
        #    cv2.putText(img, characterlabel_pred, (15, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        #    cv2.putText(img, characterlabel_gt, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
        # datacard=obj["imgname"].split('/')[0]
        # if not os.path.exists(os.path.join(wpath,datacard,"images")):
        #     os.makedirs(os.path.join(wpath,datacard,"images"))   
        if not os.path.exists(wpath):
            os.makedirs(wpath)   
        
        cv2.imwrite(os.path.join(wpath,obj["imgname"].split("/")[-1]),img)
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
    colorpered=[]
    colorget=[]
    colorhead=[]
    shapepred=[]
    shapegt=[]
    shapehead=[]
    characterpred=[]
    charactergt=[]
    characterhead=[]
    towardpred=[]
    towardgt=[]
    towardhead=[]
    simplepred=[]
    simplegt=[]
    simplehead=[]
    name=[]
    for result_output in labels["objects"]:
        if result_output["label_color_gt"] in ['0','1','2','4'] and result_output["label_shape_gt"]=='7':
            result_output["lightboxshape_head"]="False"
        if result_output["label_color_gt"] in ['0','1','2','4'] and result_output["label_shape_gt"]=='7' and result_output["label_toward_gt"]=='1':
            result_output["lightboxcolor_head"]="False"
        if result_output["label_color_gt"]=='4':
            result_output["lightboxcolor_head"]=False
        if result_output["label_character_gt"]=='1' and result_output["label_toward_gt"]=='1':
            result_output["character_head"]="False"
        
        colorpered.append(result_output["label_color_pred"])
        colorget.append(result_output["label_color_gt"]   )
        colorhead.append(result_output['lightboxcolor_head'])
        shapepred.append(result_output["label_shape_pred"])
        shapegt.append(result_output["label_shape_gt"] )
        shapehead.append(result_output['lightboxshape_head'])
        characterpred.append(result_output["label_character_pred"] )
        charactergt.append(result_output["label_character_gt"])
        characterhead.append(result_output['character_head'])
        towardpred.append(result_output["label_toward_pred"])
        towardgt.append(result_output["label_toward_gt"])
        towardhead.append( result_output['toward_head'])
        simplepred.append(result_output["label_simplelight_pred"])
        simplegt.append(result_output["label_simplelight_gt"])
        simplehead.append(result_output['simplelight_head'])
        name.append(result_output["imgname"])
        
    df=pd.DataFrame({"label_color_pred":colorpered,"label_color_gt":colorget,'lightboxcolor_head':colorhead,
    "label_shape_pred":shapepred,"label_shape_gt":shapegt,'lightboxshape_head':shapehead,
    "label_character_pred":characterpred,"label_character_gt":charactergt,'character_head':characterhead,"label_toward_pred":towardpred,
    "label_toward_gt":towardgt,'toward_head':towardhead,"label_simplelight_pred":simplepred,"label_simplelight_gt":simplegt,'simplelight_head':simplehead,"imgname":name})
    ####颜色
    color_tp=((df['lightboxcolor_head']=='True' ) & (df["label_color_gt"]==df["label_color_pred"])).sum()

    color_all=(df['lightboxcolor_head']=='True').sum() 
    color_precious=color_tp/color_all

    print("#############color############################")
    print("color precious is ",color_precious)
    for i in range(len(COLOR_CLASSES)):
        tp=((df['lightboxcolor_head']=='True') & (df["label_color_gt"]==str(i)) & (df["label_color_pred"]==str(i))).sum()
        full=((df['lightboxcolor_head']=='True') & (df["label_color_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", COLOR_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(full)
        print(COLOR_CLASSES[i]," pricious is :",precious)  
    ###形状
    shape_tp=((df['lightboxshape_head']=='True' ) & (df["label_shape_gt"]==df["label_shape_pred"])).sum()
    shape_all=(df['lightboxshape_head']=='True').sum()
    shape_precious=shape_tp/shape_all
    
    print("#############shape############################")
    print(shape_all)
    print("shape precious is ",shape_precious)
    
    
    for i in range(len(SHAPE_CLASSES)):
        tp=((df['lightboxshape_head']=='True') & (df["label_shape_gt"]==str(i)) & (df["label_shape_pred"]==str(i))).sum()
        full=((df['lightboxshape_head']=='True') & (df["label_shape_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", SHAPE_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(full)
        print(SHAPE_CLASSES[i]," pricious is :",precious)   
    
    ###类型
    character_tp=((df['character_head']=='True' ) & (df["label_character_gt"]==df["label_character_pred"])).sum()
    character_all=(df['character_head']=='True').sum()
    character_precious=character_tp/character_all

    print("#############character############################")
    # print(character_all)
    print("character precious is ",character_precious)
    for i in range(len(CHARACTER_CLASSES)):
        tp=((df['character_head']=='True') & (df["label_character_gt"]==str(i)) & (df["label_character_pred"]==str(i))).sum()
        full=((df['character_head']=='True') & (df["label_character_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", CHARACTER_CLASSES[i],"is 0")
            continue
        precious=tp/full
        # print(full)
        print(CHARACTER_CLASSES[i]," pricious is :",precious)  

    ###朝向
    toward_tp=((df['toward_head']=='True' ) & (df["label_toward_gt"]==df["label_toward_pred"])).sum()
    toward_all=(df['toward_head']=='True').sum()
    toward_precious=toward_tp/toward_all
    print("#############toward############################")
    # print(toward_all)
    print("toward precious is ",toward_precious)

    for i in range(len(TOWARD_CLASSES)):
        tp=((df['toward_head']=='True') & (df["label_toward_gt"]==str(i)) & (df["label_toward_pred"]==str(i))).sum()
        full=((df['toward_head']=='True') & (df["label_toward_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", TOWARD_CLASSES[i],"is 0")
            continue
        precious=tp/full
        # print(full)
        print(TOWARD_CLASSES[i]," pricious is :",precious)   
    ###简单复杂
    simple_tp=((df['simplelight_head']=='True' ) & (df["label_simplelight_gt"]==df["label_simplelight_pred"])).sum()
    simple_all=(df['simplelight_head']=='True').sum()
    simple_precious=simple_tp/simple_all
    print("#############simpleligth############################")
    # print(simple_all)
    print("simplelight precious is ",simple_precious)

    for i in range(len(SIMPLE_CLASSES)):
        tp=((df['simplelight_head']=='True') & (df["label_simplelight_gt"]==str(i)) & (df["label_simplelight_pred"]==str(i))).sum()
        full=((df['simplelight_head']=='True') & (df["label_simplelight_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", SIMPLE_CLASSES[i],"is 0")
            continue
        precious=tp/full
        # print(full)
        print(SIMPLE_CLASSES[i]," pricious is :",precious) 

    #  ####绿色黄色箭头的准确率
    # colorshape_tp=((df['lightboxshape_head']=='True')&(df['lightboxcolor_head']=='True') &(df["label_shape_gt"]!='0')&(df["label_shape_gt"]!='6')&(df["label_shape_gt"]!='7')&((df["label_color_gt"]=='1')|(df["label_color_gt"]=='2'))
    # & (df["label_color_gt"]==df["label_color_pred"])&(df["label_shape_gt"]==df["label_shape_pred"])).sum()
    # colorshape_all=colorshape_tp=((df['lightboxshape_head']=='True')&(df['lightboxcolor_head']=='True') &(df["label_shape_gt"]!='0')&(df["label_shape_gt"]!='6')&(df["label_shape_gt"]!='7')&((df["label_color_gt"]=='1')|(df["label_color_gt"]=='2'))
    # ).sum()
    # print(colorshape_tp)
    # print(colorshape_all)
    # colorshape_precious=colorshape_tp/colorshape_all

    # print("colorshape precious is ",colorshape_precious)


# labelpath="/disk3/zs1/mmclassification/work_dirs/complexlight_img/little2_data_complex/12_result.json"
labelpath="/disk3/zs1/mmclassification/work_dirs/complexlight_img/addharddata/1hardcase_result.json"
wpath="../work_dirs/test_moni/1_result_eval.json"
evalresult(labelpath,wpath)


# imgpath="/disk3/zbh/Datasets/2022_Q1_icu30_test_crop/"
# wimg="../work_dirs/complexlight_img/ALL/testshape"

# plotcrop(imgpath,labelpath,wimg)