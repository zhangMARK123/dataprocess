import json
import pandas as pd
import numpy as np
import os
####传为pandas参数，分别统计各类别的precious和recall

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
        print(COLOR_CLASSES[i]," pricious is :",precious)  
    ###形状
    shape_tp=((df['lightboxshape_head']=='True' ) & (df["label_shape_gt"]==df["label_shape_pred"])).sum()
    shape_all=(df['lightboxshape_head']=='True').sum()
    shape_precious=shape_tp/shape_all
    
    print("#############shape############################")
    print("shape precious is ",shape_precious)
    
    for i in range(len(SHAPE_CLASSES)):
        tp=((df['lightboxshape_head']=='True') & (df["label_shape_gt"]==str(i)) & (df["label_shape_pred"]==str(i))).sum()
        full=((df['lightboxshape_head']=='True') & (df["label_shape_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", SHAPE_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(SHAPE_CLASSES[i]," pricious is :",precious)   
    
    ###类型
    character_tp=((df['character_head']=='True' ) & (df["label_character_gt"]==df["label_character_pred"])).sum()
    character_all=(df['character_head']=='True').sum()
    character_precious=character_tp/character_all

    print("#############character############################")
    print("character precious is ",character_precious)
    for i in range(len(CHARACTER_CLASSES)):
        tp=((df['character_head']=='True') & (df["label_character_gt"]==str(i)) & (df["label_character_pred"]==str(i))).sum()
        full=((df['character_head']=='True') & (df["label_character_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", CHARACTER_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(CHARACTER_CLASSES[i]," pricious is :",precious)  

    ###朝向
    toward_tp=((df['toward_head']=='True' ) & (df["label_toward_gt"]==df["label_toward_pred"])).sum()
    toward_all=(df['toward_head']=='True').sum()
    toward_precious=toward_tp/toward_all
    print("#############toward############################")
    print("toward precious is ",toward_precious)

    for i in range(len(TOWARD_CLASSES)):
        tp=((df['toward_head']=='True') & (df["label_toward_gt"]==str(i)) & (df["label_toward_pred"]==str(i))).sum()
        full=((df['toward_head']=='True') & (df["label_toward_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", TOWARD_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(TOWARD_CLASSES[i]," pricious is :",precious)   
    ###简单复杂
    simple_tp=((df['simplelight_head']=='True' ) & (df["label_simplelight_gt"]==df["label_simplelight_pred"])).sum()
    simple_all=(df['simplelight_head']=='True').sum()
    simple_precious=simple_tp/simple_all
    print("#############simpleligth############################")
    print("simplelight precious is ",simple_precious)

    for i in range(len(SIMPLE_CLASSES)):
        tp=((df['simplelight_head']=='True') & (df["label_simplelight_gt"]==str(i)) & (df["label_simplelight_pred"]==str(i))).sum()
        full=((df['simplelight_head']=='True') & (df["label_simplelight_gt"]==str(i))).sum()
        if full==0:
            print("the number of ", SIMPLE_CLASSES[i],"is 0")
            continue
        precious=tp/full
        print(SIMPLE_CLASSES[i]," pricious is :",precious) 


labelpath="../result.json"
wpath="evalresult.json"

evalresult(labelpath,wpath)
