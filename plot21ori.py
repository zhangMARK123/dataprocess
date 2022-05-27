import json
import os
import cv2
imgpath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30\61d928aa910d7200e75be08e\images"
labelpath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30\61d928aa910d7200e75be08e\labels"
wpath=r"C:\Users\zxpan\Desktop\daily_report\data\2022_Q1_icu30\61d928aa910d7200e75be08e\wpath"
SHAPE_CLASSES = [
        'gothrough',
        'pedestrian',
        'number',
        'character',
        'others',
        'unknown',
    ]

COLOR_CLASSES = [
        'red',
        'yellow',
        'green',
        'black',
        'others',
        'unknown',
    ]
TOWARD_CLASSES = [
        'fornt',
        'side',
        'reverse',
        'unknown',
    ]
SUBLIGHTSHAPE_CLASSES=[
    'circle',
    'bicycle',
    'singlearrow',
    'composearrow',
    'number',
    'pedestrian',
    'others',
    'unknown',
]
ARROW_ORIENTATION_CLASSES=[
    'up_arrow',
    'down_arrow',
    'left_arrow',
    'right_arrow',
    'u_turn',
]
for path in os.listdir(imgpath):
    img=cv2.imread(os.path.join(imgpath,path))
    if not os.path.exists(os.path.join(labelpath,path.split(".")[0]+".json")):
        continue
    label=json.loads(open(os.path.join(labelpath,path.split(".")[0]+".json"),'r').read())
    for obj in label["objects"]:
        bbox=obj["bbox"]
        x1=int(bbox[0])
        y1=int(bbox[1])
        x2=int(bbox[0]+bbox[2])
        y2=int(bbox[1]+bbox[3])     
        max_l=3*max(bbox[2],bbox[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),2)
        
        xx1=int(max(0,bbox[0]-(max_l-bbox[2])/2))
        xx2=int(min(1920,bbox[0]+(max_l+bbox[2])/2))
        yy1=int(max(0,bbox[1]-(max_l-bbox[3])/2))
        yy2=int(min(1080,bbox[1]+(max_l+bbox[3])/2))
        cv2.rectangle(img,(xx1,yy1),(xx2,yy2),(0,0,255),2)
        cv2.putText(img, SHAPE_CLASSES[obj['characteristic']], (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
        if obj["num_sub_lights"]==0:
            continue
        for i in obj["splits"]:
            ptStart=(int(i[0]),int(i[1]))
            ptEnd=(int(i[2]),int(i[3]))
            cv2.line(img, ptStart, ptEnd, (255,0,0), 2, 4)
        if obj["num_sub_lights"]!=len(obj["splits"])+1:
            print(obj["num_sub_lights"],"splits",len(obj["splits"])+1)
            print(path)
            continue
        for i in range(obj["num_sub_lights"]):          
            sublight= obj["sub_lights"][i]         
            color_label=COLOR_CLASSES[sublight["color"]]
            shape_label=SUBLIGHTSHAPE_CLASSES[sublight['shape']]
            
            if i==0:
                cv2.putText(img, color_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
                cv2.putText(img, shape_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
                if sublight["arrow_orientation"]:
                    for j in sublight["arrow_orientation"]:
                        if sublight["arrow_orientation"][j]!=0:                
                            cv2.putText(img, j, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
            elif i==1:
                x1=int(obj["splits"][0][0])
                y1=int(obj["splits"][0][1])
                cv2.putText(img, color_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
                cv2.putText(img, shape_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
                if sublight["arrow_orientation"]:
                    for j in sublight["arrow_orientation"]:
                        if sublight["arrow_orientation"][j]!=0:                
                            cv2.putText(img, j, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
               
                    
            elif i==2:
                x1=int(obj["splits"][1][0])
                y1=int(obj["splits"][1][1])
                cv2.putText(img, color_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
                cv2.putText(img, shape_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0))
                if sublight["arrow_orientation"]:
                    for j in sublight["arrow_orientation"]:
                        if sublight["arrow_orientation"][j]!=0:                
                            cv2.putText(img, j, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0,0,0))
                
            elif i==3:
                x1=int(obj["splits"][2][0])
                y1=int(obj["splits"][2][1])
                cv2.putText(img, color_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0,0,0))
                cv2.putText(img, shape_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0,0,0))
                if sublight["arrow_orientation"]:
                    for j in sublight["arrow_orientation"]:
                        if sublight["arrow_orientation"][j]!=0:                
                            cv2.putText(img, j, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0,0,0))
                
            elif i==4:
                x1=int(obj["splits"][3][0])
                y1=int(obj["splits"][3][1])
                cv2.putText(img, color_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0,0,0))
                cv2.putText(img, shape_label, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0,0,0))
                if sublight["arrow_orientation"]:
                    for j in sublight["arrow_orientation"]:
                        if sublight["arrow_orientation"][j]!=0:                
                            cv2.putText(img, j, (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.25, (0,0,0))
                
    cv2.imwrite(os.path.join(wpath,path),img)


        
