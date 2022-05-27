import json 
import os

rootdir=r"C:\Users\zxpan\Desktop\daily_report\data\test0324"
wdir=r"C:\Users\zxpan\Desktop\daily_report\data\test0324res"
for data_card_id in os.listdir(rootdir):
    labelpath=os.path.join(rootdir,data_card_id,"labels")
    wpath=os.path.join(wdir,data_card_id,"labels")
    if not os.path.exists(wpath):
        os.makedirs(wpath)
    for label in os.listdir(labelpath):
        gt_label=json.loads(open(os.path.join(labelpath,label),'r').read())
        for obj in gt_label["objects"]:
            shape=obj["boxshape"]
            if shape==0:
                obj["boxshape"]=0
            elif shape==6:
                obj["boxshape"]=1
            elif shape in [1,2,3,4,5]:
                if shape==1:
                    obj["arrow_orientation"]={
            "up_arrow": 1,
            "down_arrow": 0,
            "left_arrow": 0,
            "right_arrow": 0,
            "u_turn": 0
          }
                elif shape==2:
                   obj["arrow_orientation"]={
            "up_arrow": 0,
            "down_arrow": 1,
            "left_arrow": 0,
            "right_arrow": 0,
            "u_turn": 0
          }
                elif shape==3:
                    obj["arrow_orientation"]={
            "up_arrow": 0,
            "down_arrow": 0,
            "left_arrow":1,
            "right_arrow": 0,
            "u_turn": 0
          }
                elif shape==4:
                    obj["arrow_orientation"]={
            "up_arrow": 0,
            "down_arrow": 0,
            "left_arrow": 0,
            "right_arrow":1,
            "u_turn": 0
          }
                elif shape==5:
                    obj["arrow_orientation"]={
            "up_arrow": 0,
            "down_arrow": 0,
            "left_arrow": 0,
            "right_arrow": 0,
            "u_turn": 1
          }
                obj["boxshape"]=2
            else:
                obj["boxshape"]=7
        wlabelpath=os.path.join(wpath,label)
        with open(wlabelpath, 'w+') as f:
            f.write(json.dumps({"objects":gt_label["objects"]},ensure_ascii=False, indent=4))

