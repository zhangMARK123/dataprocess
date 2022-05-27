import json 
import os

addpath="/disk3/zs1/mmclassification/work_dirs/complexlight_img/addharddata/1hardcase_result.json"
labelpath1="/disk3/zbh/Datasets/2022_QA_icu30_train_complexlight0523_adddata.json"
labelpath2="/disk3/zbh/Datasets/2022_Q1_icu30_train_complexlight0523.json"
wpath="/disk3/zbh/Datasets/2022_QA_icu30_train_onlyhardcase.json"
labels1=json.loads(open(labelpath1,'r').read())
labels2=json.loads(open(labelpath2,'r').read())
addlabels=json.loads(open(addpath,'r').read())
addobj={}
# for obj in addlabels["objects"]:
#     if obj["lightboxshape_head"]=="True" and obj["label_toward_gt"]=="0":
#         if obj["label_shape_pred"]!=obj["label_shape_gt"] and float(obj["score_shape_pred"])<0.8:
#             if obj["label_shape_gt"]=="0" and obj["label_color_gt"]=="0":
#                 continue
#             if obj["imgname"].split("/")[0] not in addobj:
#                 addobj[obj["imgname"].split("/")[0]]=[obj["imgname"].split("/")[-1]]
#             else:
#                 addobj[obj["imgname"].split("/")[0]].append(obj["imgname"].split("/")[-1])
#         elif obj["label_toward_gt"]=="0" and obj["label_shape_gt"] in ["1","2","3","4","5","6"] and obj["label_shape_pred"]==obj["label_shape_gt"] and float(obj["score_shape_pred"])<0.6:
#             if obj["imgname"].split("/")[0] not in addobj:
#                 addobj[obj["imgname"].split("/")[0]]=[obj["imgname"].split("/")[-1]]
#             else:
#                 addobj[obj["imgname"].split("/")[0]].append(obj["imgname"].split("/")[-1])

# res=[]
# for obj in labels1["objects"]:
#     if obj["data_card_id"] in addobj and obj["img_info"]["filename"].split("/")[-1] in addobj[obj["data_card_id"]]:
#         res.append(obj)
# # res=res+labels2["objects"]
# with open(wpath,'w+') as f:
#          f.write(json.dumps({"objects": res}, ensure_ascii=False,
#                            indent=4))

for obj in addlabels["objects"]:
    if obj["lightboxshape_head"]=="True":
        if obj["label_shape_pred"]!=obj["label_shape_gt"] and float(obj["score_shape_pred"])<0.8:
            if obj["imgname"].split("/")[0] not in addobj:
                addobj[obj["imgname"].split("/")[0]]=[obj["imgname"].split("/")[-1]]
            else:
                addobj[obj["imgname"].split("/")[0]].append(obj["imgname"].split("/")[-1])
        elif obj["label_shape_pred"]==obj["label_shape_gt"] and float(obj["score_shape_pred"])<0.6:
            if obj["imgname"].split("/")[0] not in addobj:
                addobj[obj["imgname"].split("/")[0]]=[obj["imgname"].split("/")[-1]]
            else:
                addobj[obj["imgname"].split("/")[0]].append(obj["imgname"].split("/")[-1])

res=[]
for obj in labels1["objects"]:
    if obj["data_card_id"] in addobj and obj["img_info"]["filename"].split("/")[-1] in addobj[obj["data_card_id"]]:
        res.append(obj)
for obj in labels2["objects"]:
    if obj["data_card_id"] in addobj and obj["img_info"]["filename"].split("/")[-1] in addobj[obj["data_card_id"]]:
        res.append(obj)
with open(wpath,'w+') as f:
         f.write(json.dumps({"objects": res}, ensure_ascii=False,
                           indent=4))


