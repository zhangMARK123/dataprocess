import os
import json
import shutil

spath="/disk3/zbh/Datasets/2022_Q1_icu30_test_crop/"
# spath="/disk5/zhangshuo/Datasets/Q1train_data/"
namelist=["6231aa8b625d0d750434f5ca",
"623454b1e53b5981783de76b",
"623455b1969d98de67bb444b",
"62345e50e53b59817847f749",
"623c1b53e53b59817862445f",
"623c1d12e53b59817863f614",
"623c1e29e53b59817864a1b5",
"623d2eefe53b5981784c7057",
"623d2f16e53b5981784cb683",
"623d3821e53b5981786b05ce",
"623d383be53b5981786b7f0d"
]
count=0

for i in os.listdir(spath):
    if i in namelist:
        count=count+1
        # namelist.remove(i)
        print(i)
print(count)

# namelist2=[]
# for i in os.listdir(spath):
#     namelist2.append(i)
# print(namelist2)



# namelist="hld_card_0330.txt"
# wpath="namelist2.txt"
# srcname=[]

# wlist=open(wpath,"r+")
# name=[]
# f=open(namelist).readlines()
# for l in f:
#     if l=="\n":
#         continue
#     name.append(l.strip())

# for i in name:
#     if not os.path.exists(os.path.join(spath,i)):
#         print(i)
#         continue
#     for j in os.listdir(os.path.join(spath,i,"labels")):        
#         srcname.append(os.path.join(i,"labels",j))
# count=0
# for l in name:  
#     for i in os.listdir("/disk3/zbh/Datasets/nalelist/traffic_light/front_middle_camera"):
#         labelname=os.path.join(l,"labels",i.split("_")[-1])
#         if labelname in srcname:
#             wlist.write(labelname)
#             wlist.write("\n")
#             count+=1
# print(count)
        

# namelist1=open(wpath).readlines()
# dname=[]
# for i in namelist1:
#     j=i.strip().split("/")[0]
#     if j not in dname:
#         dname.append(j)
# namelist2=open(namelist).readlines()
# sname=[]
# for i in namelist2:
#     if i.strip() not in sname:
#         sname.append(i.strip())
# print(len(dname))
# print(dname)
# print(len(sname))
# print(sname)

# t=os.listdir("/disk3/zbh/Datasets/2022_Q1_icu30/62346050e53b5981784bfa18/labels")
# print(len(t))





    




