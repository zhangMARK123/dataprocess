import math
import os, json, random, cv2


def analyze_target_list(target_list):
    CHARACTER_pass = CHARACTER_president = CHARACTER_number = CHARACTER_word = CHARACTER_others = CHARACTER_unknow = 0
    COLOR_red = COLOR_yellow = COLOR_green = COLOR_others = COLOR_unknown = COLOR_black = 0
    SHAPE_circle = SHAPE_return = SHAPE_bicycle = SHAPE_up = SHAPE_others = SHAPE_left = SHAPE_right = SHAPE_down = 0
    TOWARD_front = TOWARD_side = TOWARD_unknown = TOWARD_backside = 0
    SIMPLE_light = COMPLEX_light = 0
    for object in target_list:
        if object["bbox"][2]<10 or object["bbox"][3]<10:
            continue
        if object["simplelight_head"]:
            if object["simplelight"] == 0:
                SIMPLE_light += 1
            else:
                COMPLEX_light += 1
        if object["lightboxcolor_head"]:
            if object["boxcolor"] == 0:  # red
                COLOR_red += 1
            elif object["boxcolor"] == 1:  # yellow
                COLOR_yellow += 1
            elif object["boxcolor"] == 2:  # green
                COLOR_green += 1
            elif object["boxcolor"] == 3:  # black
                COLOR_black += 1
            elif object["boxcolor"] == 4:  # others
                COLOR_others += 1
            elif object["boxcolor"] == 5:  # unknown
                COLOR_unknown += 1
        if object["lightboxshape_head"]:
            if object["boxshape"] == 0:  # 圆
                SHAPE_circle += 1
            elif object["boxshape"] == 1:  # 上箭头
                SHAPE_up += 1
            elif object["boxshape"] == 2:  # 下箭头
                SHAPE_down += 1
            elif object["boxshape"] == 3:  # 左箭头
                SHAPE_left += 1
            elif object["boxshape"] == 4:  # 右箭头
                SHAPE_right += 1
            elif object["boxshape"] == 5:  # 掉头箭头
                SHAPE_return += 1
            elif object["boxshape"] == 6:  # 非机动车箭头
                SHAPE_bicycle += 1
            elif object["boxshape"] == 7:  # 其他
                SHAPE_others += 1
        if object["toward_head"]:
            if object["toward_orientation"] == 0:
                TOWARD_front += 1
            elif object["toward_orientation"] == 1:
                TOWARD_side += 1
            elif object["toward_orientation"] == 2:
                TOWARD_backside += 1
            elif object["toward_orientation"] == 3:
                TOWARD_unknown += 1

        if object["character_head"]:
            if object["characteristic"] == 0:  # 通行灯
                CHARACTER_pass += 1
            elif object["characteristic"] == 1:  # 行人灯
                CHARACTER_president += 1
            elif object["characteristic"] == 2:  # 数字灯
                CHARACTER_number += 1
            elif object["characteristic"] == 3:  # 符号灯
                CHARACTER_word += 1
            elif object["characteristic"] == 4:  # 其他灯
                CHARACTER_others += 1
            elif object["characteristic"] == 5:  # 未知
                CHARACTER_unknow += 1
    print("##############   Color Head   ##############")
    print("number of red: ", COLOR_red)
    print("number of yellow: ", COLOR_yellow)
    print("number of green: ", COLOR_green)
    print("number of others: ", COLOR_others)
    print("number of unknown: ", COLOR_unknown)
    print("number of black: ",COLOR_black)
    print("##############   Shape Head   ##############")
    print("number of circle: ", SHAPE_circle)
    print("number of left arrow: ", SHAPE_left)
    print("number of bicycle: ", SHAPE_bicycle)
    print("number of up arrow: ", SHAPE_up)
    print("number of right arrow: ", SHAPE_right)
    print("number of return arrow: ", SHAPE_return)
    print("number of others: ", SHAPE_others)
    print("##############   Toward Head   ##############")
    print("number of front: ", TOWARD_front)
    print("number of side: ", TOWARD_side)
    print("number of unknown: ", TOWARD_unknown)
    print(" ")
    print("##############   Character Head   ##############")
    print("number of CHARACTER_pass: ", CHARACTER_pass)
    print("number of CHARACTER_president: ", CHARACTER_president)
    print("number of CHARACTER_number: ", CHARACTER_number)
    print("number of CHARACTER_word: ", CHARACTER_word)
    print("number of CHARACTER_others: ", CHARACTER_others)
    print("number of CHARACTER_unknow: ", CHARACTER_unknow)
    print(" ")

    number_simple_dict = {"simple": SIMPLE_light,
                          "complex": COMPLEX_light,
                          }
    number_color_dict = {"red": COLOR_red,
                         "yellow": COLOR_yellow,
                         "green": COLOR_green,
                         "others": COLOR_others,
                         "unknown": COLOR_unknown,
                         "black": COLOR_black,
                         }
    number_shape_dict = {"circle": SHAPE_circle,
                         "left arrow": SHAPE_left,
                         "bicycle": SHAPE_bicycle,
                         "up arrow": SHAPE_up,
                         "right arrow": SHAPE_right,
                         "others": SHAPE_others,
                         "return arrow": SHAPE_return,
                         "down arrow": SHAPE_down,
                         }
    number_toward_dict = {"front": TOWARD_front,
                          "side": TOWARD_side,
                          "backside": TOWARD_backside,
                          "unknown": TOWARD_unknown,
                          }

    number_character_dict = {"CHARACTER_pass": CHARACTER_pass,
                             "CHARACTER_president": CHARACTER_president,
                             "CHARACTER_number": CHARACTER_number,
                             "CHARACTER_word": CHARACTER_word,
                             "CHARACTER_others": CHARACTER_others,
                             "CHARACTER_unknow": CHARACTER_unknow,
                             }

    return number_color_dict, number_shape_dict, number_toward_dict, number_character_dict, number_simple_dict


data_root = "/disk3/zbh/Datasets/2022_Q1_icu30_test/"
train_set_json_file =data_root+ "../2022_Q1_QA_train.json"
test_set_json_file =data_root+ "../2022_Q1_QA_test.json"
debug_set_json_file =data_root+ "../2022_Q1_QA_test_debug.json"

card_id_list = []
object_list = []
OBJECT_COUNT = 0
for data_card_id in os.listdir(data_root):
    if data_card_id not in ["61ee749f761998edc2524071",
"61ef593d761998edc2779eeb",
"61f0ad7b761998edc2b70800",
"6204e6aad3572380d59a5701",
"620798ccd3572380d56192c3",
"6209f9a0d3572380d56f0080",
"6209fb57d3572380d5767ff4",
"6209fdc1d3572380d577ad42",
"6212f759c4792a209de71519",
"6212f808c4792a209de728e8",
"62159f3fecf096dc1b785e1b",
"6215ca2fecf096dc1b0f8e4a",
"6215e0edecf096dc1b612251",
"6231aa8b625d0d750434f5ca",
"623454b1e53b5981783de76b",
"623455b1969d98de67bb444b",
"62345e50e53b59817847f749",
"6239b096e53b59817895380f",
"6239b25ce53b598178993f7d",
"6239b440e53b598178a054e0",
"6239b555e53b598178a07375",
"6239b6eae53b598178a34299",
"623c1b53e53b59817862445f",
"623c1d12e53b59817863f614",
"623c1e29e53b59817864a1b5",
"623d2eefe53b5981784c7057",
"623d383be53b5981786b7f0d"]:
        continue
    for idx, json_file in enumerate(os.listdir(os.path.join(data_root, data_card_id, "labels"))):
        with open(os.path.join(data_root, data_card_id, "labels", json_file)) as f:
            sample_json = json.load(f)
            img_name = json_file.split(".")[0] + ".jpg"
            images_path = os.path.join(os.path.join(data_root, data_card_id, "images", img_name))
            index_per_img = 0
            imgw = sample_json["width"]
            imgh = sample_json["height"]
            for object in sample_json["objects"]:
                object["character_head"] = True  # 默认训练所有Head
                object["toward_head"] = True
                object["lightboxcolor_head"] = True
                object["lightboxshape_head"] = True
                object["simplelight_head"] = True 
                ########################先对图像进行裁剪并保存,并对齐新框位置#####################
                bbox_ori = object["bbox"]
                if(bbox_ori[2]<0 or bbox_ori[3]<0):
                    continue
                max_w = math.ceil(max(bbox_ori[2], bbox_ori[3]) * 3)
                bbox_crop = [max(0, bbox_ori[0] - (max_w - bbox_ori[2]) / 2),
                             max(0, bbox_ori[1] - (max_w - bbox_ori[3]) / 2),
                             min(bbox_ori[0] + bbox_ori[2] + (max_w - bbox_ori[2]) / 2 + 1, imgw),
                             min(bbox_ori[1] + bbox_ori[3] + (max_w - bbox_ori[3]) / 2 + 1, imgh)]
                bbox_crop = list(map(int, bbox_crop))

                if bbox_crop[0] == 0:
                    x1 = max(bbox_crop[2] - (max_w + bbox_ori[2]) / 2, 0)
                else:
                    x1 = max((max_w - bbox_ori[2]) / 2, 0)
                if bbox_crop[1] == 0:
                    y1 = max(bbox_crop[3] - (max_w + bbox_ori[3]) / 2, 0)
                else:
                    y1 = max((max_w - bbox_ori[3]) / 2, 0)
                object["bbox"] = [x1, y1, bbox_ori[2], bbox_ori[3]]  # 框在子图的坐标
                #########总筛选 框小于6 遮挡截断的灯箱不计入总数,直接跳过即可,框小于10，只训练颜色朝向简单复杂灯######################
                # 因为标签没有灯箱颜色形状是否为简单灯情况，先预设一个。之后只要head为0不影响其训练
                object["boxcolor"] = 5   ##颜色未知
                object["boxshape"] = 7  ##形状unknow
                object["simplelight"] = 0  # 默认简单灯

                # 若朝向为背面，颜色形状和是否为简单灯都不计入训练。规定都为others方便格式统一，但都不计入训练
                if object["toward_orientation"] == 2 or object["toward_orientation"] == 3:
                    object["lightboxcolor_head"] = False
                    object["lightboxshape_head"] = False
                    object["simplelight_head"] = False
                    object["character_head"] = False
                # 若非通行灯，简单灯head不计入训练，但颜色和形状是否记入训练？？颜色计入训练
                ##此时的颜色如何分辨？
                if object["characteristic"] != 0:
                    object["simplelight_head"] = False
                    object["lightboxshape_head"] = False
                    ##如果小车，把这个head打开
                    object["lightboxcolor_head"] = False
                ###小车需要训练行人灯颜色
                # if object["characteristic"]!=0 or object["characteristic"]!=1:
                #      object["lightboxcolor_head"] = False

                ##若子灯个数为0，不加入训练。此类注意重点查看一下。此类只有背面或者unknow情况
                if object["num_sub_lights"] == 0:
                    object["lightboxcolor_head"] = False
                    object["lightboxshape_head"] = False
                    # if max(object["bbox"][2], object["bbox"][3]) / min(object["bbox"][2], object["bbox"][3]) < 3.5:
                    #     object["simplelight_head"] = False

                ################### 1. 判断是看否为简单灯####################################
                # 若灯箱颜色只有红黄绿三种一种其余为全黑或者灯箱全黑、未知，则为简单灯，该灯颜色为灯箱颜色，形状为灯箱形状,否则为复杂灯,先不考虑复杂箭头，数据多之后统计复杂箭头类型加类别
                if object["num_sub_lights"] != 0 and ("sub_lights" in object.keys()):
                    num_color = 0
                    num_black=0
                    subcolor = 5
                    subshape = 7
                    for sub in object["sub_lights"]:
                        if sub["color"]==3:
                            num_black+=1
                        if sub["color"] in [0, 1, 2,4]:
                            num_color += 1
                            subcolor = sub["color"]
                            if sub["shape"] == 0:
                                subshape = 0
                            elif sub["shape"] == 1:     # 非机动车
                                subshape = 6
                            elif sub["shape"] == 2:                              
                                if sub["arrow_orientation"]["up_arrow"]==1:
                                    subshape=1
                                elif sub["arrow_orientation"]["down_arrow"]==1:
                                    subshape=2
                                elif sub["arrow_orientation"]["left_arrow"]==1:
                                    subshape=3
                                elif sub["arrow_orientation"]["right_arrow"]==1:
                                    subshape=4
                                elif sub["arrow_orientation"]["u_turn"]==1:
                                    subshape=5
                    if num_color == 1 or num_color == 0:    # 只有一个子灯，或者全部unknown，全部黑。有个问题，有没有可能子灯颜色全部标注unknown
                        object["boxcolor"] = subcolor
                        object["boxshape"] = subshape
                        object["simplelight"] = 0
                        if num_black==object["num_sub_lights"]:
                            object["boxcolor"] = 3
                    else:
                        object["lightboxcolor_head"] = False
                        object["lightboxshape_head"] = False
                        object["simplelight"] = 1
                if object["simplelight"] == 1:
                    object["sublightcolor_head"]=True
                    # object["sublightshape_head"]=True
                    object["boxcolor"] = 3
                    object["boxshape"] = 7
                    num_sub=object["num_sub_lights"]
                OBJECT_COUNT += 1
                object["id"] = '{:0>10d}'.format(OBJECT_COUNT)
                object["img_info"] = {
                    'filename': "images/" + json_file.split(".")[0] + "_" + str(index_per_img) + ".jpg"}

                index_per_img += 1
                object["data_card_id"] = data_card_id
               
                object_list.append(object)
                card_id_list.append(data_card_id)
card_id_list = list(set(card_id_list))
if True:
    ###数据集2/8分为训练集测试集
    random.shuffle(object_list)
    random.shuffle(object_list)
    train_object_list = []
    test_object_list = []
    for idx, object in enumerate(object_list):
        if idx <= int(1 * len(object_list)):
            train_object_list.append(object)
        else:
            test_object_list.append(object)
    print("number of train set : ", len(train_object_list))
    print("number of test set : ", len(test_object_list))
    number_color_dict, number_shape_dict, number_toward_dict, number_character_dict, number_simple_dict = analyze_target_list(
        test_object_list)
    with open(debug_set_json_file,'w+') as f:
        f.write(json.dumps({"card_id": card_id_list,
                            "postscript": "Delete targets with a width less than 4 pixels.",
                            "number_color_dict": number_color_dict,
                            "number_shape_dict": number_shape_dict,
                            "number_toward_dict": number_toward_dict,
                            "number_character_dict": number_character_dict,
                            "number_simple_dict": number_simple_dict,
                            "objects": test_object_list[:int(0.01 * len(test_object_list))]}, ensure_ascii=False,
                           indent=4))
    print("deal debug set done.")

    with open(test_set_json_file,'w+') as f:
        f.write(json.dumps({"card_id": card_id_list,
                            "postscript": "Delete targets with a width less than 4 pixels.",
                            "number_color_dict": number_color_dict,
                            "number_shape_dict": number_shape_dict,
                            "number_toward_dict": number_toward_dict,
                            "number_character_dict": number_character_dict,
                            "number_simple_dict": number_simple_dict,
                            "objects": test_object_list}, ensure_ascii=False, indent=4))
    print("deal test set done.")
    number_color_dict, number_shape_dict, number_toward_dict, number_character_dict, number_simple_dict = analyze_target_list(
        train_object_list)
    with open(train_set_json_file,'w+') as f:
        f.write(json.dumps({"card_id": card_id_list,
                            "postscript": "Delete targets with a width less than 6 pixels.",
                            "number_color_dict": number_color_dict,
                            "number_shape_dict": number_shape_dict,
                            "number_toward_dict": number_toward_dict,
                            "number_character_dict": number_character_dict,
                            "number_simple_dict": number_simple_dict,
                            "objects": train_object_list}, ensure_ascii=False, indent=4))
    print("deal train set done.")
