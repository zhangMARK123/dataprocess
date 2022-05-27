import onnxruntime as rt
import numpy as np
import cv2
import copy
import math
import numpy as np
from argparse import ArgumentParser
import os
from mmcls.apis import init_model,inference_sublightmodel



COLOR_CLASSES = [
    'red',
    'yellow',
    'green',
    'black',
    'others',
    'unknow'
]
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
    
model = init_model("../work_dirs/resnet18_sampler2/resnet18_trafficlightcomplex.py", "../work_dirs/resnet18_sampler2/iter_80000.pth", device='cpu')
rootpath="testonnximg/61e5171c910d7200e711bf5c/images"
sess = rt.InferenceSession('../work_dirs/resnet18_sampler2/batch_multi_head_cls2.onnx')
input_name = sess.get_inputs()[0].name
pred_score = sess.get_outputs()[0].name
pred_shape = sess.get_outputs()[1].name
pred_toward = sess.get_outputs()[2].name
pred_character=sess.get_outputs()[3].name
pred_simple=sess.get_outputs()[4].name
for i in os.listdir(rootpath):
    imgpath=os.path.join(rootpath,i)
    print(i)
    print("=====================================")
    img = cv2.imread(imgpath)
    img_roi = copy.deepcopy(img)
    longer_side = max(img_roi.shape[0], img_roi.shape[1])
    resize_factor = 128.0 / longer_side
    resize_height = int(np.round(img_roi.shape[0] * resize_factor))
    resize_width = int(np.round(img_roi.shape[1] * resize_factor))
    print("resize_height: ", resize_height, "resize_width: ", resize_width)
    img_roi = cv2.resize(img_roi, (resize_width, resize_height), dst=None, interpolation=cv2.INTER_LINEAR)
    total_pad = 128 - min(img_roi.shape[1],img_roi.shape[0])
    if img_roi.shape[1]<img_roi.shape[0]:
        img_roi = cv2.copyMakeBorder(img_roi, 0, 128 - img_roi.shape[0], int(total_pad / 2),
                                        total_pad - int(total_pad / 2), cv2.BORDER_CONSTANT, 0).astype(np.float32)[None, ...]
    else:
        img_roi = cv2.copyMakeBorder(img_roi, int(total_pad / 2),
                                        total_pad - int(total_pad / 2), 0,128 - img_roi.shape[1],cv2.BORDER_CONSTANT, 0).astype(np.float32)[None, ...]

    #print(img_roi.shape)
    img_roi = img_roi[:, :, :, [2, 1, 0]]
    img_roi = img_roi.transpose(0, 3, 1, 2)
    img_roi = (img_roi - np.array([0, 0, 0], dtype=np.float)[None, :, None, None]) / np.array(
        [255., 255., 255.], dtype=np.float)[None, :, None, None]

    pred_color, pred_shape, pred_toward,pred_character,pred_simple= sess.run(None, {input_name: img_roi.astype(np.float32)})
    color_score = np.max(pred_color[0])
    shape_score = np.max(pred_shape[0])
    toward_score = np.max(pred_toward[0])
    character_score=np.max(pred_character[0])
    simple_score=np.max(pred_simple[0])
    color_label = np.argmax(pred_color[0])
    shape_label = np.argmax(pred_shape[0])
    toward_label = np.argmax(pred_toward[0])
    character_label=np.argmax(pred_character[0])
    simple_label=np.argmax(pred_simple[0])

    torchresult = inference_sublightmodel(model,imgpath)
    print(color_score)
    #print(color_label)
    print(shape_score)
    print(toward_score)
    print(character_score)
    print(simple_score)
    for key in torchresult.keys():
        #print(key)
        print(torchresult[key])



