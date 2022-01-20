# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import os.path
import os.path as osp
import random
import math

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class CropByBbox(object):

    def __init__(self, expand_prob=0.3, expand_ratio=0.1, save_crop_result=False):
        assert 0 <= expand_prob <= 1
        self.expand_prob = expand_prob
        self.expand_ratio = expand_ratio
        self.save_crop_result = save_crop_result

    def __call__(self, results):
        if np.random.rand() < self.expand_prob:
            xmin = math.ceil(max(0, results["bbox"][0] - results["bbox"][2] * self.expand_ratio))
        else:
            xmin = math.ceil(results["bbox"][0])

        if np.random.rand() < self.expand_prob:
            ymin = math.ceil(max(0, results["bbox"][1] - results["bbox"][3] * self.expand_ratio))
        else:
            ymin = math.ceil(results["bbox"][1])

        if np.random.rand() < self.expand_prob:
            xmax = math.ceil(min(results["ori_shape"][1], results["bbox"][0] + results["bbox"][2] * (1. + self.expand_ratio)))
        else:
            xmax = math.ceil(results["bbox"][0] + results["bbox"][2])

        if np.random.rand() < self.expand_prob:
            ymax = math.ceil(min(results["ori_shape"][0], results["bbox"][1] + results["bbox"][3] * (1. + self.expand_ratio)))
        else:
            ymax = math.ceil(results["bbox"][1] + results["bbox"][3])

        img = copy.deepcopy(results['img'][ymin:ymax, xmin:xmax, :])
        results['img'] = img
        results['img_shape'] = img.shape
        # if self.save_crop_result:
        #     results['img_bk'] = copy.deepcopy(img)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'expand_prob={self.expand_prob}, '
                    f'expand_ratio={self.expand_ratio})')
        return repr_str
