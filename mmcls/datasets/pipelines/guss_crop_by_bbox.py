# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
import cv2
import os.path
import os.path as osp
import random
import math

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class SimDet(object):
    """Simulate the detector effect.

    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, u=0, std=0.1, prob=0.5):
        self.u = u
        self.std = std
        self.prob = prob

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        # get tl coord
        img = results['img']
        coord_str_list = results['filename'].replace('.jpg', '').split('-')[-4:]
        xmin = int(results['bbox'][0])
        ymin = int(results['bbox'][1])
        xmax = math.ceil(results['bbox'][0] + results['bbox'][2])
        ymax = math.ceil(results['bbox'][1] + results['bbox'][3])
        # try:
        #     xmin, ymin, xmax, ymax = [int(each) for each in coord_str_list]  # tl coordinate in img
        # except Exception as e:
        #     print(e)
        #     print(coord_str_list)
        #     sys.exit(1)

        # set max shift dist
        h, w, c = img.shape

        if random.random() < self.prob:
            short_size = min(xmax-xmin, ymax-ymin)
            max_shifted_size = short_size // 5
            shifted_range = np.array(range(-max_shifted_size, max_shifted_size+1))

            # sample a shifted dist
            u = self.u
            std = self.std
            mi = -((shifted_range - u) * (shifted_range - u) / 2 / std / std)
            gauss_distribute = np.exp(mi) / ((2*np.pi)**0.5) / std
            xmin_shifted_size, ymin_shifted_size, xmax_shifted_size, ymax_shifted_size = random.choices(shifted_range, gauss_distribute, k=4)[0:4]
        else:
            xmin_shifted_size = 0
            ymin_shifted_size = 0
            xmax_shifted_size = 0
            ymax_shifted_size = 0

        # compute sim coord
        new_xmin = max(xmin + xmin_shifted_size, 0)
        new_ymin = max(ymin + ymin_shifted_size, 0)
        new_xmax = min(w, xmax + xmax_shifted_size)
        new_ymax = min(h, ymax + ymax_shifted_size)
        img = img[new_ymin: new_ymax, new_xmin:new_xmax]
        results['img'] = img

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class GussCropByBbox(object):

    def __init__(self, expand_prob=0.8, expand_ratio_range=[0.5,0.1], save_crop_result=False):
        assert 0 <= expand_prob <= 1
        self.expand_prob = expand_prob
        self.expand_ratio_range = expand_ratio_range
        self.save_crop_result = save_crop_result

    def __call__(self, results):
        max_l=max(results["bbox"][2],results["bbox"][3])
        resized_l=max(results["bbox"][2],results["bbox"][3])*min(max(random.gauss(2,1),1.6),2.5)
        expand_ratio_x1=max(min(random.gauss(self.expand_ratio_range[0],self.expand_ratio_range[1]),0.3),0.7)
        expand_ratio_y1=max(min(random.gauss(self.expand_ratio_range[0],self.expand_ratio_range[1]),0.3),0.7)
        prob=np.random.rand()
        
        if prob < self.expand_prob:
            xmin = results["bbox"][0] -max_l * expand_ratio_x1
        else:
            xmin = results["bbox"][0]-max_l/2

        if prob< self.expand_prob:
            ymin = results["bbox"][1] - max_l * expand_ratio_y1
        else:
            ymin = results["bbox"][1]-max_l/2
        if prob< self.expand_prob:
            xmax=xmin+resized_l
        else:
            xmax=xmin+max_l*2
        if prob< self.expand_prob:
            ymax=ymin+resized_l
        else:
            ymax=ymin+max_l*2
        ymin=int(min(max(0,ymin),results["bbox"][1]))
        xmin=int(min(max(0,xmin),results["bbox"][0]))
        ymax=int(min(max(ymax,results["bbox"][1]+results["bbox"][3]+5),results["ori_shape"][0]))
        xmax=int(min(max(xmax,results["bbox"][0]+results["bbox"][2]+5),results["ori_shape"][1]))
        
        img = copy.deepcopy(results['img'][ymin:ymax, xmin:xmax, :])
        results['img'] = img
        results['img_shape'] = img.shape
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'expand_prob={self.expand_prob}, '
                    f'expand_ratio={self.expand_ratio})')
        return repr_str
