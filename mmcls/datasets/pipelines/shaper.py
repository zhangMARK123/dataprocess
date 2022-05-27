# Copyright (c) OpenMMLab. All rights reserved.

import sys
import cv2
import os.path
import random
import math
import numpy as np
from ..builder import PIPELINES
@PIPELINES.register_module()
class Shaper(object):

    def __init__(self,prob=0.2):
        self.prob = prob

    def __call__(self, results):
        """Call function to flip image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """    
        img = results['img']
        # shape=results["boxshape"]
        kernel_sharpen4 = np.array([
        [0, -1, 0],
        [0, 5, 0],
        [0, -1, 0]])       
        # h, w,c= img.shape
        # center=(w//2,h//2)

        if random.random() < self.prob:
            blur = cv2.bilateralFilter(img, 5, 10, 10)
            img = cv2.filter2D(blur, -1, kernel_sharpen4)
        results['img'] = img

        return results

    def __repr__(self):
        return self.__class__.__name__


