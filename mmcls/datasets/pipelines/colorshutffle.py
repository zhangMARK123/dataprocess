# Copyright (c) OpenMMLab. All rights reserved.

import sys
import cv2
import os.path
import random
import math
import numpy as np
from ..builder import PIPELINES
@PIPELINES.register_module()
class Colorshuffle(object):

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
        shape=results["boxshape"]       
        # h, w,c= img.shape
        # center=(w//2,h//2)

        if shape==0 and random.random() < self.prob:
            blue,green,red=cv2.split(img)
            img=cv2.merge((blue,red,green))
        results['img'] = img

        return results

    def __repr__(self):
        return self.__class__.__name__


