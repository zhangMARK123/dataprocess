# Copyright (c) OpenMMLab. All rights reserved.
import sys
import cv2
import os.path
import random
import math
import numpy as np
from ..builder import PIPELINES
@PIPELINES.register_module()
class Rotateimg():

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
        h, w,c= img.shape
      
       
        # center=(w//2,h//2) 
        # results["toward_orientation"]==0 是否加这条？
        if random.random() < self.prob and h>w and results["toward_orientation"]==0 and results["simplelight"]==0: 
              
            if results["boxshape"]==0:
                img=cv2.transpose(img)
            elif results["boxshape"]==1:
                img=cv2.transpose(img)
                if random.random()<0.5:                   
                    results["boxshape"]=3
                else: 
                    img=cv2.flip(img,1) 
                    results["boxshape"]=4    
            elif results["boxshape"]==3:
                img=cv2.transpose(img)
                results["boxshape"]=1
            elif results["boxshape"]==4:
                img=cv2.transpose(img)
                img=cv2.flip(img,0)
                results["boxshape"]=1
            elif results["boxshape"]==7 and results["boxcolor"]==3:
                img=cv2.transpose(img)


        results['img'] = img

        return results

    def __repr__(self):
        return self.__class__.__name__


