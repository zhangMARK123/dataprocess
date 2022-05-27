# Copyright (c) OpenMMLab. All rights reserved.
import copy
import cv2
import os.path
import os.path as osp
import random

import mmcv
import numpy as np

from ..builder import PIPELINES



class Pad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
            # results['img_shape'] = padded_img.shape
            results['ori_shape'] = [800, 800, 3]

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class ResizeAndPad(object):
    """
    ① resize the image so that the longer side has the same length as size, while keeping the aspect ratio
    ② put the image in the center of the size x size black bg
    """
    def __init__(self,
                 size,
                 backend='cv2'
                 ):
        assert isinstance(size, int)
        self.size = size
        self.backend = backend

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            img_height = img.shape[0]
            img_width = img.shape[1]
            if img_height < 4 or img_width < 4:
                print(results["img_info"])
                print("img_height or img_width should above 4 pixel.")
            # assert img_height >= 4 and img_width >= 4, "img_height or img_width should above 4 pixel."
            assert img_height >= 4 and img_width >= 4, results["img_info"]
            longer_side = max(img_height, img_width)
            resize_factor = self.size / longer_side
            resize_height = int(np.round(img_height * resize_factor))
            resize_width = int(np.round(img_width * resize_factor))
            img = mmcv.imresize(img, (resize_width, resize_height), backend=self.backend)

            bg_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)

            if img_height > img_width:
                bg_img[:, int((self.size-resize_width)/2):int((self.size-resize_width)/2)+resize_width] = img
            else:
                bg_img[int((self.size-resize_height)/2):int((self.size-resize_height)/2)+resize_height, :] = img

            results[key] = bg_img
            return results