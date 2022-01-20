# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .Muti_head_classifiers import MultiHeadClassifier

__all__ = ['BaseClassifier', 'ImageClassifier','MultiHeadClassifier']
