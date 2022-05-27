# Copyright (c) OpenMMLab. All rights reserved.
from .gap import GlobalAveragePooling
from .reshape_neck import ReshapeNeck
from .maxpoolgap import GlobalMaxPooling

__all__ = ['GlobalAveragePooling', 'ReshapeNeck','GlobalMaxPooling']
