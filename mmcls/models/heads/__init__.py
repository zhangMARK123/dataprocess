# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .stacked_head import StackedLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .traffic_sublight_multi_head import TrafficSubLightMultiClsHead
from .traffic_sublight_complex_multi_head import TrafficSubLightcomplexMultiClsHead
__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'DeiTClsHead',
    'ConformerHead','TrafficSubLightMultiClsHead','TrafficSubLightcomplexMultiClsHead'
]
