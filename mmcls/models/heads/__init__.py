# Copyright (c) OpenMMLab. All rights reserved.
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .deit_head import DeiTClsHead
from .linear_head import LinearClsHead
from .multi_label_head import MultiLabelClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .stacked_head import StackedLinearClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .traffic_sublight_complex_multi_head import TrafficSubLightcomplexMultiClsHead
from .traffic_sublight1_complex_multi_head import TrafficSubLightcomplexMultiClsHead1
from .traffic_sublight_night_multi_head import TrafficSubLightneightMultiClsHead
from .traffic_sublight_split_multi_head import TrafficSubLightsplitMultiClsHead
__all__ = [
    'ClsHead', 'LinearClsHead', 'StackedLinearClsHead', 'MultiLabelClsHead',
    'MultiLabelLinearClsHead', 'VisionTransformerClsHead', 'DeiTClsHead','TrafficSubLightneightMultiClsHead',
    'ConformerHead','TrafficSubLightcomplexMultiClsHead','TrafficSubLightcomplexMultiClsHead1','TrafficSubLightsplitMultiClsHead'
]
