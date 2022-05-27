# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class ReshapeNeck(nn.Module):

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple(
                [out.view(out.size(0), -1) for out in inputs])
        else:
            raise TypeError('neck inputs should be tuple')
        return outs
