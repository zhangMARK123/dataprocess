# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class GlobalMaxPooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalMaxPooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveMaxPool1d(1)
        elif dim == 2:
            self.gap=nn.AdaptiveMaxPool2d((1,1))
        else:
            self.gap = nn.AdaptiveMaxPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple): 
            outs = tuple([self.gap(x) for x in inputs])
            z=torch.cat((outs[0],outs[1]),1)        
            # w=nn.Conv2d(576,512,(1,1)).cuda()(z) 
            outs = tuple(
                [z.view(inputs[0].size(0), -1)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
