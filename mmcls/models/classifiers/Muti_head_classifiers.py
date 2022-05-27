# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
import torch
import torch.distributed as dist

from ..builder import CLASSIFIERS
from .image import ImageClassifier

@CLASSIFIERS.register_module()
class MultiHeadClassifier(ImageClassifier):

    def forward_train(self, img, **kwargs):
        
        x = self.extract_feat(img)
        
        losses = dict()
        try:
            loss = self.head.forward_train(x, **kwargs)
        except TypeError as e:
            if 'not tuple' in str(e) and self.return_tuple:
                return TypeError(
                    'Seems the head cannot handle tuple input. We have '
                    'changed all backbones\' output to a tuple. Please '
                    'update your custom head\'s forward function. '
                    'Temporarily, you can set "return_tuple=False" in '
                    'your backbone config to disable this feature.')
            raise e

        losses.update(loss)
        
        return losses
