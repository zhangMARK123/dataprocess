# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import HEADS
from .cls_head import ClsHead
from ..utils import is_tracing


@HEADS.register_module()
class TrafficSubLightsplitMultiClsHead(ClsHead):

    def __init__(self,
                 shape_classes,
                 color_classes,
                 toward_classes,
                 character_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 freeze_color_head=False,
                 freeze_shape_head=False,
                 freeze_toward_head=False,
                 freeze_character_head=False,
                 *args,
                 **kwargs):
        super(TrafficSubLightsplitMultiClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.shape_classes = shape_classes
        self.color_classes = color_classes
        self.toward_classes = toward_classes
        self.character_classes = character_classes

        if self.shape_classes <= 0:
            raise ValueError(
                f'num_classes={shape_classes} must be a positive integer')
        if self.color_classes <= 0:
            raise ValueError(
                f'num_classes={color_classes} must be a positive integer')
        if self.toward_classes <= 0:
            raise ValueError(
                f'num_classes={toward_classes} must be a positive integer')
        if self.character_classes <= 0:
            raise ValueError(
                f'num_classes={character_classes} must be a positive integer')
        self.fc_shape = nn.Linear(self.in_channels, self.shape_classes)
        self.fc_color = nn.Linear(self.in_channels, self.color_classes)
        self.fc_toward = nn.Linear(self.in_channels, self.toward_classes)
        self.fc_character = nn.Linear(self.in_channels, self.character_classes)
        # print("fc_shape:", self.fc_shape)
        self.freeze_color_head = freeze_color_head
        self.freeze_shape_head = freeze_shape_head
        self.freeze_toward_head = freeze_toward_head
        self.freeze_character_head = freeze_character_head
        if self.freeze_color_head and self.freeze_shape_head and self.freeze_toward_head:
            raise ValueError(
                'can not freeze all head at the same time.')
        self._freeze_head()

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]
        color_cls_score = self.fc_color(x)
        shape_cls_score = self.fc_shape(x)
        toward_cls_score = self.fc_toward(x)
        character_cls_score = self.fc_character(x)
        
        if isinstance(color_cls_score, list):
            color_cls_score = sum(color_cls_score) / float(len(color_cls_score))
        if isinstance(shape_cls_score, list):
            shape_cls_score = sum(shape_cls_score) / float(len(shape_cls_score))
        if isinstance(toward_cls_score, list):
            toward_cls_score = sum(toward_cls_score) / float(len(toward_cls_score))
        if isinstance(character_cls_score, list):
            character_cls_score = sum(character_cls_score) / float(len(character_cls_score))
        color_pred = F.softmax(color_cls_score, dim=1) if color_cls_score is not None else None
        shape_pred = F.softmax(shape_cls_score, dim=1) if shape_cls_score is not None else None
        toward_pred = F.softmax(toward_cls_score, dim=1) if shape_cls_score is not None else None
        character_pred = F.softmax(character_cls_score, dim=1) if character_cls_score is not None else None
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return [color_pred, shape_pred, toward_pred, character_pred]
        pred = np.concatenate((color_pred.detach().cpu().numpy(),
                               shape_pred.detach().cpu().numpy(),
                               toward_pred.detach().cpu().numpy(),
                               character_pred.detach().cpu().numpy(),
                               ), axis=1)
        return pred

    def forward_train(self, x, **kwargs):
        if isinstance(x, tuple):
            x = x[-1]
        color_cls_score = self.fc_color(x)
        shape_cls_score = self.fc_shape(x)
        toward_cls_score = self.fc_toward(x)
        character_cls_score = self.fc_character(x)

        losses = dict()
        # compute loss
        color_score_selected = color_cls_score[torch.squeeze(kwargs["color_head"] == 1)]
        color_gt_selected = kwargs['color'][torch.squeeze(kwargs["color_head"] == 1)]
        color_loss = self.compute_loss(color_score_selected, color_gt_selected, avg_factor=len(color_score_selected))

        shape_score_selected = shape_cls_score[torch.squeeze(kwargs['shape_head'] == 1)]
        shape_gt_selected = kwargs['shape'][torch.squeeze(kwargs['shape_head'] == 1)]
        shape_loss = self.compute_loss(shape_score_selected, shape_gt_selected, avg_factor=len(shape_score_selected))

        

        toward_score_selected = toward_cls_score[torch.squeeze(kwargs['toward_head'] == 1)]
        toward_gt_selected = kwargs['toward'][torch.squeeze(kwargs['toward_head'] == 1)]
        toward_loss = self.compute_loss(toward_score_selected, toward_gt_selected,
                                        avg_factor=len(toward_score_selected))

        character_score_selected = character_cls_score[torch.squeeze(kwargs['character_head'] == 1)]
        character_gt_selected = kwargs['characteristic'][torch.squeeze(kwargs['character_head'] == 1)]
        character_loss = self.compute_loss(character_score_selected, character_gt_selected,
                                           avg_factor=len(character_score_selected))

        if self.cal_acc:
            # compute accuracy
            color_acc = self.compute_accuracy(color_score_selected, color_gt_selected)
            assert len(color_acc) == len(self.topk)
            losses['color_accuracy'] = {
                f'color_top-{k}': a
                for k, a in zip(self.topk, color_acc)
            }
            # compute accuracy
            shape_acc = self.compute_accuracy(shape_score_selected, shape_gt_selected)
            assert len(shape_acc) == len(self.topk)
            losses['shape_accuracy'] = {
                f'shape_top-{k}': a
                for k, a in zip(self.topk, shape_acc)
            }
            # compute accuracy
            toward_acc = self.compute_accuracy(toward_score_selected, toward_gt_selected)
            assert len(toward_acc) == len(self.topk)
            losses['toward_accuracy'] = {
                f'toward_top-{k}': a
                for k, a in zip(self.topk, toward_acc)
            }
            # compute accuracy
            character_acc = self.compute_accuracy(character_score_selected, character_gt_selected)
            assert len(character_acc) == len(self.topk)
            losses['character_accuracy'] = {
                f'character_top-{k}': a
                for k, a in zip(self.topk, character_acc)
            }
        if not self.freeze_color_head:
            losses['color_loss'] = {'color_loss': color_loss}
        if not self.freeze_shape_head:
            losses['shape_loss'] = {'shape_loss': shape_loss}
        if not self.freeze_toward_head:
            losses['toward_loss'] = {'toward_loss': toward_loss}
        if not self.freeze_character_head:
            losses['character_loss'] = {'character_loss': character_loss}

        return losses

    def train(self, mode=True):
        super(TrafficSubLightsplitMultiClsHead, self).train(mode)
        self._freeze_head()

    def _freeze_head(self):
        if self.freeze_color_head:
            self.fc_color.eval()
            for param in self.fc_color.parameters():
                param.requires_grad = False
        if self.freeze_shape_head:
            self.fc_shape.eval()
            for param in self.fc_shape.parameters():
                param.requires_grad = False
