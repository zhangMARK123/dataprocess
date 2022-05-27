# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import HEADS
from .cls_head import ClsHead
from ..utils import is_tracing


@HEADS.register_module()
class TrafficSubLightcomplexMultiClsHead1(ClsHead):

    def __init__(self,
                 shape_classes,
                 color_classes,
                 toward_classes,
                 character_classes,
                 simplelight_classes,
                 numsublight_classes,
                 sublightcolor_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 freeze_color_head=False,
                 freeze_shape_head=False,
                 freeze_toward_head=False,
                 freeze_character_head=False,
                 freeze_simplelight_head=False,
                 freeze_numsublight_head=False,
                 freeze_sublightcolor_head=False,
                 *args,
                 **kwargs):
        super(TrafficSubLightcomplexMultiClsHead1, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.shape_classes = shape_classes
        self.color_classes = color_classes
        self.toward_classes = toward_classes
        self.character_classes = character_classes
        self.simplelight_classes = simplelight_classes
        self.numsublight_classes=numsublight_classes
        self.sublightcolor_classes=sublightcolor_classes

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
        if self.simplelight_classes <= 0:
            raise ValueError(
                f'num_classes={simplelight_classes} must be a positive integer')
        if self.numsublight_classes <= 0:
            raise ValueError(
                f'num_classes={simplelight_classes} must be a positive integer')

        if self.sublightcolor_classes <= 0:
            raise ValueError(
                f'num_classes={sublightcolor_classes} must be a positive integer')
        self.fc_shape = nn.Linear(self.in_channels, self.shape_classes)
        self.fc_color = nn.Linear(self.in_channels, self.color_classes)
        self.fc_toward = nn.Linear(self.in_channels, self.toward_classes)
        self.fc_character = nn.Linear(self.in_channels, self.character_classes)
        self.fc_simplelight = nn.Linear(self.in_channels, self.simplelight_classes)

        self.fc_numsublight = nn.Linear(self.in_channels, self.numsublight_classes)
        self.fc_subcolor=nn.Linear(self.in_channels, self.sublightcolor_classes)
        # print("fc_numsublight:", self.fc_numsublight)
        self.freeze_color_head = freeze_color_head
        self.freeze_shape_head = freeze_shape_head
        self.freeze_toward_head = freeze_toward_head
        self.freeze_character_head = freeze_character_head
        self.freeze_simplelight_head = freeze_simplelight_head
        self.freeze_numsublight_head=freeze_numsublight_head
        self.freeze_sublightcolor_head=freeze_sublightcolor_head
        if self.freeze_color_head and self.freeze_shape_head and self.freeze_toward_head and self.freeze_simplelight_head:
            raise ValueError(
                'can not freeze all head at the same time.')
        self._freeze_head()

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]
       
        # return x.detach().cpu().numpy()
        color_cls_score = self.fc_color(x)
        shape_cls_score = self.fc_shape(x)
        toward_cls_score = self.fc_toward(x)
        character_cls_score = self.fc_character(x)
        simplelight_cls_score = self.fc_simplelight(x)
        numsublight_classes_score=self.fc_numsublight(x)
        subcolor_classes_score=self.fc_subcolor(x)
        if isinstance(color_cls_score, list):
            color_cls_score = sum(color_cls_score) / float(len(color_cls_score))
        if isinstance(shape_cls_score, list):
            shape_cls_score = sum(shape_cls_score) / float(len(shape_cls_score))
        if isinstance(toward_cls_score, list):
            toward_cls_score = sum(toward_cls_score) / float(len(toward_cls_score))
        if isinstance(character_cls_score, list):
            character_cls_score = sum(character_cls_score) / float(len(character_cls_score))
        if isinstance(simplelight_cls_score, list):
            simplelight_cls_score = sum(simplelight_cls_score) / float(len(simplelight_cls_score))
        if isinstance(numsublight_classes_score, list):
            numsublight_classes_score = sum(numsublight_classes_score) / float(len(numsublight_classes_score))
        if isinstance(subcolor_classes_score, list):
            subcolor_classes_score = sum(subcolor_classes_score) / float(len(subcolor_classes_score))

        color_pred = F.softmax(color_cls_score, dim=1) if color_cls_score is not None else None
        shape_pred = F.softmax(shape_cls_score, dim=1) if shape_cls_score is not None else None
        toward_pred = F.softmax(toward_cls_score, dim=1) if shape_cls_score is not None else None
        character_pred = F.softmax(character_cls_score, dim=1) if character_cls_score is not None else None
        simplelight_pred = F.softmax(simplelight_cls_score, dim=1) if simplelight_cls_score is not None else None
        ######复杂子灯

        numsublight_pred = F.softmax(numsublight_classes_score, dim=1) if numsublight_classes_score is not None else None
        
        subcolor_pred=F.softmax(subcolor_classes_score,dim=1) if subcolor_classes_score is not None else None
        
        on_trace = is_tracing()
        if torch.onnx.is_in_onnx_export() or on_trace:
            return [color_pred, shape_pred, toward_pred, character_pred, simplelight_pred,numsublight_pred,subcolor_pred]
        pred = np.concatenate((color_pred.detach().cpu().numpy(),
                               shape_pred.detach().cpu().numpy(),
                               toward_pred.detach().cpu().numpy(),
                               character_pred.detach().cpu().numpy(),
                               simplelight_pred.detach().cpu().numpy(),
                               numsublight_pred.detach().cpu().numpy(),
                               subcolor_pred.detach().cpu().numpy()
                               ), axis=1)
        return pred

    def forward_train(self, x, **kwargs):
        if isinstance(x, tuple):
            x = x[-1]
        color_cls_score = self.fc_color(x)
        shape_cls_score = self.fc_shape(x)
        toward_cls_score = self.fc_toward(x)
        character_cls_score = self.fc_character(x)
        simplelight_cls_score = self.fc_simplelight(x)
        numsublight_cls_score=self.fc_numsublight(x)
        subcolor_cls_score=self.fc_subcolor(x)

        losses = dict()

        # compute loss
        color_score_selected = color_cls_score[torch.squeeze(kwargs["lightboxcolor_head"] == 1)]
        color_gt_selected = kwargs['boxcolor'][torch.squeeze(kwargs["lightboxcolor_head"] == 1)]
        if len(color_gt_selected)!=0:
            color_loss = self.compute_loss(color_score_selected, color_gt_selected, avg_factor=len(color_score_selected))
        else:
            color_loss=torch.tensor(0.00000).cuda()

        shape_score_selected = shape_cls_score[torch.squeeze(kwargs['lightboxshape_head'] == 1)]
        shape_gt_selected = kwargs['boxshape'][torch.squeeze(kwargs['lightboxshape_head'] == 1)]
        if len(shape_gt_selected)!=0:
            shape_loss = self.compute_loss(shape_score_selected, shape_gt_selected, avg_factor=len(shape_score_selected))
        else:
            shape_loss=torch.tensor(0.00000).cuda()
        

        toward_score_selected = toward_cls_score[torch.squeeze(kwargs['toward_head'] == 1)]
        toward_gt_selected = kwargs['toward_orientation'][torch.squeeze(kwargs['toward_head'] == 1)]
        if len(toward_gt_selected)!=0:
            toward_loss = self.compute_loss(toward_score_selected, toward_gt_selected,
                                        avg_factor=len(toward_score_selected))
        else:
            toward_loss=torch.tensor(0.00000).cuda()

        character_score_selected = character_cls_score[torch.squeeze(kwargs['character_head'] == 1)]
        character_gt_selected = kwargs['characteristic'][torch.squeeze(kwargs['character_head'] == 1)]
        if len(character_gt_selected)!=0:
            character_loss = self.compute_loss(character_score_selected, character_gt_selected,
                                           avg_factor=len(character_score_selected))
        else:
            character_loss=torch.tensor(0.00000).cuda()

        simplelight_score_selected = simplelight_cls_score[torch.squeeze(kwargs['simplelight_head'] == 1)]
        simplelight_gt_selected = kwargs['simplelight'][torch.squeeze(kwargs['simplelight_head'] == 1)]
        
        if len(simplelight_gt_selected)!=0:
            simplelight_loss = self.compute_loss(simplelight_score_selected, simplelight_gt_selected,
                                             avg_factor=len(simplelight_score_selected))
        else:
            simplelight_loss=torch.tensor(0.00000).cuda()

        ###复杂子灯
        numsublight_score_selected = numsublight_cls_score[torch.squeeze(kwargs["numsublight_head"] == 1)]
        numsublight_gt_selected = kwargs["numcolorlight"][torch.squeeze(kwargs["numsublight_head"] == 1)]
        # print("==========debug3===================")
        # print(numsublight_score_selected.size())
        # print(numsublight_gt_selected.size())
        if len(numsublight_gt_selected)!=0:
            numsublight_loss = self.compute_loss(numsublight_score_selected, numsublight_gt_selected,
                                             avg_factor=len(numsublight_score_selected))
        else:
            numsublight_loss=torch.tensor(0.00000).cuda()
        ##子灯颜色
        subcolor1_score_selected = subcolor_cls_score[torch.squeeze(kwargs["sublightcolor_head"] == 1)][:,:6]
        subcolor1_gt_selected = kwargs["subcolor0"][torch.squeeze(kwargs["sublightcolor_head"] == 1)]
        if len(subcolor1_gt_selected)!=0:
             subcolor1_loss = self.compute_loss(subcolor1_score_selected, subcolor1_gt_selected, avg_factor=len(subcolor1_score_selected))
        else:
            subcolor1_loss=torch.tensor(0.00000).cuda()
        subcolor2_score_selected = subcolor_cls_score[torch.squeeze(kwargs["sublightcolor_head"] == 1)][:,6:12]
        subcolor2_gt_selected = kwargs["subcolor1"][torch.squeeze(kwargs["sublightcolor_head"] == 1)]
        if len(subcolor2_gt_selected)!=0:
            subcolor2_loss = self.compute_loss(subcolor2_score_selected, subcolor2_gt_selected, avg_factor=len(subcolor2_score_selected))
        else:
            subcolor2_loss=torch.tensor(0.00000).cuda()
        subcolor3_score_selected = subcolor_cls_score[torch.squeeze(kwargs["sublightcolor_head"] == 1)][:,12:18]
        subcolor3_gt_selected = kwargs["subcolor2"][torch.squeeze(kwargs["sublightcolor_head"] == 1)]
        if len(subcolor3_gt_selected)!=0:
            subcolor3_loss = self.compute_loss(subcolor3_score_selected, subcolor3_gt_selected, avg_factor=len(subcolor3_score_selected))
        else:
            subcolor3_loss=torch.tensor(0.00000).cuda()
        subcolor4_score_selected = subcolor_cls_score[torch.squeeze(kwargs["sublightcolor_head"] == 1)][:,18:24]
        subcolor4_gt_selected = kwargs["subcolor3"][torch.squeeze(kwargs["sublightcolor_head"] == 1)]
        if len(subcolor4_gt_selected)!=0:
            subcolor4_loss = self.compute_loss(subcolor4_score_selected, subcolor4_gt_selected, avg_factor=len(subcolor1_score_selected))
        else:
            subcolor4_loss=torch.tensor(0.00000).cuda()
       
        subcolor5_score_selected = subcolor_cls_score[torch.squeeze(kwargs["sublightcolor_head"] == 1)][:,24:30]
        subcolor5_gt_selected = kwargs["subcolor4"][torch.squeeze(kwargs["sublightcolor_head"] == 1)]
        if len(subcolor5_gt_selected)!=0:
            subcolor5_loss = self.compute_loss(subcolor5_score_selected, subcolor5_gt_selected, avg_factor=len(subcolor5_score_selected))
        else:
            subcolor5_loss=torch.tensor(0.00000).cuda()

        subcolor_loss =subcolor1_loss+subcolor2_loss+subcolor3_loss+subcolor4_loss+subcolor5_loss
        if self.cal_acc:
            # compute accuracy
            color_acc = self.compute_accuracy(color_score_selected, color_gt_selected)
            if len(color_acc)!=0:
                assert len(color_acc) == len(self.topk)
                losses['color_accuracy'] = {
                f'color_top-{k}': a
                for k, a in zip(self.topk, color_acc)
            }
            # compute accuracy
            shape_acc = self.compute_accuracy(shape_score_selected, shape_gt_selected)
            if len(shape_acc)!=0:
                assert len(shape_acc) == len(self.topk)
                losses['shape_accuracy'] = {
                f'shape_top-{k}': a
                for k, a in zip(self.topk, shape_acc)
            }
            # compute accuracy
            toward_acc = self.compute_accuracy(toward_score_selected, toward_gt_selected)
            if len(toward_acc)!=0:
                assert len(toward_acc) == len(self.topk)
                losses['toward_accuracy'] = {
                f'toward_top-{k}': a
                for k, a in zip(self.topk, toward_acc)
            }
            # compute accuracy
            character_acc = self.compute_accuracy(character_score_selected, character_gt_selected)
            if character_acc!=0:
                assert len(character_acc) == len(self.topk)
                losses['character_accuracy'] = {
                f'character_top-{k}': a
                for k, a in zip(self.topk, character_acc)
            }
            simplelight_acc = self.compute_accuracy(simplelight_score_selected, simplelight_gt_selected)
            if len(simplelight_acc)!=0:
                assert len(simplelight_acc) == len(self.topk)
                losses['simplelight_accuracy'] = {
                f'simplelight_top-{k}': a
                for k, a in zip(self.topk, simplelight_acc)
            }

            numsublight_acc = self.compute_accuracy(numsublight_score_selected, numsublight_gt_selected)
            if len(numsublight_acc)!=0:
                assert len(numsublight_acc) == len(self.topk)
                losses['numsublight_accuracy'] = {
                f'numsublight_top-{k}': a
                for k, a in zip(self.topk, numsublight_acc)
            }

             ####compute sublight
            subcolor1_acc = self.compute_accuracy(subcolor1_score_selected, subcolor1_gt_selected)
            if len(subcolor1_acc)!=0:
                assert len(subcolor1_acc) == len(self.topk)
                losses['subcolor1_accuracy'] = {
                f'subcolor1_top-{k}': a
                for k, a in zip(self.topk, subcolor1_acc)
            }
           
            subcolor2_acc = self.compute_accuracy(subcolor2_score_selected, subcolor2_gt_selected)
            if len(subcolor2_acc)!=0:
                assert len(subcolor2_acc) == len(self.topk)
                losses['subcolor2_accuracy'] = {
                f'subcolor2_top-{k}': a
                for k, a in zip(self.topk, subcolor2_acc)
            }
            
            subcolor3_acc = self.compute_accuracy(subcolor3_score_selected, subcolor3_gt_selected)
            if len(subcolor3_acc)!=0:
                assert len(subcolor3_acc) == len(self.topk)
                losses['subcolor3_accuracy'] = {
                f'subcolor3_top-{k}': a
                for k, a in zip(self.topk, subcolor3_acc)
            }
          
            subcolor4_acc = self.compute_accuracy(subcolor4_score_selected, subcolor4_gt_selected)
            if len(subcolor4_acc)!=0:
                assert len(subcolor4_acc) == len(self.topk)
                losses['subcolor4_accuracy'] = {
                f'subcolor4_top-{k}': a
                for k, a in zip(self.topk, subcolor4_acc)
            }
            
            subcolor5_acc = self.compute_accuracy(subcolor5_score_selected, subcolor5_gt_selected)
            if len(subcolor5_acc)!=0:
                assert len(subcolor5_acc) == len(self.topk)
                losses['subcolor5_accuracy'] = {
                f'subcolor5_top-{k}': a
                for k, a in zip(self.topk, subcolor5_acc)
            }

        if not self.freeze_color_head:
            losses['color_loss'] = {'color_loss': color_loss}
        if not self.freeze_shape_head:
            losses['shape_loss'] = {'shape_loss': shape_loss}
        if not self.freeze_toward_head:
            losses['toward_loss'] = {'toward_loss': toward_loss}
        if not self.freeze_character_head:
            losses['character_loss'] = {'character_loss': character_loss}
        if not self.freeze_simplelight_head:
            losses['simplelight_loss'] = {'simplelight_loss': simplelight_loss}
        if not self.freeze_numsublight_head:
            losses['numsublight_loss'] = {'numsublight_loss': numsublight_loss}
        if not self.freeze_sublightcolor_head:
            losses['sublightcolor_loss'] = {'sublightcolor_loss': subcolor_loss}

        return losses

    def train(self, mode=True):
        super(TrafficSubLightcomplexMultiClsHead1, self).train(mode)
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
