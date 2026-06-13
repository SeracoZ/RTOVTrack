# Copyright (c) Tencent Inc. All rights reserved.
import math
import copy
from typing import List, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.dist import get_dist_info
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.registry import MODELS
from mmyolo.models.dense_heads import YOLOv8HeadModule, YOLOv8Head
from mmyolo.models.utils import gt_instances_preprocess
from mmcv.cnn.bricks import build_norm_layer

from mmyolo.registry import TASK_UTILS
from mmdet.models.task_modules.samplers import CombinedSampler, InstanceBalancedPosSampler, RandomSampler
from mmdet.models.task_modules.assigners import MaxIoUAssigner, BboxOverlaps2D, AssignResult
from mmdet.models.task_modules.builder import build_sampler, build_assigner
TASK_UTILS.register_module()(CombinedSampler)
TASK_UTILS.register_module()(InstanceBalancedPosSampler)
TASK_UTILS.register_module()(RandomSampler)
TASK_UTILS.register_module()(MaxIoUAssigner)
TASK_UTILS.register_module()(BboxOverlaps2D)

from mmengine.structures import InstanceData
#from mmdet.engine.hooks.utils import build_sampler
#from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmyolo.models.losses import bbox_overlaps
from yolo_world.models.PCS import PromptConditionedSuppression



@MODELS.register_module()
class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """
    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class RepBNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """
    def __init__(self,
                 embed_dims: int,
                 num_guide_embeds: int,
                 norm_cfg: ConfigDict,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.conv = nn.Conv2d(embed_dims, num_guide_embeds, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        x = self.conv(x)
        return x


@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    """Head Module for YOLO-World

    Args:
        embed_dims (int): embed dim for text feautures and image features
        use_bn_head (bool): use batch normalization head
    """
    def __init__(self,
                 *args,
                 embed_dims: int,
                 use_bn_head: bool = False,
                 use_einsum: bool = True,
                 freeze_all: bool = False,
                 use_track_head: bool = False,
                 **kwargs) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        self.use_track_head = use_track_head
        super().__init__(*args, **kwargs)
        self.track_proj = nn.Linear(256, self.embed_dims)

        self.embed_fuser = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512)
        )


    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        """
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))
        """
        if self.use_track_head:
            for track_pred in self.track_preds:
                for m in track_pred.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.track_proj.weight, std=0.02)
        nn.init.constant_(self.track_proj.bias, 0)


    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        if self.use_track_head:
            self.track_preds = nn.ModuleList()

        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        track_out_channels = max(self.in_channels[0] // 2, 128)  ##seraco: todo:need edit

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=4 * self.reg_max,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            if self.use_track_head:
                self.track_preds.append(
                    nn.Sequential(
                        ConvModule(in_channels=self.in_channels[i],
                                   out_channels=track_out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg),
                        ConvModule(in_channels=track_out_channels,
                                   out_channels=track_out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   norm_cfg=self.norm_cfg,
                                   act_cfg=self.act_cfg),
                        nn.Conv2d(in_channels=track_out_channels,
                                  out_channels=512,
                                  kernel_size=1)))
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg,
                                      use_einsum=self.use_einsum))
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims,
                                    use_einsum=self.use_einsum))

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        #txt_masks = [txt_masks for _ in range(self.num_levels)]
        if self.use_track_head:
            return multi_apply(self.forward_single, img_feats, txt_feats, self.cls_preds, self.reg_preds, self.cls_contrasts, self.track_preds)
        else:
            return multi_apply(self.forward_single, img_feats, txt_feats, self.cls_preds, self.reg_preds, self.cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList, track_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)


        bbox_dist_preds = reg_pred(img_feat)
        if self.use_track_head:
            track_embed = track_pred(img_feat)
        else:
            track_embed = None
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            if self.use_track_head:
                return cls_logit, cls_embed, bbox_preds, bbox_dist_preds, track_embed
            else:
                return cls_logit, bbox_preds, bbox_dist_preds
        else:
            if self.use_track_head:
                return cls_logit, bbox_preds, cls_embed, track_embed
            else:
                return cls_logit, bbox_preds, cls_embed


@MODELS.register_module()
class RepYOLOWorldHeadModule(YOLOWorldHeadModule):
    def __init__(self,
                 *args,
                 embed_dims: int,
                 num_guide: int,
                 freeze_all: bool = False,
                 **kwargs) -> None:
        super().__init__(*args,
                         embed_dims=embed_dims,
                         use_bn_head=True,
                         use_einsum=False,
                         freeze_all=freeze_all,
                         **kwargs)

        # using rep head
        cls_contrasts = []
        for _ in range(self.num_levels):
            cls_contrasts.append(
                RepBNContrastiveHead(embed_dims=embed_dims,
                                     num_guide_embeds=num_guide,
                                     norm_cfg=self.norm_cfg))
        self.cls_contrasts = nn.ModuleList(cls_contrasts)

    def forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList, track_pred: nn.ModuleList) -> Tuple:
        """Forward features from the upstream network."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed)
        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds

    def forward(self, img_feats: Tuple[Tensor]) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        return multi_apply(self.forward_single, img_feats, self.cls_preds,
                           self.reg_preds, self.cls_contrasts)


@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    """YOLO-World Head
    """
    def __init__(self,
                 world_size=-1,
                 use_PCS=False,
                 use_aci=False,
                 global_threshold=0.3,
                 semi_threshold=0.4,
                 detail_threshold=0.2,
                 track_train_cfg=None,
                 track_test_cfg=None,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size
        category_path = 'data/tao/annotations/tao_val_lvis_v1_classes.json'
        self.pcs = PromptConditionedSuppression(topk=3, threshold=0.25, sim_threshold=0.2, category_path=category_path)
        self.use_PCS = use_PCS
        self.use_aci = use_aci
        self.global_threshold = global_threshold
        self.semi_threshold = semi_threshold
        self.detail_threshold = detail_threshold

        if track_test_cfg is not None:
            self.track_test_cfg = track_test_cfg
        #if track_train_cfg is not None:
            #self.track_train_cfg = track_train_cfg
            #self.init_track_assigner_sampler()

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get("assigner", None):
            self.track_assigner = build_assigner(self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_assigner = self.bbox_assigner
            self.track_share_assigner = True
        if self.track_train_cfg.get("sampler", None):
            self.track_sampler = build_sampler(self.track_train_cfg.sampler)
            self.track_share_sampler = False
        else:
            self.track_sampler = self.bbox_sampler
            self.track_share_sampler = True

    """YOLO World v8 head."""

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor, batch_data_samples: Union[list, dict], selected_id_list) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        outs = self(img_feats, txt_feats)
        # Fast version
        loss_inputs = outs + (batch_data_samples, selected_id_list)
        '''
        loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'],
                              batch_data_samples['instance_ids'],
                              batch_data_samples['selected_ids'])
        '''
        if self.head_module.use_track_head:
            losses, track_embeds, track_ids, track_lables = self.loss_by_feat(*loss_inputs)
            return losses, track_embeds, track_ids, track_lables
        else:
            losses = self.loss_by_feat(*loss_inputs)
            return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(img_feats, txt_feats, txt_masks)

        loss_inputs = outs + (txt_masks, batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           cfg=proposal_cfg)
        return losses, predictions

    def forward(self, img_feats: Tuple[Tensor], txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network."""
        return self.head_module(img_feats, txt_feats)

    def predict(self,
                img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        if self.use_aci:
            batch_img_metas = [
                {**data_sample.metainfo, 'active_cls_ids': data_sample.active_cls_ids}
                for data_sample in batch_data_samples
            ]
        else:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
        outs = self(img_feats, txt_feats) # xiao: yolov8: cls_logit, bbox_preds, todo: add cls_embeds
        if self.head_module.use_track_head:
            cls_scores, bbox_preds, embeds, track_preds = outs
        else:
            cls_scores, bbox_preds, embeds = outs
            track_preds = None  # placeholder, not used

        # outs以位置参数传递
        if self.use_aci:
            predictions, active_cls_ids= self.predict_by_feat(cls_scores, bbox_preds, embeds, track_preds, batch_img_metas=batch_img_metas, rescale=rescale)
            return predictions, active_cls_ids
        else:
            predictions = self.predict_by_feat(cls_scores, bbox_preds, embeds, track_preds, batch_img_metas=batch_img_metas, rescale=rescale)
            return predictions

    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_img_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        """Test function with test time augmentation."""
        raise NotImplementedError('aug_test is not implemented yet.')

    def loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            cls_preds: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            track_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            #batch_img_metas: Sequence[dict],
            #batch_instance_ids: Sequence[InstanceData],
            selected_ids_list = Sequence[Tensor],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(selected_ids_list)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, 0:1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        gt_labels_ori= gt_labels.clone()
        #gt_labels_flat = gt_labels.squeeze(-1)
        #select_id = torch.tensor(selected_ids_list, device=gt_labels.device)
        #eq = (select_id[None, :] == gt_labels_flat[:, :, None])
        #gt_labels = eq.float().argmax(-1).unsqueeze(-1)
        batch_size, N, _ = gt_labels.shape
        batch_gt_instance_ids = []
        for i in range(batch_size):
            labels = gt_labels[i, :, 0]  # (N,)
            select_id = torch.as_tensor(selected_ids_list[i], device=gt_labels.device)
            eq = (labels.unsqueeze(1) == select_id.unsqueeze(0))
            remapped = eq.float().argmax(dim=1)  # (N,)
            gt_labels[i, :, 0] = remapped
            gt_instance_ids = batch_gt_instances[i].instance_ids
            num_obj = gt_instance_ids.shape[0]
            pad = torch.zeros(N - num_obj, dtype=gt_instance_ids.dtype, device=gt_instance_ids.device)
            padded_instance_ids = torch.cat([gt_instance_ids, pad], dim=0)
            batch_gt_instance_ids.append(padded_instance_ids)

        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.stride_tensor[..., 0])

        #print(f'gt_labels: {gt_labels.shape}, gt_bboxes: {gt_bboxes.shape}, pad_bbox_flag: {pad_bbox_flag}')
        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        '''
        if batch_text_masks is not None:
            cls_weight = batch_text_masks.view(num_imgs, 1, -1).expand(
                -1, flatten_cls_preds.shape[1], -1).to(flatten_cls_preds)

            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            _loss_cls = (loss_cls * cls_weight).sum(dim=-1)
            loss_cls = _loss_cls.sum()
        else:
        '''
        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        #print(f'The value of fg_mask_pre_prior is {num_pos}')
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.loss_dfl(pred_dist_pos.reshape(
                -1, self.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size

        if self.head_module.use_track_head:
            #get the gt_ids ands instance track embeds
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #gt_ori_instances = batch_img_metas[0]['instances']
            #gt_ids = [instance['instance_id'] for instance in gt_ori_instances]
            #gt_ids = torch.tensor(gt_ids, device=device)
            gt_ids = batch_gt_instance_ids
            if gt_bboxes.numel() == 0 or gt_bboxes.size(1) == 0:
                track_embed = torch.empty((num_imgs, 0, 512), device=device)
                instance_ids = torch.empty((0,), device=device, dtype=gt_ids.dtype)
                class_labels = torch.empty((0,), device=device)
                loss_align = flatten_cls_preds.sum() * 0
            else:
                assigned_bboxes_4track = assigned_result['assigned_bboxes'] * self.stride_tensor # (N, 4)
                fg_mask = assigned_result['fg_mask_pre_prior']  # (N,)

                iou_matrix_list = []
                for b in range(num_imgs):
                    fg_inds = fg_mask[b].nonzero(as_tuple=True)[0]
                    ab_pos = assigned_bboxes_4track[b][fg_inds] # (num_pos_b, 4)
                    gt_b = gt_bboxes[b]  # (M_b, 4) or (M, 4)
                    iou_matrix = bbox_overlaps(
                        ab_pos.unsqueeze(1),  # (num_pos_b, 1, 4)
                        gt_b.unsqueeze(0),  # (1, M_b, 4)
                        bbox_format='xyxy',
                        iou_mode='iou'
                    )  # (num_pos_b, M_b)
                    iou_matrix_list.append(iou_matrix)

                max_iou_list = []
                gt_inds_list = []
                for iou_matrix in iou_matrix_list:
                    max_iou, gt_inds = iou_matrix.max(dim=1)
                    max_iou_list.append(max_iou)
                    gt_inds_list.append(gt_inds)

                instance_ids_list = []
                class_labels_list = []

                for b in range(num_imgs):
                    gt_ids_img = gt_ids[b]  # shape (M_b,)
                    gt_labels_img = gt_labels[b]  # shape (M_b,) or (M_b, 1)
                    gt_inds = gt_inds_list[b]  # shape (num_pos_b,)

                    instance_ids = gt_ids_img[gt_inds]  # (num_pos_b,)
                    # Handle shape for labels:
                    if gt_labels_img.dim() == 2 and gt_labels_img.shape[1] == 1:
                        gt_labels_img = gt_labels_img.squeeze(1)
                    #TODO: remap the labels
                    class_labels = gt_labels_img[gt_inds]  # (num_pos_b,)

                    instance_ids_list.append(instance_ids)
                    class_labels_list.append(class_labels)

                #gt_ids_img = gt_ids  # shape (M,)
                #instance_ids = gt_ids_img[gt_inds]  # (num_pos,)
                #class_labels = gt_labels[0].squeeze(1)[gt_inds]

                # Extract track instance embeddings from the track branch
                flatten_track_embeds = [
                    t_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 512)
                    for t_pred in track_preds
                ]
                flatten_track_embeds = torch.cat(flatten_track_embeds, dim=1)
                # Use the same fg_mask (for the first image) to select positive track embeddings

                track_instance_embeds = [
                    flatten_track_embeds[b][fg_mask[b]]  # (num_fg_b, 512)
                    for b in range(num_imgs)
                ]

                # get cls_embed features (from forward_single)
                flatten_cls_embeds = [
                    cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.head_module.embed_dims)
                    for cls_pred in cls_preds
                ]
                flatten_cls_embeds = torch.cat(flatten_cls_embeds, dim=1)

                # get positive cls embeddings at same fg locations
                cls_instance_embeds = [
                    flatten_cls_embeds[b][fg_mask[b]]  # (num_fg_b, 512)
                    for b in range(num_imgs)
                ]
                #combined_embed = torch.cat([track_instance_embeds, cls_instance_embeds], dim=-1)
                #fused_embed = self.head_module.embed_fuser(combined_embed)

                # ===== Sample at most one prediction per instance ID =====
                for b in range(num_imgs):
                    selected = []
                    unique_ids = batch_gt_instances[b].instance_ids
                    this_instance_ids = instance_ids_list[b]
                    if track_instance_embeds[b].size(0) == 0:
                        # nothing to select, skip or assign empty
                        track_instance_embeds[b] = torch.empty((0, 512), device=device)
                        cls_instance_embeds[b] = torch.empty((0, 512), device=device)
                        instance_ids_list[b] = torch.empty((0,), dtype=torch.long, device=device)
                        class_labels_list[b] = torch.empty((0,), dtype=torch.long, device=device)
                        continue
                    for gt_id in unique_ids:
                        indices = (this_instance_ids == gt_id).nonzero(as_tuple=True)[0]
                        if indices.numel() > 0:
                            rand_index = indices[torch.randint(0, indices.numel(), (1,)).item()]
                            selected.append(rand_index)
                    if len(selected) > 0:
                        selected = torch.stack(selected)
                        if selected.size(0) > 50:
                            perm = torch.randperm(selected.size(0), device=selected.device)
                            selected = selected[perm[:50]]
                        track_instance_embeds[b] = track_instance_embeds[b][selected]
                        cls_instance_embeds[b] = cls_instance_embeds[b][selected]
                        instance_ids_list[b] = instance_ids_list[b][selected]
                        class_labels_list[b] = class_labels_list[b][selected]
                    else:
                        # fallback if none are selected
                        track_instance_embeds = torch.empty((0, 512), device=device)
                        cls_instance_embeds = torch.empty((0, 512), device=device)
                        fused_embed = torch.empty((0, 512), device=device)
                        instance_ids = torch.empty((0,), dtype=torch.long, device=device)
                        class_labels = torch.empty((0,), dtype=torch.long, device=device)

                #track_instance_embeds = F.normalize(track_instance_embeds, dim=1)
                # InfoNCE loss (contrastive distillation)
                #with torch.no_grad():
                    #tsne_side_by_side(track_instance_embeds, cls_instance_embeds, fused_embed, instance_ids, label_type="Instance ID")


                loss_infonce = 0.0
                all_track_embed = []
                all_instance_ids = []
                all_class_labels = []
                for b in range(num_imgs):
                    t_embed = track_instance_embeds[b]
                    c_embed = cls_instance_embeds[b]
                    instance_ids_b = instance_ids_list[b]
                    class_labels_b = class_labels_list[b]
                    if t_embed.size(0) == 0 or c_embed.size(0) == 0:
                        # Pad with empty tensors so every batch index is present!
                        all_track_embed.append(torch.empty((0, 512), device=t_embed.device))
                        all_instance_ids.append(
                            torch.empty((0,), dtype=instance_ids_b.dtype, device=instance_ids_b.device))
                        all_class_labels.append(
                            torch.empty((0,), dtype=class_labels_b.dtype, device=class_labels_b.device))
                        continue
                    track_embed = F.normalize(t_embed, dim=1)
                    c_embed = F.normalize(c_embed, dim=1)
                    #temperature = 0.07
                    #logits = torch.matmul(fused_embed, c_embed.T) / temperature
                    #labels = torch.arange(logits.size(0), device=logits.device)
                    #loss_infonce = loss_infonce + F.cross_entropy(logits, labels)

                    all_track_embed.append(track_embed)
                    all_instance_ids.append(instance_ids_b)
                    all_class_labels.append(class_labels_b)

                '''
                num_valid = len(all_track_embed)
                if num_valid > 0:
                    #loss_infonce = loss_infonce / num_valid
                    fused_embed_out = torch.cat(all_track_embed, dim=0)
                    instance_ids_out = torch.cat(all_instance_ids, dim=0)
                    class_labels_out = torch.cat(all_class_labels, dim=0)
                else:
                    device = track_instance_embeds[0].device
                    #loss_infonce = torch.tensor(0.0, device=device, requires_grad=True)
                    fused_embed_out = torch.empty(0, track_instance_embeds[0].size(-1), device=device)
                    instance_ids_out = torch.empty(0, device=device)
                    class_labels_out = torch.empty(0, device=device)
                '''
                # If using distributed training and world_size > 1
                if 'world_size' in locals() and world_size > 1:
                    loss_infonce = loss_infonce * world_size
                '''
                # align feature dimensions (track: 256 → 512)
                #track_proj_embeds = self.head_module.track_proj(track_instance_embeds)
                # Normalize both embeddings
                # contrastive loss
                loss_align = F.mse_loss(track_proj_embeds,cls_instance_embeds.detach())  # detach cls to prevent interfering CLIP alignment
                '''
            return dict(loss_infonce=0), all_track_embed, all_instance_ids, all_class_labels
        else:
            return dict(loss_cls=loss_cls * num_imgs * world_size,
                        loss_bbox=loss_bbox * num_imgs * world_size,
                        loss_dfl=loss_dfl * num_imgs * world_size)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        embeds: Optional[List[Tensor]] = None,
                        track_preds: Optional[List[Tensor]] = None,
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        if embeds is None:
            with_embeds = False
        else:
            with_embeds = True

        cfg = self.track_test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        multi_label = False
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # embedding
        if with_embeds:
            flatten_cls_embeds = [
                embed_map.permute(0, 2, 3, 1).reshape(num_imgs, -1, 512)  # xiao: text_dim=512
                for embed_map in embeds
            ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)
        if with_embeds:
            flatten_cls_embeds = torch.cat(flatten_cls_embeds, dim=1)

        if self.head_module.use_track_head:
            flatten_track_embeds = [
                t_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 512)
                for t_pred in track_preds
            ]
            flatten_track_embeds = torch.cat(flatten_track_embeds, dim=1)

        fused_embeds = []
        for img_idx in range(num_imgs):
            track_feat = flatten_track_embeds[img_idx]  # (N, 512)
            cls_feat = flatten_cls_embeds[img_idx]  # (N, 512)
            combined = torch.cat([track_feat, cls_feat], dim=1)  # (N, 1024)
            fused = self.head_module.embed_fuser(combined)  # (N, 512)
            fused_embeds.append(fused)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness, cls_embeds, track_embeds, img_meta) in zip(
                flatten_decoded_bboxes, flatten_cls_scores, flatten_objectness,
                flatten_cls_embeds, flatten_track_embeds, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                empty_results.cls_embeds = cls_embeds
                empty_results.track_embeds = track_embeds
                results_list.append(empty_results)
                continue

            # ===================== PCS + DCA Filtering =====================
            if self.use_PCS:
                # Define indices explicitly based on your architecture
                idx_detail_end = 16800
                idx_semi_global_end = idx_detail_end + 4200
                idx_global_end = idx_semi_global_end + 1050

                # Slice the scores tensor explicitly by pyramid levels
                detail_scores = scores[:idx_detail_end, :]  # (16800, Num_classes)
                semi_global_scores = scores[idx_detail_end:idx_semi_global_end, :]  # (4200, Num_classes)
                global_scores = scores[idx_semi_global_end:idx_global_end, :]  # (1050, Num_classes)

                # Compute maximum scores per class for each level efficiently
                detail_class_scores = detail_scores.max(dim=0).values
                semi_global_class_scores = semi_global_scores.max(dim=0).values
                global_class_scores = global_scores.max(dim=0).values

                #threshold = 0.2
                active_detail = (detail_class_scores > self.detail_threshold).nonzero(as_tuple=True)[0]
                active_semi_global = (semi_global_class_scores > self.semi_threshold).nonzero(as_tuple=True)[0]
                active_global = (global_class_scores > self.global_threshold).nonzero(as_tuple=True)[0]

                active_cls_ids = torch.unique(torch.cat([active_detail, active_semi_global, active_global])).tolist()
                if self.use_aci:
                    last_active_cls_ids = batch_img_metas[0]['active_cls_ids']
                    combined_cls_ids = list(set(active_cls_ids) | set(last_active_cls_ids))
                else:
                    combined_cls_ids = active_cls_ids

                # Apply PCS to suppress irrelevant region predictions
                # Step 2: Get per-box predicted class
                max_scores, pred_labels = scores.max(dim=1)  # shape: (N,)
                # Step 3: Class-aware PCS filtering
                keep_mask = self.pcs.class_aware_keep_mask(pred_labels, max_scores, combined_cls_ids)
                # Concatenate masks for full image
                bboxes = bboxes[keep_mask]
                scores = scores[keep_mask]
                #labels = labels[keep_mask]
                cls_embeds = cls_embeds[keep_mask]
                track_embeds = track_embeds[keep_mask]
                #obj_embeds = obj_embeds[keep_mask]
            # ==============================================================

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            cls_embeds = cls_embeds[keep_idxs]
            track_embeds = track_embeds[keep_idxs]

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs],
                                   cls_embeds=cls_embeds,
                                   track_embeds=track_embeds)

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        if self.use_aci:
            return results_list, active_cls_ids
        else:
            return results_list
