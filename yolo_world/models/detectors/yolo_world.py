# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS

import copy
import torch.nn.functional as F
from yolo_world.models.losses import TrackHeadModule
from mmyolo.models.utils import gt_instances_preprocess
from ..utils.list_LVIS import Frequency_list_70, novel_class
from third_party.mmyolo.tests.test_models.test_backbone.utils import is_block


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

        self.max_pad_len = 250
        self.distribution_based_sampling = True
        self.frequency = torch.tensor(Frequency_list_70, dtype=torch.float32, device='cpu')
        self.novel_cls_cpu = novel_class
        self.track_loss_module = TrackHeadModule(margin=0.2, temperature=0.1, softmax_temp=-1)

    def compute_triplet_loss(self, key_embeds, ref_embeds, track_ids, ref_ids, margin, device):
        """
        Compute triplet loss given embeddings and track IDs.
        Args:
            key_embeds (Tensor): (N, D)
            ref_embeds (Tensor): (M, D)
            track_ids (Tensor): (N,)
            ref_ids (Tensor): (M,)
            margin (float): margin for triplet loss
            device (str): device
        Returns:
            loss (Tensor): scalar loss
        """
        if (
                key_embeds.size(0) == 0 or ref_embeds.size(0) == 0 or
                track_ids.size(0) == 0 or ref_ids.size(0) == 0
        ):
            return torch.tensor(0.0, requires_grad=True, device=device)
        N = key_embeds.size(0)
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_triplets = 0

        for i in range(N):
            anchor_embed = key_embeds[i]
            anchor_id = track_ids[i]

            # Positive indices: same ID
            pos_mask = (ref_ids == anchor_id)
            neg_mask = (ref_ids != anchor_id)

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_embed = ref_embeds[pos_mask][0]  # Pick first (or sample one)
            neg_embeds = ref_embeds[neg_mask]

            # Hardest negative: farthest
            d_neg = F.pairwise_distance(anchor_embed.unsqueeze(0), neg_embeds, p=2)
            hardest_neg = neg_embeds[d_neg.argmax()]

            d_ap = F.pairwise_distance(anchor_embed.unsqueeze(0), pos_embed.unsqueeze(0), p=2)
            d_an = F.pairwise_distance(anchor_embed.unsqueeze(0), hardest_neg.unsqueeze(0), p=2)

            triplet = F.relu(d_ap - d_an + margin)
            total_loss = total_loss + triplet
            valid_triplets += 1

        if valid_triplets > 0:
            return total_loss / valid_triplets
        else:
            return torch.tensor(0.0, requires_grad=True, device=device)

    def loss(self,
             batch_inputs: Tensor,
             batch_data_samples: SampleList,) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        N = int(len(batch_data_samples)/2)
        key_batch_inputs = batch_inputs[:N]
        ref_batch_inputs = batch_inputs[N:]
        key_data_samples = batch_data_samples[:N]
        ref_data_samples = batch_data_samples[N:]
        '''
        #####################################
        import cv2
        import os
        import numpy as np
        from pathlib import Path
        # key_batch_inputs shape is (1,3,W,H), lables is a ndarray, bboxes is a tensor has shape of (N,4)
        image_path = key_data_samples['img_metas']['img_path']
        image_shape = key_data_samples['img_metas']['img_shape']
        key_instance_ids = key_data_samples['instance_ids'].cpu().numpy()
        ref_instance_ids = ref_data_samples['instance_ids'].cpu().numpy()
        key_bboxes = key_data_samples['bboxes']
        ref_bboxes = ref_data_samples['bboxes']

        img = 255 * np.ones((800, 1344, 3), dtype=np.uint8)
        for box in key_bboxes:
            if hasattr(box, 'tensor'):  # If it's a mmcv Box type
                box = box.tensor[0]
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img, c1, c2, (144, 238, 144), thickness=1)

        Path('./data_vis_in_train').mkdir(parents=True, exist_ok=True)
        file_base = os.path.basename(image_path).split('.')[0]
        out_path = f'./data_vis_in_train/{file_base}.jpg'
        cv2.imwrite(out_path, img)

        img_ref = 255 * np.ones((800, 1344, 3), dtype=np.uint8)
        for box in ref_bboxes:
            if hasattr(box, 'tensor'):  # If it's a mmcv Box type
                box = box.tensor[0]
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img_ref, c1, c2, (144, 238, 144), thickness=1)

        cv2.imwrite(f'./data_vis_in_train/{file_base}_ref.jpg', img_ref)
        #print(f"Saved visualization: {out_path}")
        ######################################
        '''
        key_img_feats, key_txt_feats, selected_id_list = self.extract_feat(key_batch_inputs, key_data_samples)
        #key_data_samples['selected_id_list'] = selected_id_list
        losses = {}
        if self.bbox_head.head_module.use_track_head:
            ref_img_feats, ref_txt_feats, ref_selected_list = self.extract_feat(ref_batch_inputs, ref_data_samples)
            gt_match_indices = []
            for i in range(len(key_data_samples)):
                key_id = key_data_samples[i].instance_ids
                ref_id = ref_data_samples[i].instance_ids
                union = torch.tensor(
                    list(set(key_id.tolist()) | set(ref_id.tolist())),
                    device=key_id.device,
                    dtype=key_id.dtype,
                )
                gt_match_indices.append(union)
            _, key_track_embed, key_ids, key_labels = self.bbox_head.loss(key_img_feats, key_txt_feats, key_data_samples, selected_id_list)
            with torch.no_grad():
                _, ref_track_embed, ref_ids, ref_lables = self.bbox_head.loss(ref_img_feats, ref_txt_feats, ref_data_samples, ref_selected_list)
            match_feats = self.track_loss_module.match(
                key_track_embed, ref_track_embed, key_ids, ref_ids
            )

            def ensure_not_empty(tensor, pad_value=-1, device=None):
                if tensor.numel() == 0:
                    return torch.tensor([pad_value], dtype=torch.long, device=device)
                return tensor

            remapped_key_ids = []
            remapped_ref_ids = []
            remapped_gt_match_indices = []
            for key_id, ref_id, gt_match in zip(key_ids, ref_ids, gt_match_indices):
                all_ids = torch.cat([key_id, ref_id, gt_match]).unique()
                id2idx = {id_val.item(): i for i, id_val in enumerate(all_ids)}
                key_id_remap = torch.tensor([id2idx[x.item()] for x in key_id], device=key_id.device, dtype=torch.long)
                ref_id_remap = torch.tensor([id2idx[x.item()] for x in ref_id], device=ref_id.device, dtype=torch.long)
                gt_match_remap = torch.tensor([id2idx[x.item()] for x in gt_match], device=gt_match.device, dtype=torch.long)
                key_id_remap = ensure_not_empty(key_id_remap, pad_value=-1, device=key_id.device)
                ref_id_remap = ensure_not_empty(ref_id_remap, pad_value=-1, device=ref_id.device)
                gt_match_remap = ensure_not_empty(gt_match_remap, pad_value=-1, device=gt_match.device)
                remapped_key_ids.append(key_id_remap)
                remapped_ref_ids.append(ref_id_remap)
                remapped_gt_match_indices.append(gt_match_remap)

            asso_targets = self.track_loss_module.get_track_targets(
                remapped_gt_match_indices, remapped_key_ids, remapped_ref_ids
            )
            loss_track = self.track_loss_module.loss(*match_feats, *asso_targets)

            loss_ins_track = loss_track['loss_track']
            loss_track_aux = loss_track['loss_track_aux']
            losses['loss_track'] = loss_ins_track
            losses['loss_track_aux'] = loss_track_aux
            return losses
        else:
            losses = self.bbox_head.loss(key_img_feats, key_txt_feats, key_data_samples)
            return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(
            batch_inputs, batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        if self.bbox_head.use_aci:
            results_list, active_cls_dis = self.bbox_head.predict(img_feats, txt_feats, batch_data_samples, rescale=rescale)
            batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
            return batch_data_samples, active_cls_dis
        else:
            results_list= self.bbox_head.predict(img_feats, txt_feats, batch_data_samples, rescale=rescale)
            batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
            return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        out = self.backbone.forward_text(texts)
        if isinstance(out, tuple):
            out = out[0]
        self.text_feats = out.detach()
        #self.text_feats = self.backbone.forward_text(texts)
        #print(self.text_feats.shape)
        #self.text_feats, None = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""



        if self.training:
            num_imgs = len(batch_data_samples)
            #num_imgs = 1
            #bboxes_labels_list = [sample['bboxes_labels'] for sample in batch_data_samples]
            #bboxes_labels_clone = batch_data_samples['bboxes_labels'].clone()
            gt_info = gt_instances_preprocess(batch_data_samples, num_imgs)
            select_id_list = []
            for i in range(num_imgs):
                gt_labels = gt_info[i, :, 0].long()  # Get labels for image i (shape: 127,)
                # Remove padded zeros (assuming 0 means pad—modify if your pad value is different)
                gt_labels = gt_labels[gt_labels != 0]
                cls_num = len(torch.unique(gt_labels))
                is_first = True  # set to False if needed by your logic
                extra_labels = None
                select_id, extra_labels = self.get_select_id(cls_num, gt_labels, extra_labels, is_first)
                select_id_list.append(select_id)
        else:
            select_id, extra_labels = None, None

        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, (txt_feats,
                        txt_masks) = self.backbone(batch_inputs, texts)
        if self.training:
            cropped_txt_feats = []
            for select_id in select_id_list:  # select_id is a list/tensor of indices for one image
                # Ensure select_id is a tensor
                if not torch.is_tensor(select_id):
                    select_id = torch.tensor(select_id, device=txt_feats.device)
                # Crop the feature: shape (1, num_selected, 512)
                txt_feat_img = txt_feats[:, select_id, :]  # (1, num_sel, 512)
                cropped_txt_feats.append(txt_feat_img)
            #txt_feats = txt_feats[:, select_id, :]
            #txt_feats = txt_feats.to(batch_inputs.device)
        if self.with_neck:
            if self.mm_neck:
                if self.training:
                    per_img_outputs = []
                    batch_size = img_feats[0].shape[0]
                    for i in range(batch_size):
                        per_img_feats = [f[i:i + 1] for f in img_feats]  # Each is (1, C, H, W)
                        per_txt_feat = cropped_txt_feats[i]  # (1, N_select, 512) or similar
                        out = self.neck(per_img_feats, per_txt_feat)  # Returns a tuple/list (len=3)
                        per_img_outputs.append(out)
                    stacked_outputs = []
                    num_levels = len(per_img_outputs[0])
                    for lvl in range(num_levels):
                        # For each level, gather the outputs from each image and stack
                        stacked = torch.cat([out[lvl] for out in per_img_outputs], dim=0)
                        stacked_outputs.append(stacked)
                    img_feats = tuple(stacked_outputs)
                    txt_feats = torch.cat(cropped_txt_feats, dim=0)
                else:
                    img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)


            if self.training:
                return img_feats, txt_feats, select_id_list
            else:
                return img_feats, txt_feats

    def _distribution_based_sampling(self, pad_len, uniq_labels=None):
        frequency = self.frequency.clone()
        frequency[uniq_labels] = 0
        extra_labels = torch.multinomial(frequency, pad_len)
        extra_labels = extra_labels[torch.isin(extra_labels, self.novel_cls_cpu, invert=True)]
        extra_labels = extra_labels[torch.randperm(len(extra_labels))]
        return extra_labels

    def get_select_id(self, cls_num, labels_list, extra_labels, is_first):
        max_pad_len = max(cls_num, self.max_pad_len)
        # get input categories
        uniq_labels = torch.unique(labels_list).to("cpu")
        if is_first:  # first frame detection
            if len(uniq_labels) < max_pad_len:
                pad_len = max_pad_len - len(uniq_labels)
                if self.distribution_based_sampling:  # Sample negative categories based on the distribution.
                    extra_labels = self._distribution_based_sampling(pad_len, uniq_labels=uniq_labels)
                else:
                    extra_list = torch.tensor([i for i in self.all_ids if i not in uniq_labels])
                    extra_labels = extra_list[torch.randperm(len(extra_list))][:pad_len]
                select_id = extra_labels.tolist() + uniq_labels.tolist()
            else:
                select_id = uniq_labels.tolist()
                extra_labels = torch.LongTensor([])
        else:  # subsequent frame tracking
            extra_label_notin = torch.isin(extra_labels, uniq_labels, invert=True)
            extra_labels_cur = extra_labels[extra_label_notin]
            select_id = uniq_labels.tolist() + extra_labels_cur.tolist()
            if len(select_id) < max_pad_len:
                sampled_labels = self._distribution_based_sampling(max_pad_len - len(select_id),
                                                                   uniq_labels=torch.tensor(select_id))
                if extra_labels is not None:
                    extra_labels = torch.cat([extra_labels_cur, sampled_labels])
                else:
                    extra_labels = sampled_labels
                select_id = uniq_labels.tolist() + extra_labels.tolist()
            elif len(select_id) > max_pad_len:
                select_id = select_id[:max_pad_len]
        #print(f'select id len is {len(select_id)} and extra_labels len is {len(extra_labels)} and is {is_first}')
        return select_id, extra_labels

    @staticmethod
    def set_bn_eval_except_track_head(model):
        """
        Set all BatchNorm layers except those in track_preds (track head) to eval mode.
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                # If not in the track_preds branch, set to eval
                if 'track_preds' not in name:
                    module.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.set_bn_eval_except_track_head(self)
            if hasattr(self.bbox_head.head_module, 'track_preds'):
                self.bbox_head.head_module.track_preds.train()

@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
