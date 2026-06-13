# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import numpy as np
import torch
from mmengine.dataset import COLLATE_FUNCTIONS


@COLLATE_FUNCTIONS.register_module()
def yolow_collate(data_batch: Sequence,
                  use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """

    batch_key_imgs, batch_ref_imgs = [], []
    batch_key_data_samples, batch_ref_data_samples = [], []

    for i, sample in enumerate(data_batch):
        datasamples = sample['gt_instances']   # [key, ref]
        imgs = sample['imgs']                  # [key_img, ref_img]
        for idx, (frame, img) in enumerate(zip(datasamples, imgs)):
            # Standardize bboxes/labels/instance_ids
            bboxes = frame['bboxes'].tensor if hasattr(frame['bboxes'], 'tensor') else frame['bboxes']
            labels = frame['labels'].data
            #labels = torch.from_numpy(frame['labels']) if isinstance(frame['labels'], np.ndarray) else frame['labels']
            #labels = labels.to(bboxes.device)
            instance_ids = frame.get('instance_ids', torch.full((len(labels),), -1, dtype=torch.long, device=bboxes.device))
            if hasattr(instance_ids, 'tensor'):
                instance_ids = instance_ids.tensor
            instance_ids = instance_ids.to(bboxes.device)
            batch_idx = labels.new_full((len(labels), 1), i)
            bboxes_labels = torch.cat((batch_idx, labels[:, None], bboxes), dim=1)
            data_dict = {
                'bboxes_labels': bboxes_labels,
                'instance_ids': instance_ids,
                'img_metas': frame['img_metas'],
                'img_info': frame['img_info'],
                'ann_info': frame['ann_info'],
                'bboxes': frame['bboxes'],
                'labels': frame['labels'].data,
            }
            if idx == 0:
                batch_key_imgs.append(img)
                batch_key_data_samples.append(data_dict)
            else:
                batch_ref_imgs.append(img)
                batch_ref_data_samples.append(data_dict)

    collated_results = {
        'key_inputs': torch.stack(batch_key_imgs, dim=0),
        'key_data_samples': batch_key_data_samples,
        'ref_inputs': torch.stack(batch_ref_imgs, dim=0),
        'ref_data_samples': batch_ref_data_samples,
    }

    return collated_results