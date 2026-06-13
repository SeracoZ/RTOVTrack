# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.structures import BaseDataElement

from mmyolo.registry import MODELS
from mmdet.models.data_preprocessors import TrackDataPreprocessor
from mmengine.structures import InstanceData


CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str,
                 None]


class SimpleInstance:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

@MODELS.register_module()
class YOLOWDetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolow_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

        self.track_preprocessor = TrackDataPreprocessor(
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
            bgr_to_rgb=True,
            pad_size_divisor=32
        )

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return self.track_preprocessor.forward(data, training)
            #return super().forward(data, training)

        data = self.cast_data(data)
        key_inputs, key_data_samples = data['key_inputs'], data['key_data_samples']
        ref_inputs, ref_data_samples = data['ref_inputs'], data['ref_data_samples']
        #inputs, data_samples = data['inputs'], data['data_samples']
        #assert isinstance(data['data_samples'], dict)

        for inputs in [key_inputs, ref_inputs]:
            if self._channel_conversion and inputs.shape[1] == 3:
                inputs = inputs[:, [2, 1, 0], ...]
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std

        def process_samples(data_samples):
            new_data_samples = []
            for sample in data_samples:
                # Separate annotation fields and meta fields
                data_fields = {}
                meta_fields = {}
                for k, v in sample.items():
                    if k in ('bboxes', 'labels'):
                        data_fields[k] = v
                    else:
                        meta_fields[k] = v
                # Create InstanceData
                instance = InstanceData(metainfo=meta_fields)
                for k, v in data_fields.items():
                    setattr(instance, k, v)
                new_data_samples.append(instance)
            return new_data_samples

        key_data_samples = process_samples(key_data_samples)
        ref_data_samples = process_samples(ref_data_samples)

        all_inputs = torch.cat([key_inputs, ref_inputs], dim=0)  # if both are tensors, batch axis 0
        all_data_samples = key_data_samples + ref_data_samples

        return {'inputs': all_inputs, 'data_samples': all_data_samples}
