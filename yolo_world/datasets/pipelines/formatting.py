# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from OVTrack (https://github.com/SysCV/ovtrack)
# ------------------------------------------------------------------------
import torch
import numpy as np
import torch

from mmyolo.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from .data_container import DataContainer as DC
from mmdet.structures import ReIDDataSample
from typing import Sequence
from .data_container import DataContainer

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) :
        #and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

@TRANSFORMS.register_module()
class SeqDefaultFormatBundle(BaseTransform):
    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def transform(self, results):
        outs = []
        for _results in results:
            _results = self._transform(_results)
            if "gt_match_indices" in _results:
                _results["gt_match_indices"] = torch.tensor(_results["gt_match_indices"])
            #_results["img"] = torch.from_numpy(_results["img"].transpose(2, 0, 1))
            outs.append(_results)
        return outs

    def _transform(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with \
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            results = self._add_default_meta_keys(results)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(
                to_tensor(img), padding_value=self.pad_val['img'], stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_bboxes_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(
                results['gt_masks'],
                padding_value=self.pad_val['masks'],
                cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]),
                padding_value=self.pad_val['seg'],
                stack=True)
        return results

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']
        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results

@TRANSFORMS.register_module(force=True)
class SeqCollect(BaseTransform):
    def __init__(
        self,
        keys,
        ref_prefix="ref",
        meta_keys=(
            "img_path",
            "ori_filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "img_norm_cfg",
            "frame_id",
        ),
    ):
        self.keys = keys
        self.ref_prefix = ref_prefix
        self.meta_keys = meta_keys


    def transform(self, results):
        # results is a list of dicts
        outs = []
        for _results in results:
            data = {}
            img_meta = {}
            for key in self.keys:
                if key in _results:
                    data[key] = _results[key]
            for key in self.meta_keys:
                if key in _results:
                    img_meta[key] = _results[key]
            data["img_meta"] = DC(img_meta, cpu_only=True)
            outs.append(data)

        # flatten keys like img_0, img_1, etc.
        merged = {}
        for i, sample in enumerate(outs):
            for k, v in sample.items():
                merged[f"{k}_{i}"] = v
        return merged

    def _match_gts(self, inds, ref_inds):
        match_indices = np.array([ref_inds.index(i) if i in ref_inds else -1 for i in inds])
        ref_match_indices = np.array([inds.index(i) if i in inds else -1 for i in ref_inds])
        return match_indices, ref_match_indices

@TRANSFORMS.register_module()
class PackPairTrackInputs(BaseTransform):  # <- no need to inherit from PackReIDInputs
    def __init__(self, meta_keys: Sequence[str] = ()):
        self.meta_keys = (
            'img_path', 'ori_shape', 'img_shape', 'scale', 'scale_factor'
        )
        if meta_keys:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys,)
            self.meta_keys += tuple(meta_keys)

    def transform(self, results: dict) -> dict:
        imgs = []
        labels = []
        data_samples = []

        for i in range(2):
            img = results[f'img_{i}']
            if isinstance(img, DataContainer):
                img = img.data  # get the tensor
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(img)
            imgs.append(img.unsqueeze(0))  # (1, C, H, W) for each

            labels = results[f'gt_bboxes_labels_{i}']
            labels = labels.data

            data_sample = ReIDDataSample()
            data_sample.set_gt_label(labels)
            img_meta = results[f'img_meta_{i}']
            if hasattr(img_meta, 'data'):
                img_meta = img_meta.data
            meta_info = {key: img_meta.get(key, None)
                         for key in self.meta_keys if key in img_meta}
            data_sample.set_metainfo(meta_info)
            data_samples.append(data_sample)

        # Stack along batch dimension (N=2, C, H, W)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.tensor(labels)

        return {
            'inputs': imgs,             # (2, C, H, W) tensor
            'labels': labels,           # (2,) tensor
            'data_samples': data_samples
        }