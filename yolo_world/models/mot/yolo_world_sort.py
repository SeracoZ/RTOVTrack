"""
Author: Xiao
Licensed: Apache-2.0 License
"""

import copy
import os
import pickle
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmdet.models.mot.base import BaseMOTModel
from mmdet.structures import TrackSampleList
from mmdet.utils import OptConfigType, OptMultiConfig
from mmengine.structures import InstanceData
from scipy.special.cython_special import log_expit
from torch import Tensor
# xiao: use registry of mmyolo
from mmyolo.registry import MODELS


@MODELS.register_module()
class YOLOWorldSort(BaseMOTModel):
    """Simple online and realtime tracking with a IoU association metric.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        reid (dict): Configuration of reid. Defaults to None
        tracker (dict): Configuration of tracker. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`TrackDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or list[dict]): Configuration of initialization.
            Defaults to None.
    """

    def __init__(self,
                 detector: Optional[dict] = None,
                 reid: Optional[dict] = None,
                 tracker: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if reid is not None:
            self.reid = MODELS.build(reid)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)

        self.preprocess_cfg = data_preprocessor

        self.base_active_cls_ids = []
        self.cls_id_memory = {}
        self.max_age = 10

    def loss(self, inputs: Tensor, data_samples: TrackSampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'  # (1, 1, 3, 640, 1088)
        track_data_sample = data_samples[0]
        track_ref_sample = data_samples[1]
        # 将inputs分离为key_frame和ref_frame
        key_inputs = inputs[0] # From (1, 3, 800, 1344) to (1, 1, 3, 800, 1344)
        ref_inputs = inputs[1]
        img_data_sample = track_data_sample[0]
        img_ref_sample = track_ref_sample[0]
        losses = self.detector.loss(key_inputs, ref_inputs, [img_data_sample], [img_ref_sample])
        return losses


    def predict(self,
                inputs: Tensor,
                data_samples: TrackSampleList,
                rescale: bool = True,
                **kwargs) -> TrackSampleList:

        """Predict results from a video and data samples with post-processing.

        Args:
            inputs (Tensor): of shape (N, T, C, H, W) encoding
                input images. The N denotes batch size.
                The T denotes the number of key frames
                and reference frames.
            data_samples (list[:obj:`TrackDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance`.
            rescale (bool, Optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to True.

        Returns:
            TrackSampleList: List[TrackDataSample]
            Tracking results of the input videos.
            Each DetDataSample usually contains ``pred_track_instances``.
        """
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).' # (1, 1, 3, 640, 1088)
        #assert inputs.size(0) == 1, \
            #'SORT/DeepSORT inference only support ' \
            #'1 batch size per gpu for now.'

        #assert len(data_samples) == 1, \
            #'SORT/DeepSORT inference only support ' \
            #'1 batch size per gpu for now.'
        key_inputs = inputs[0].unsqueeze(0)  # From (1, 3, 800, 1344) to (1, 1, 3, 800, 1344)
        ref_inputs = inputs[1].unsqueeze(0)

        track_data_sample = data_samples[0]
        ref_data_sample = data_samples[1]

        video_len = len(track_data_sample) # xiao: sort推理单帧处理，预留以适配video clip

        if track_data_sample[0].frame_id == 0:
            self.tracker.reset()
            self.base_active_cls_ids = []
            self.cls_id_memory = {}

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            if self.detector.bbox_head.use_aci:
                img_data_sample.active_cls_ids = self.base_active_cls_ids
            ref_data_sample = ref_data_sample[0]
            single_img = key_inputs[:, frame_id].contiguous() # (1, 3, 640, 1088)
            ref_img = ref_inputs[:, frame_id].contiguous()
            # det_results List[DetDataSample]
            if self.detector.bbox_head.use_aci:
                det_results, active_cls_ids = self.detector.predict(single_img, [img_data_sample])
                #########################class memory#######################
                for cid in active_cls_ids:
                    self.cls_id_memory[cid] = self.cls_id_memory.get(cid, 0) + 1
                # Decay memory: subtract from unseen IDs
                for cid in list(self.cls_id_memory.keys()):
                    if cid not in active_cls_ids:
                        self.cls_id_memory[cid] -= 1
                        if self.cls_id_memory[cid] <= 0:
                            del self.cls_id_memory[cid]

                # Final active list is just the keys still alive
                self.base_active_cls_ids = list(self.cls_id_memory.keys())
                #print(f"[Frame {frame_id}] New: {active_cls_ids}, Memory: {self.base_active_cls_ids}")
                #print(f"[Frame {frame_id}] Class Memory State: {self.cls_id_memory}")
                #########################class memory#######################
            else:
                det_results = self.detector.predict(single_img, [img_data_sample])
            #ref_results = self.detector.predict(ref_img, [ref_data_sample])

            assert len(det_results) == 1, 'Batch inference is not supported.'

            pred_track_instances = self.tracker.track(
                model=self,   # deepsort, 目的是提供reid模型
                img=single_img,
                feats=None,
                data_sample=det_results[0],
                data_preprocessor=self.preprocess_cfg,
                rescale=rescale,
                **kwargs)

            img_data_sample.pred_track_instances = pred_track_instances

        #print('pred_track_id: ', img_data_sample.pred_track_instances.instances_id)
        return [track_data_sample]
