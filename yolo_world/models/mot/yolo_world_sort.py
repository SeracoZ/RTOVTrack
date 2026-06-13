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

#from yolo_world import OVTracker


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
                 motion: Optional[dict] = None,
                 data_preprocessor: OptConfigType = None,
                 visual: bool = False,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor, init_cfg)

        if detector is not None:
            self.detector = MODELS.build(detector)

        if reid is not None:
            self.reid = MODELS.build(reid)

        if tracker is not None:
            self.tracker = MODELS.build(tracker)
        if motion is not None:
            self.motion = MODELS.build(motion)

        self.preprocess_cfg = data_preprocessor

        self.base_active_cls_ids = []
        self.cls_id_memory = {}
        self.max_age = 10
        self.visual = visual

    def loss(self, inputs: Tensor, data_samples: TrackSampleList,
             **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        '''
        assert inputs.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'  # (1, 1, 3, 640, 1088)
        track_data_sample = data_samples[0]
        track_ref_sample = data_samples[1]
        # 将inputs分离为key_frame和ref_frame
        key_inputs = inputs[0] # From (1, 3, 800, 1344) to (1, 1, 3, 800, 1344)
        ref_inputs = inputs[1]
        img_data_sample = track_data_sample[0]
        img_ref_sample = track_ref_sample[0]
        losses = self.detector.loss(key_inputs, ref_inputs, [img_data_sample], [img_ref_sample])
        '''
        losses = self.detector.loss(inputs, data_samples)
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
        #ref_inputs = inputs[1].unsqueeze(0)

        track_data_sample = data_samples[0]
        #ref_data_sample = data_samples[1]

        video_len = len(track_data_sample) # xiao: sort推理单帧处理，预留以适配video clip

        if track_data_sample[0].frame_id == 0:
            self.tracker.reset()
            self.base_active_cls_ids = []
            self.cls_id_memory = {}

        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            if self.detector.bbox_head.use_aci:
                img_data_sample.active_cls_ids = self.base_active_cls_ids
            single_img = key_inputs[:, frame_id].contiguous() # (1, 3, 640, 1088)
            if self.detector.bbox_head.use_aci:
                det_results, active_cls_ids = self.detector.predict(single_img, [img_data_sample])
                #########################class memory#######################
                for cid in active_cls_ids:
                    boost = 1.0
                    self.cls_id_memory[cid] = self.cls_id_memory.get(cid, 0.0) * 0.9 + boost

                for cid in list(self.cls_id_memory.keys()):
                    if cid not in active_cls_ids:
                        self.cls_id_memory[cid] *= 0.9
                    if self.cls_id_memory[cid] < 0.3:
                        del self.cls_id_memory[cid]

                self.base_active_cls_ids = [cid for cid, score in self.cls_id_memory.items() if score > 0.3]
                #print(f"[Frame {frame_id}] New: {active_cls_ids}, Memory: {self.base_active_cls_ids}")
                #print(f"[Frame {frame_id}] Class Memory State: {self.cls_id_memory}")
                #########################class memory#######################
            else:
                det_results = self.detector.predict(single_img, [img_data_sample])
            #ref_results = self.detector.predict(ref_img, [ref_data_sample])

            assert len(det_results) == 1, 'Batch inference is not supported.'

            if self.tracker.__class__.__name__ == "SortEmbedTracker":
                pred_track_instances = self.tracker.track(
                    model=self,   # deepsort, 目的是提供reid模型
                    img=single_img,
                    feats=None,
                    data_sample=det_results[0],
                    data_preprocessor=self.preprocess_cfg,
                    rescale=rescale,
                    **kwargs)
            else:
                instances = det_results[0].pred_instances
                bboxes = instances.bboxes
                det_labels = instances.labels
                track_feats = instances.track_embeds
                cem_feats = instances.cls_embeds
                scores = instances.scores
                det_bboxes = torch.cat([bboxes, scores.unsqueeze(1)], dim=1)
                self.method = 'ovtrack-teta'
                if self.tracker.__class__.__name__ == "OVTracker":
                    bboxes, labels, ids = self.tracker.match(
                        bboxes=det_bboxes,
                        labels=det_labels,
                        embeds=track_feats,
                        cls_embeds=cem_feats,
                        frame_id=frame_id,
                        method=self.method,
                    )
                else:
                    #if self.motion is not None:
                        #self.init_motion()
                    bboxes, labels, ids = self.tracker.track(
                        model=self,
                        bboxes=det_bboxes,
                        labels=det_labels,
                        embeds=track_feats,
                        cls_embeds=cem_feats,
                        frame_id=frame_id,
                        method=self.method,
                    )
                pred_track_instances = InstanceData()

                # return zero bboxes if there is no track targets
                if bboxes.shape[0] == 0:
                    ids = torch.zeros_like(labels)
                    pred_track_instances = det_results[0].pred_instances.clone()
                    pred_track_instances.instances_id = ids
                else:
                    pred_track_instances.bboxes = bboxes[:, :4]
                    pred_track_instances.labels = labels
                    pred_track_instances.scores = bboxes[:, 4]
                    pred_track_instances.instances_id = ids
            ################################################
            if self.visual:
                import os
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                from PIL import Image
                import random
                import json

                def compute_iou(boxA, boxB):
                    # boxA and boxB: [x1, y1, x2, y2]
                    xA = max(boxA[0], boxB[0])
                    yA = max(boxA[1], boxB[1])
                    xB = min(boxA[2], boxB[2])
                    yB = min(boxA[3], boxB[3])
                    interW = max(0, xB - xA)
                    interH = max(0, yB - yA)
                    interArea = interW * interH
                    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
                    return iou

                def filter_pred_by_iou(pred_bboxes, gt_bboxes, iou_thr=0.5):
                    keep_indices = []
                    for i, pbox in enumerate(pred_bboxes):
                        for gtbox in gt_bboxes:
                            if compute_iou(pbox, gtbox) > iou_thr:
                                keep_indices.append(i)
                                break
                    return keep_indices

                def draw_boxes(ax, image, bboxes, labels, ids, ious=None, title=None):
                    dataset = 'OVT-B'
                    if dataset == 'TAO':
                        annotation_file = 'data/tao/annotations/tao_val_lvis_v1_classes_plot.json'
                    else:
                        annotation_file = 'data/OVT-B/ovtb_ann_plot.json'

                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                    categories = data['categories']
                    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

                    color_palette = [
                        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"
                    ]
                    label_to_color = {
                        label: color_palette[label % len(color_palette)] for label in labels
                    }

                    ax.imshow(image)
                    indices = list(range(len(bboxes)))
                    if ious is not None:
                        indices = sorted(indices, key=lambda i: ious[i])
                    used_positions = set()

                    for idx in indices:
                        bbox = bboxes[idx]
                        label = labels[idx]
                        iid = ids[idx]
                        category_name = cat_id_to_name.get(label + 1, None)
                        if category_name is None:
                            continue
                        edgecolor = color_palette[label % len(color_palette)]

                        x, y, x2, y2 = [float(v) for v in bbox]

                        # Clamp bbox to image size
                        img_w, img_h = image.size
                        x = max(0, min(x, img_w - 1))
                        y = max(0, min(y, img_h - 1))
                        x2 = max(0, min(x2, img_w - 1))
                        y2 = max(0, min(y2, img_h - 1))

                        w, h = x2 - x, y2 - y
                        label_x, label_y = x, y # 5 pixels above
                        if label_y < 0:
                            label_y = 0  # clamp so it doesn’t go outside image

                        # Default values
                        linewidth = 3

                        # Adjust linewidth by IoU only
                        if ious is not None:
                            iou = ious[idx]
                            if iou >= 0.7:
                                linewidth = 4
                            elif iou >= 0.3:
                                linewidth = 2
                            else:
                                linewidth = 1

                        '''
                        edgecolor = 'red'
                        facecolor = 'none'
                        alpha = 1.0
                        fontsize = 8
                        fontweight = 'normal'
                        textcolor = 'yellow'
                        bgcolor = 'black'
                        label_x, label_y = x, y  # default: left-up

                        if ious is not None:
                            iou = ious[idx]
                            if iou >= 0.7:
                                linewidth = 4
                                alpha = 1.0
                                fontsize = 10
                                fontweight = 'bold'
                                textcolor = 'lime'
                                bgcolor = 'black'
                                label_x, label_y = x, y
                            elif iou >= 0.3:
                                linewidth = 2
                                alpha = 0.7
                                fontsize = 8
                                textcolor = 'yellow'
                                bgcolor = 'gray'
                            else:
                                linewidth = 1
                                alpha = 0.3
                                fontsize = 7
                                textcolor = 'white'
                                bgcolor = 'gray'
                        '''
                        rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=edgecolor,
                                                 facecolor='none', alpha=1.0)
                        ax.add_patch(rect)
                        pos_tuple = (int(label_x), int(label_y))
                        if pos_tuple not in used_positions:


                            ax.text(label_x, label_y, f'{category_name}/{iid}',
                                    color='white',
                                    fontsize=10,
                                    backgroundcolor=edgecolor,
                                    verticalalignment='top',
                                    fontweight='normal',
                                    bbox=dict(facecolor=edgecolor, edgecolor='none', boxstyle='round,pad=0.2'))
                            '''
                            ax.text(label_x, label_y, f'{category_name}/{iid}',
                                    color=edgecolor,  # same as bbox color
                                    fontsize=20,
                                    verticalalignment='bottom',
                                    fontweight='normal')
                                    '''


                            used_positions.add(pos_tuple)
                        # (If overlap still occurs, you can increase label_w/label_h, or try to find an unused corner)

                    if title:
                        ax.set_title(title)
                    ax.axis('off')

                def get_pred_ious(pred_bboxes, gt_bboxes):
                    ious = []
                    for pbox in pred_bboxes:
                        if len(gt_bboxes) > 0:
                            ious.append(max([compute_iou(pbox, gtbox) for gtbox in gt_bboxes]))
                        else:
                            ious.append(0.0)
                    return ious

                def visualize_gt_pred(
                        image_path,
                        gt_bboxes, gt_labels, gt_ids,
                        pred_bboxes, pred_labels, pred_ids, video_id,
                        tracker_class,
                        iou_thr=0.7
                ):
                    # Prepare output directory
                    outdir = os.path.join(tracker_class, str(video_id))
                    os.makedirs(outdir, exist_ok=True)

                    # Load image (PIL or cv2)
                    pred_bboxes = pred_bboxes.cpu().tolist()
                    pred_labels = pred_labels.cpu().tolist()
                    pred_ids = pred_ids.cpu().tolist()

                    #keep_idx = filter_pred_by_iou(pred_bboxes, gt_bboxes, iou_thr)
                    #pred_bboxes = [pred_bboxes[i] for i in keep_idx]
                    #pred_labels = [pred_labels[i] for i in keep_idx]
                    #pred_ids = [pred_ids[i] for i in keep_idx]
                    pred_ious = get_pred_ious(pred_bboxes, gt_bboxes)
                    image = Image.open(image_path)

                    # -------- Save GT figure --------
                    basename = os.path.basename(image_path)
                    base_noext = os.path.splitext(basename)[0]
                    fig, ax = plt.subplots(figsize=(8, 8))
                    draw_boxes(ax, image, gt_bboxes, gt_labels, gt_ids)
                    ax.axis('off')
                    outfile_gt = os.path.join(outdir, f"{base_noext}_gt.png")
                    plt.savefig(outfile_gt, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # -------- Save Prediction figure --------
                    fig, ax = plt.subplots(figsize=(8, 8))
                    draw_boxes(ax, image, pred_bboxes, pred_labels, pred_ids, ious=pred_ious)
                    ax.axis('off')
                    outfile_pred = os.path.join(outdir, f"{base_noext}_pred.png")
                    plt.savefig(outfile_pred, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    '''
                    # Create figure with two subplots in a row
                    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                    draw_boxes(axes[0], image, gt_bboxes, gt_labels, gt_ids, title="Ground Truth")
                    draw_boxes(axes[1], image, pred_bboxes, pred_labels, pred_ids, ious=pred_ious, title="Prediction")

                    plt.suptitle(f'Method: {tracker_class}', fontsize=16)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    # Save
                    basename = os.path.basename(image_path)
                    outfile = os.path.join(outdir, f"{os.path.splitext(basename)[0]}.png")
                    plt.savefig(outfile)
                    plt.close()
                    '''


                image_path = img_data_sample.img_path
                gt_instances = img_data_sample.instances
                gt_bboxes = [instance['bbox'] for instance in gt_instances]
                gt_labels = [instance['bbox_label'] for instance in gt_instances]
                gt_ids = [instance['instance_id'] for instance in gt_instances]
                pred_bboxes = pred_track_instances.bboxes
                pred_labels = pred_track_instances.labels
                pred_ids = pred_track_instances.instances_id
                video_id = img_data_sample.video_id
                visualize_gt_pred(
                    image_path, gt_bboxes, gt_labels, gt_ids,
                    pred_bboxes, pred_labels, pred_ids, video_id,
                    tracker_class=self.tracker.__class__.__name__
                )
            ###############################################


            img_data_sample.pred_track_instances = pred_track_instances

        #print('pred_track_id: ', img_data_sample.pred_track_instances.instances_id)
        return [track_data_sample]
