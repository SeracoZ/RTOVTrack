# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import TrackDataSample
from mmdet.structures.bbox import bbox_overlaps
from mmyolo.registry import MODELS
from .base_tracker import BaseTracker


results_for_video = []

@MODELS.register_module()
class SortEmbedTracker(BaseTracker):
    """Tracker for Quasi-Dense Tracking.
    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.8.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.5.
        match_score_thr (float): The match threshold. Defaults to 0.5.
        memo_tracklet_frames (int): The most frames in a tracklet memory.
            Defaults to 10.
        memo_backdrop_frames (int): The most frames in the backdrops.
            Defaults to 1.
        memo_momentum (float): The momentum value for embeds updating.
            Defaults to 0.8.
        nms_conf_thr (float): The nms threshold for confidence.
            Defaults to 0.5.
        nms_backdrop_iou_thr (float): The nms threshold for backdrop IoU.
            Defaults to 0.3.
        nms_class_iou_thr (float): The nms threshold for class IoU.
            Defaults to 0.7.
        with_cats (bool): Whether to track with the same category.
            Defaults to True.
        match_metric (str): The match metric. Defaults to 'bisoftmax'.
    """

    def __init__(self,
                 init_score_thr: float = 0.8,
                 obj_score_thr: float = 0.5,
                 match_score_thr: float = 0.5,
                 memo_tracklet_frames: int = 10,
                 memo_backdrop_frames: int = 1,
                 memo_momentum: float = 0.8,
                 nms_conf_thr: float = 0.5,
                 nms_backdrop_iou_thr: float = 0.3,
                 nms_class_iou_thr: float = 0.7,
                 with_cats: bool = True,
                 match_metric: str = 'bisoftmax',
                 **kwargs):
        super().__init__(**kwargs)
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ['bisoftmax', 'softmax', 'cosine']
        self.match_metric = match_metric

        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

    def update(self, ids: Tensor, bboxes: Tensor, embeds: Tensor,
               labels: Tensor, scores: Tensor, frame_id: int) -> None:
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes (Tensor): of shape (N, 5).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            scores (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        for id, bbox, embed, label, score in zip(ids[tracklet_inds],
                                                 bboxes[tracklet_inds],
                                                 embeds[tracklet_inds],
                                                 labels[tracklet_inds],
                                                 scores[tracklet_inds]):
            id = int(id)
            # update the tracked ones and initialize new tracks
            if id in self.tracks.keys():
                velocity = (bbox - self.tracks[id]['bbox']) / (
                    frame_id - self.tracks[id]['last_frame'])
                self.tracks[id]['bbox'] = bbox
                self.tracks[id]['embed'] = (
                    1 - self.memo_momentum
                ) * self.tracks[id]['embed'] + self.memo_momentum * embed
                self.tracks[id]['last_frame'] = frame_id
                self.tracks[id]['label'] = label
                self.tracks[id]['score'] = score
                self.tracks[id]['velocity'] = (
                    self.tracks[id]['velocity'] * self.tracks[id]['acc_frame']
                    + velocity) / (
                        self.tracks[id]['acc_frame'] + 1)
                self.tracks[id]['acc_frame'] += 1
            else:
                self.tracks[id] = dict(
                    bbox=bbox,
                    embed=embed,
                    label=label,
                    score=score,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0)
        # backdrop update according to IoU
        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = bbox_overlaps(bboxes[backdrop_inds], bboxes)
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]
        # old backdrops would be removed at first
        self.backdrops.insert(
            0,
            dict(
                bboxes=bboxes[backdrop_inds],
                embeds=embeds[backdrop_inds],
                labels=labels[backdrop_inds]))

        # pop memo
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self) -> Tuple[Tensor, ...]:
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        # velocity of tracks
        memo_vs = []
        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v['bbox'][None, :])
            memo_embeds.append(v['embed'][None, :])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])

        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        # get backdrops
        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop['embeds'].size(0)),
                                      -1,
                                      dtype=torch.long)
            backdrop_vs = torch.zeros_like(backdrop['bboxes'])
            memo_bboxes.append(backdrop['bboxes'])
            memo_embeds.append(backdrop['embeds'])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels.append(backdrop['labels'][:, None])
            memo_vs.append(backdrop_vs)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_bboxes, memo_labels, memo_embeds, memo_ids.squeeze(
            0), memo_vs

    def track(self,
              model: torch.nn.Module,
              img: torch.Tensor,
              feats: List[torch.Tensor],
              data_sample: TrackDataSample,
              rescale=True,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                QDTrack method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_instances`.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                True.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo

        bboxes = data_sample.pred_instances.bboxes
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores
        embeds = data_sample.pred_instances.track_embeds


        frame_id = metainfo.get('frame_id', -1)
        # create pred_track_instances
        pred_track_instances = InstanceData()

        # return zero bboxes if there is no track targets
        if bboxes.shape[0] == 0:
            ids = torch.zeros_like(labels)
            pred_track_instances = data_sample.pred_instances.clone()
            pred_track_instances.instances_id = ids
            return pred_track_instances

        # sort according to the object_score
        _, inds = scores.sort(descending=True)
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        embeds = embeds[inds, :] # torch.Size([9, 512])

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.new_ones((bboxes.size(0)))
        ious = bbox_overlaps(bboxes, bboxes)

        for i in range(1, bboxes.size(0)):
            thr = self.nms_backdrop_iou_thr if scores[
                i] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0

        valids = valids == 1
        bboxes = bboxes[valids]
        scores = scores[valids]
        labels = labels[valids]
        embeds = embeds[valids, :]

        #########################
        import numpy as np
        import cv2
        import os

        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        def frame_accuracy(gt_bboxes, gt_labels, pred_bboxes, pred_labels, iou_thresh=0.5):
            if len(gt_bboxes) == 0:
                return 1.0 if len(pred_bboxes) == 0 else 0.0
            matched = 0
            for gt_box, gt_label in zip(gt_bboxes, gt_labels):
                for pred_box, pred_label in zip(pred_bboxes, pred_labels):
                    if pred_label == gt_label and iou(gt_box, pred_box) > iou_thresh:
                        matched += 1
                        break
            return matched / len(gt_bboxes)

        # --- Drawing function for labels ---
        def draw_box_with_label(img, box, label, color, label_offset=-10):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_pos = (x1, max(0, y1 + label_offset))
            cv2.putText(img, str(label), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        #check the detect result
        global results_for_video
        if data_sample.video_id == 98:
            frame_id = int(data_sample.metainfo['frame_id'])
            if frame_id == 0:
                results_for_video = []

            instances = data_sample.instances
            gt_bboxes = []
            gt_labels = []
            for instance in instances:
                gt_bboxes.append(instance['bbox'])
                gt_labels.append(instance['bbox_label'])
            results_for_video.append({
                "frame_id": frame_id,
                "pred_bboxes": bboxes,
                "pred_labels": labels,
                "pred_scores": scores,
                "gt_bboxes": np.array(gt_bboxes),
                "gt_labels": np.array(gt_labels)
            })

            if frame_id == 39:
                per_frame_acc = []
                for res in results_for_video:
                    acc = frame_accuracy(
                        res["gt_bboxes"], res["gt_labels"], res["pred_bboxes"], res["pred_labels"]
                    )
                    per_frame_acc.append(acc)

                print(f"Mean accuracy over video {data_sample.video_id}: {np.mean(per_frame_acc):.3f}")

            # -- Visualize a few frames --
            os.makedirs('det_vis', exist_ok=True)
            img_path = data_sample.metainfo['img_path']
            image = cv2.imread(img_path)

            image_gt = image.copy()
            for bbox, label in zip(gt_bboxes, gt_labels):
                draw_box_with_label(image_gt, bbox, f"GT:{label}", (0, 0, 255))
            # Draw predictions in green
            image_pred = image.copy()
            for pred_box, pred_label in zip(bboxes, labels):
                max_iou = 0
                match_gt_label = None
                for gt_box, gt_label in zip(gt_bboxes, gt_labels):
                    iou_val = iou(pred_box, gt_box)
                    if iou_val > max_iou:
                        max_iou = iou_val
                        match_gt_label = gt_label
                if max_iou > 0.5:  # Draw only if high IoU
                    draw_box_with_label(image_pred, pred_box, f"Pred:{pred_label}|GT:{match_gt_label}", (0, 255, 0))
            combined = np.concatenate([image_gt, image_pred], axis=1)
            out_path = f"det_vis/frame_{frame_id:05d}.jpg"
            cv2.imwrite(out_path, combined)

        #########################
        # init ids container
        ids = torch.full((bboxes.size(0), ), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.size(0) > 0 and not self.empty:
            (memo_bboxes, memo_labels, memo_embeds, memo_ids,
             memo_vs) = self.memo

            if self.match_metric == 'bisoftmax':
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                match_scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feats = torch.mm(embeds, memo_embeds.t())
                match_scores = feats.softmax(dim=1)
            elif self.match_metric == 'cosine':
                match_scores = torch.mm(
                    F.normalize(embeds, p=2, dim=1),
                    F.normalize(memo_embeds, p=2, dim=1).t())
            else:
                raise NotImplementedError

            # track with the same category
            if self.with_cats:
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                match_scores *= cat_same.float().to(match_scores.device)

            # track according to match_scores
            for i in range(bboxes.size(0)):
                conf, memo_ind = torch.max(match_scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        # keep bboxes with high object score
                        # and remove background bboxes
                        if scores[i] > self.obj_score_thr:
                            ids[i] = id
                            match_scores[:i, memo_ind] = 0
                            match_scores[i + 1:, memo_ind] = 0
                        else:
                            if conf > self.nms_conf_thr:
                                ids[i] = -2

        # initialize new tracks
        new_inds = (ids == -1) & (scores > self.init_score_thr).cpu()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(
            self.num_tracks, self.num_tracks + num_news, dtype=torch.long)
        self.num_tracks += num_news

        self.update(ids, bboxes, embeds, labels, scores, frame_id)
        tracklet_inds = ids > -1
        # update pred_track_instances
        pred_track_instances.bboxes = bboxes[tracklet_inds]
        pred_track_instances.labels = labels[tracklet_inds]
        pred_track_instances.scores = scores[tracklet_inds]
        pred_track_instances.instances_id = ids[tracklet_inds]

        return pred_track_instances
