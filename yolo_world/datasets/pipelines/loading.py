#from mmdet.datasets.builder import PIPELINES
#from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
#from mmdet.datasets.pipelines.loading import FilterAnnotations

from mmcv.transforms import BaseTransform
from mmyolo.registry import TRANSFORMS
from mmdet.datasets.transforms import LoadAnnotations
from mmcv.transforms import LoadImageFromFile
import numpy as np
import torch
from mmdet.structures.bbox import get_box_type

@TRANSFORMS.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    class LoadMultiImagesFromFile(LoadImageFromFile):
        def __call__(self, results):
            # If results is a list of dicts → apply super() on each
            if isinstance(results, list):
                outs = []
                for _results in results:
                    _results = super().__call__(_results)
                    outs.append(_results)
                return outs
            # If results is a single dict → normal behavior
            elif isinstance(results, dict):
                return super().__call__(results)
            else:
                raise TypeError(f"Unsupported input type: {type(results)}")


@TRANSFORMS.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    def __init__(self, with_ins_id=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_ins_id = with_ins_id

    def _load_ins_ids(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded label annotations.
        """

        results["gt_match_indices"] = results["ann_info"]["match_indices"].copy()

        return results

    def transform(self, results):
        if isinstance(results, dict):
            results = [results]
        outs = []
        for _results in results:
            # ---- Add 'instances' list ----
            ann_info = _results.get('ann_info', {})
            bboxes = ann_info.get('bboxes', [])
            if len(bboxes) > 0 and isinstance(bboxes[0], np.ndarray):
                bboxes = np.array(bboxes, dtype=np.float32)
            labels = ann_info.get('labels', [])
            instance_ids = ann_info.get('instance_ids', [])
            match_indices = ann_info.get('match_indices', [])
            bboxes_ignore = ann_info.get('bboxes_ignore', [])
            # Ensure all are tensors or arrays and same length
            instances = []
            for i in range(len(bboxes)):
                instance = {
                    'bbox': bboxes[i],
                    'bbox_label': labels[i] if len(labels) > i else -1,
                    'instance_id': instance_ids[i] if len(instance_ids) > i else -1,
                    'match_index': match_indices[i] if len(match_indices) > i else -1,
                    'ignore_flag': bboxes_ignore[i] if len(bboxes_ignore) > i else False,
                }
                instances.append(instance)
            _results['instances'] = instances
            # -----------------------------
            _results = super().transform(_results)
            if self.with_ins_id:
                _results = self._load_ins_ids(_results)

            outs.append(_results)

        return outs

    def _load_bboxes(self, results: dict) -> None:
        gt_bboxes = [instance['bbox'] for instance in results.get('instances', [])]
        # Always convert list of arrays to a stacked array (N, 4)
        if len(gt_bboxes) > 0:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        if self.box_type is None:
            results['gt_bboxes'] = gt_bboxes
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(
            [instance['ignore_flag'] for instance in results.get('instances', [])], dtype=bool
        )


@TRANSFORMS.register_module()
class SeqFilterAnnotations(BaseTransform):
    def __init__(self, min_gt_bbox_wh=(1., 1.), keep_empty=True):
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty

    def transform(self, results):
        assert isinstance(results, list), "Expected list of results for sequence filtering."
        return [self._filter(res) for res in results if res is not None]

    def _filter(self, results):
        assert "gt_bboxes" in results
        gt_bboxes = results["gt_bboxes"]
        if gt_bboxes.shape[0] == 0:
            return results
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes = gt_bboxes.tensor  # shape (N,4), torch.Tensor
        w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
        h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
        keep = (w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1])

        if not keep.any():
            return results if self.keep_empty else None

        for key in ['gt_bboxes', 'gt_bboxes_labels', 'gt_masks',
                    'gt_semantic_seg', 'gt_match_indices']:
            if key in results:
                if isinstance(keep, torch.Tensor):
                    keep = keep.cpu().numpy()
                if keep.dtype != bool:
                    # Convert indices to boolean mask
                    mask = np.zeros(results[key].shape[0], dtype=bool)
                    mask[keep] = True
                    keep = mask
                results[key] = results[key][keep]
                '''
                try:
                    results[key] = results[key][keep]
                except Exception as e:
                    print("=" * 60)
                    print(f"Error for key: {key}")
                    print(f"type(results[key]): {type(results[key])}")
                    print(f"results[{key}]: {results[key]}")
                    print(f"results[{key}].shape: {getattr(results[key], 'shape', None)}")
                    print(f"keep: {keep}")
                    print(f"keep.shape: {getattr(keep, 'shape', None)}")
                    print(f"All results keys: {list(results.keys())}")
                    print(f"results: {results}")
                    print(f"Exception: {e}")
                    print("=" * 60)
                    raise
                    '''
        return results
