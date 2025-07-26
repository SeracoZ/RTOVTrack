import os.path as osp
from collections import defaultdict
from typing import Any, List, Tuple
import traceback
import numpy as np
from mmdet.datasets import BaseVideoDataset, LVISV1Dataset, LVISV05Dataset
from mmyolo.registry import DATASETS
#from mmdet.registry import DATASETS
import json


# from masa
@DATASETS.register_module()
class Taov05Dataset(BaseVideoDataset):
    """Dataset for TAO benchmark.

    """

    METAINFO = LVISV05Dataset.METAINFO

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def prepare_data(self, idx) -> Any:
        """Get date processed by ``self.pipeline``. Note that ``idx`` is a
        video index in default since the base element of video dataset is a
        video. However, in some cases, we need to specific both the video index
        and frame index. For example, in traing mode, we may want to sample the
        specific frames and all the frames must be sampled once in a epoch; in
        test mode, we may want to output data of a single image rather than the
        whole video for saving memory.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        if isinstance(idx, tuple):
            assert len(idx) == 2, "The length of idx must be 2: "
            "(video_index, frame_index)"
            video_idx, frame_idx = idx[0], idx[1]
        else:
            video_idx, frame_idx = idx, None

        data_info = self.get_data_info(video_idx)
        if self.test_mode:
            # Support two test_mode: frame-level and video-level
            final_data_info = defaultdict(list)
            ref_data_info = defaultdict(list)
            if frame_idx is None:
                frames_idx_list = list(range(data_info["video_length"]))
            else:
                frames_idx_list = [frame_idx]
            image_len = len(data_info["images"])
            for index in frames_idx_list:
                # 处理原始帧数据
                frame_ann = data_info["images"][index]
                frame_ann["video_id"] = data_info["video_id"]
                # Collate data_list (list of dict to dict of list)
                for key, value in frame_ann.items():
                    final_data_info[key].append(value)

                # 生成参考帧索引，在index的基础上随机加减2，但不要0偏移
                offset_options = [-1]  # 排除0的偏移选项
                offset = np.random.choice(offset_options)  # 从选项中随机选择
                ref_index = max(0, min(image_len - 1, index + offset))  # 确保在有效范围内

                # 处理参考帧数据
                ref_frame_ann = data_info["images"][ref_index]
                ref_frame_ann["video_id"] = data_info["video_id"]
                for key, value in ref_frame_ann.items():
                    ref_data_info[key].append(value)

                # copy the info in video-level into img-level
                # TODO: the value of this key is the same as that of
                # `video_length` in test mode
                final_data_info["ori_video_length"].append(data_info["video_length"])
                ref_data_info["ori_video_length"].append(data_info["video_length"])

            final_data_info["video_length"] = [len(frames_idx_list)] * len(frames_idx_list)
            ref_data_info["video_length"] = [len(frames_idx_list)] * len(frames_idx_list)

            # xiao: add texts
            # texts = []
            # final_data_info["texts"] = texts.append(self.METAINFO["classes"]["bbox_label"])
            # ref_data_info["texts"] = texts.append(self.METAINFO["classes"]["bbox_label"])

            final_data = self.pipeline(final_data_info)
            ref_data = self.pipeline(ref_data_info)

            return {
                'key_frame': final_data,
                'ref_frame': ref_data
            }
        else:
            # Specify `key_frame_id` for the frame sampling in the pipeline
            final_data_info = defaultdict(list)
            ref_data_info = defaultdict(list)
            if frame_idx is None:
                frames_idx_list = list(range(data_info["video_length"]))
            else:
                frames_idx_list = [frame_idx]
            image_len = len(data_info["images"])
            for index in frames_idx_list:
                # 处理原始帧数据
                frame_ann = data_info["images"][index]
                frame_ann["video_id"] = data_info["video_id"]
                # Collate data_list (list of dict to dict of list)
                for key, value in frame_ann.items():
                    final_data_info[key].append(value)

                # 生成参考帧索引，在index的基础上随机加减2，但不要0偏移
                offset_options = [-2, -1, 1, 2]  # 排除0的偏移选项
                offset = np.random.choice(offset_options)  # 从选项中随机选择
                ref_index = max(0, min(image_len - 1, index + offset))  # 确保在有效范围内

                # 处理参考帧数据
                ref_frame_ann = data_info["images"][ref_index]
                ref_frame_ann["video_id"] = data_info["video_id"]
                for key, value in ref_frame_ann.items():
                    ref_data_info[key].append(value)

                # copy the info in video-level into img-level
                # TODO: the value of this key is the same as that of
                # `video_length` in test mode
                final_data_info["ori_video_length"].append(data_info["video_length"])
                ref_data_info["ori_video_length"].append(data_info["video_length"])

            final_data_info["video_length"] = [len(frames_idx_list)] * len(frames_idx_list)
            ref_data_info["video_length"] = [len(frames_idx_list)] * len(frames_idx_list)

            # xiao: add texts
            # texts = []
            # final_data_info["texts"] = texts.append(self.METAINFO["classes"]["bbox_label"])
            # ref_data_info["texts"] = texts.append(self.METAINFO["classes"]["bbox_label"])

            final_data = self.pipeline(final_data_info)
            ref_data = self.pipeline(ref_data_info)
            '''
            for i in range(self.max_refetch):
                # To confirm the results passed the training pipeline
                # of the wrapper is not None.
                try:
                    data = self.pipeline(final_data_info)
                except Exception as e:
                    print("Error occurred while running pipeline", f" with error: {e}")
                    # print('Empty instances due to augmentation, re-sampling...')
                    traceback.print_exc()
                    video_idx = self._rand_another(video_idx)
                    final_data_info = self.get_data_info(video_idx)
                    continue

                if data is not None:
                    break

            return data
            '''
            return {
                'key_frame': final_data,
                'ref_frame': ref_data
            }


@DATASETS.register_module()
class Taov1Dataset(Taov05Dataset):
    """Dataset for TAO benchmark.

    """
    METAINFO = LVISV1Dataset.METAINFO.copy()

@DATASETS.register_module()
class Taov1_TrainDataset(Taov05Dataset):
    """Dataset for TAO benchmark.

    """
    file_path = 'data/tao/annotations/train_ours_f.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    categories = data['categories']
    texts = [[category['name'] for category in categories]]
    METAINFO = LVISV1Dataset.METAINFO.copy()
    METAINFO['classes'] = texts[0]
    #print(METAINFO)

@DATASETS.register_module()
class TaoTestDataset(Taov05Dataset):
    file_path = 'data/tao/annotations/test_categories.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = data['categories']
    texts = [[category['name'] for category in categories]]
    METAINFO = LVISV1Dataset.METAINFO.copy()
    METAINFO['classes'] = texts[0]
    #print(METAINFO)

@DATASETS.register_module()
class OVTBDataset(Taov05Dataset):
    file_path = 'data/OVT-B/ovtb_classes.txt'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    texts = [line.strip() for line in lines if line.strip()]
    METAINFO = LVISV1Dataset.METAINFO.copy()
    METAINFO['classes'] = texts
    #print(METAINFO)

