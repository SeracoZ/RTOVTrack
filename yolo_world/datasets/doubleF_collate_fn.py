# yolo_world/datasets/collate_functions.py
from mmengine.registry import FUNCTIONS
from mmengine.dataset import default_collate
from mmdet.datasets.transforms.formatting import PackDetInputs, PackDetInputs

@FUNCTIONS.register_module()
def doubleF_collate_fn(batch):
    """Custom collation function to correctly batch key_frame and ref_frame."""
    key_frame = batch[0]['key_frame']
    ref_frame = batch[0]['ref_frame']
    
    key_inputs = key_frame['inputs']  # Should be a tensor
    key_data_samples = key_frame['data_samples']  # TrackDataSample object
    
    ref_inputs = ref_frame['inputs']
    ref_data_samples = ref_frame['data_samples']
    
    output_batch = {
        'inputs': default_collate([key_inputs, ref_inputs]),  # Batch the tensors
        'data_samples': [key_data_samples, ref_data_samples]  # Batch data samples
    }
    
    return output_batch