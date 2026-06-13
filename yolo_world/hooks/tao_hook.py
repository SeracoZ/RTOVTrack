# yolo_world/hooks/custom_hooks.py
from mmengine.registry import HOOKS
from mmengine.hooks import CheckpointHook
from mmengine.runner.checkpoint import load_checkpoint

@HOOKS.register_module()
class TaoCheckpointHook(CheckpointHook):
    def __init__(self, val_texts=None, detector_ckpt_path=None, **kwargs):
        super().__init__(**kwargs)
        self.val_texts = val_texts
        self.original_texts = None
        self.detector_ckpt_path = detector_ckpt_path

    '''
    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        model.eval()
        model.detector.bbox_head.head_module.track_preds.train()
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        # Save a snapshot of all detector params except track_preds
        self.ref_state_dict = {
            k: v.clone().cpu() for k, v in model.state_dict().items()
            if 'track_preds' not in k
        }

    def after_train_epoch(self, runner):
        import torch
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        for k, v in model.state_dict().items():
            if 'track_preds' in k:
                continue
            if not torch.equal(v.cpu(), self.ref_state_dict[k]):
                print(f"❗ [ParamCheckHook] Param {k} has changed!")
    '''

    def before_val_epoch(self, runner):
        #if self.val_texts is None:
            #return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model.detector, 'texts'):
            self.original_texts = model.detector.texts
        #model.detector.reparameterize(self.val_texts)
        #if self.detector_ckpt_path:
            #print(f"[ValTextReparamHook] Reloading detector from {self.detector_ckpt_path}")
            #load_checkpoint(model.detector, self.detector_ckpt_path, map_location='cpu', strict=False)

    def after_val_epoch(self, runner, metrics=None):
        if isinstance(metrics, dict):
            if 'tao_teta_metric/TAO' in metrics and isinstance(metrics['tao_teta_metric/TAO'], dict):
                if 'TETA' in metrics['tao_teta_metric/TAO']:
                    metrics['tao_teta_metric/TAO'] = float(metrics['tao_teta_metric/TAO']['TETA'][2])  # Extract the float value
                else:
                    raise KeyError("TETA key not found in tao_teta_metric/TAO dictionary.")
            else:
                raise KeyError("tao_teta_metric/TAO key not found or is not a dictionary.")

        super().after_val_epoch(runner, metrics=metrics)