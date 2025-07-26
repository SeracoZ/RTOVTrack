# yolo_world/hooks/custom_hooks.py
from mmengine.registry import HOOKS
from mmengine.hooks import CheckpointHook

@HOOKS.register_module()
class TaoCheckpointHook(CheckpointHook):
    def __init__(self, val_texts=None, **kwargs):
        super().__init__(**kwargs)
        self.val_texts = val_texts
        self.original_texts = None

    def before_val_epoch(self, runner):
        if self.val_texts is None:
            return
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model.detector, 'texts'):
            self.original_texts = model.detector.texts
        model.detector.reparameterize(self.val_texts)

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