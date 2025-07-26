from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner.checkpoint import load_checkpoint

@HOOKS.register_module()
class ValTextReparamHook(Hook):
    def __init__(self, val_texts, train_texts, detector_ckpt_path=None):
        self.val_texts = val_texts
        self.train_texts = train_texts
        self.detector_ckpt_path = detector_ckpt_path

    def before_val_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        if self.detector_ckpt_path:
            print(f"[ValTextReparamHook] Reloading detector from {self.detector_ckpt_path}")
            load_checkpoint(model.detector, self.detector_ckpt_path, map_location='cpu', strict=False)


        print("[ValTextReparamHook] Switching to val texts.")
        model.detector.reparameterize(self.val_texts)

    def after_val_epoch(self, runner, metrics=None):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        print("[ValTextReparamHook] Restoring train texts.")
        model.detector.reparameterize(self.train_texts)
