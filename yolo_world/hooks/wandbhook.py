from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import wandb
import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F


def draw_bboxes(img, bboxes, labels, scores, class_names, score_thr=0.3):
    """
    Draw bounding boxes on an image using OpenCV.

    Args:
        img (numpy array): The image.
        bboxes (numpy array): Bounding box coordinates (x1, y1, x2, y2).
        labels (numpy array): Class labels.
        scores (numpy array): Confidence scores.
        class_names (list): List of class names.
        score_thr (float): Score threshold for visualization.

    Returns:
        img (numpy array): Image with bounding boxes drawn.
    """
    for bbox, label, score in zip(bboxes, labels, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        class_name = class_names[label]

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put class name and confidence score
        text = f"{class_name} {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img


@HOOKS.register_module()
class WandBHook(Hook):
    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        # Log loss values to WandB
        log_data = {"train/loss": outputs['loss'].item(), "iteration": batch_idx}
        if 'track_loss' in outputs:
            log_data["train/track_loss"] = outputs['track_loss'].item()
        if 'loss_cls' in outputs:
            log_data["train/loss_cls"] = outputs['loss_cls'].item()
        if 'loss_bbox' in outputs:
            log_data["train/loss_bbox"] = outputs['loss_bbox'].item()
        if 'loss_dfl' in outputs:
            log_data["train/loss_dfl"] = outputs['loss_dfl'].item()
        wandb.log(log_data)

    def after_train_epoch(self, runner):
        # Log learning rate
        lr = runner.optim_wrapper.get_lr()
        wandb.log({"train/lr": lr["base_lr"][0], "epoch": runner.epoch})

    def after_val_epoch(self, runner, metrics=None):
        if metrics:
            wandb.log(metrics)
        try:
            # Get results already computed in val loop
            outputs = runner.outputs  # ← this contains predictions
            dataset = runner.val_dataloader.dataset

            # Pick a random result and corresponding input
            rand_idx = random.randint(0, len(outputs) - 1)
            pred_result = outputs[rand_idx]
            img_info = dataset.get_data_info(rand_idx)
            img_path = img_info['img_path']

            # Load the original image from disk
            import cv2
            img = cv2.imread(img_path)

            # Get prediction data
            if hasattr(pred_result, 'pred_track_instances'):
                bboxes = pred_result.pred_track_instances.bboxes.cpu().numpy()
                labels = pred_result.pred_track_instances.labels.cpu().numpy()
                scores = pred_result.pred_track_instances.scores.cpu().numpy()

                # Draw and log
                img_with_boxes = draw_bboxes(
                    img, bboxes, labels, scores, dataset.METAINFO['classes']
                )
                wandb.log({"val_sample": wandb.Image(img_with_boxes)})
            else:
                print("No predictions for sample:", rand_idx)
        except Exception as e:
            print(f"Error during validation visualization: {e}")