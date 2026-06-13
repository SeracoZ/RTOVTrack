import torch
import torch.nn as nn
import torch.nn.functional as F
from ..tracker import cal_similarity

from mmyolo.registry import MODELS
from mmdet.models.losses import MultiPosCrossEntropyLoss, L2Loss
#from mmdet.registry import MODELS
MODELS.register_module()(MultiPosCrossEntropyLoss)
MODELS.register_module()(L2Loss)


class TrackHeadModule(nn.Module):
    def __init__(self, margin=0.2, temperature=0.1, softmax_temp=-1,
                 loss_track=dict(type="MultiPosCrossEntropyLoss", loss_weight=0.25),
                 loss_track_aux=dict(
                     type="L2Loss", neg_pos_ub=3, pos_margin=0.3, loss_weight=1.0, hard_mining=True)
                 ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.softmax_temp = softmax_temp
        self.loss_track = MODELS.build(loss_track)
        if loss_track_aux is not None:
            self.loss_track_aux = MODELS.build(loss_track_aux)
        else:
            self.loss_track_aux = None

    def get_track_targets(
        self, gt_match_indices, key_ids, ref_ids
    ):
        track_targets = []
        track_weights = []
        for _gt_match_indices, key_id, ref_id in zip(
            gt_match_indices, key_ids, ref_ids
        ):
            targets = _gt_match_indices.new_zeros(
                (len(key_id), len(ref_id)), dtype=torch.int
            )
            _match_indices = _gt_match_indices[key_id]
            pos2pos = (
                _match_indices.view(-1, 1) == ref_id.view(1, -1)
            ).int()
            targets[:, : pos2pos.size(1)] = pos2pos
            weights = (targets.sum(dim=1) > 0).float()
            track_targets.append(targets)
            track_weights.append(weights)
        return track_targets, track_weights

    def match(self, key_embeds, ref_embeds, key_ids, ref_ids):
        dists, cos_dists = [], []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            dist = cal_similarity(
                key_embed,
                ref_embed,
                method="dot_product",
                temperature=self.softmax_temp,
            )
            dists.append(dist)
            if self.loss_track_aux is not None:
                cos_dist = cal_similarity(key_embed, ref_embed, method="cosine")
                cos_dists.append(cos_dist)
            else:
                cos_dists.append(None)
        return dists, cos_dists

    def loss(self, dists, cos_dists, targets, weights):
        losses = dict()

        loss_track = 0.0
        loss_track_aux = 0.0
        for _dists, _cos_dists, _targets, _weights in zip(
            dists, cos_dists, targets, weights
        ):
            if _dists.shape != _targets.shape:
                print(f"[Warning] Skipped due to shape mismatch: {_dists.shape} vs {_targets.shape}")
                continue
            loss_track += self.loss_track(
                _dists, _targets, _weights, avg_factor=_weights.sum()
            )

            if self.loss_track_aux is not None:
                loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses["loss_track"] = loss_track / len(dists)

        if self.loss_track_aux is not None:
            losses["loss_track_aux"] = loss_track_aux / len(dists)

        return losses

    def compute_batch_hard_triplet_loss(self, key_embeds, ref_embeds, key_ids, ref_ids):
        device = key_embeds.device
        key_embeds = key_embeds.view(-1, key_embeds.size(-1))
        ref_embeds = ref_embeds.view(-1, ref_embeds.size(-1))
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)

        dists = 1.0 - torch.mm(key_embeds, ref_embeds.t())  # (N, M)

        loss = 0.0
        valid = 0

        for i in range(key_embeds.size(0)):
            anchor_id = key_ids[i]
            pos_mask = (ref_ids == anchor_id)
            neg_mask = (ref_ids != anchor_id)

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            pos_dists = dists[i][pos_mask]
            neg_dists = dists[i][neg_mask]

            hardest_pos = pos_dists.max()
            hardest_neg = neg_dists.min()

            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            loss += triplet_loss
            valid += 1

        if valid > 0:
            final_loss = loss / valid
            if torch.isnan(final_loss):
                #print("[Triplet Loss NaN] After reduction")
                final_loss = torch.tensor(0.0, requires_grad=True, device=device)
        else:
                #print("[Triplet Loss Warning] No valid triplets found.")
                final_loss = torch.tensor(0.0, requires_grad=True, device=device)

        return final_loss

    def compute_supcon_loss(self, key_embeds, ref_embeds, key_labels, ref_labels):
        device = key_embeds.device
        key_embeds = key_embeds.view(-1, key_embeds.size(-1))
        ref_embeds = ref_embeds.view(-1, ref_embeds.size(-1))
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)

        logits = torch.mm(key_embeds, ref_embeds.t()) / self.temperature  # (N, M)
        labels = key_labels.view(-1, 1) == ref_labels.view(1, -1)  # (N, M)
        labels = labels.float().to(device)

        if key_embeds.size(0) == ref_embeds.size(0) and torch.allclose(key_embeds, ref_embeds):
            mask = ~torch.eye(labels.size(0), dtype=torch.bool, device=device)
            labels = labels.masked_fill(~mask, 0)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (labels * log_prob).sum(dim=1) / (labels.sum(dim=1) + 1e-8)
        loss = -mean_log_prob_pos.mean()

        if torch.isnan(loss):
            #print("[SupCon Loss NaN] Returning 0")
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return loss

    def forward(self, key_embeds, ref_embeds, key_ids, ref_ids, key_labels=None, ref_labels=None):
        total_triplet = 0.0
        total_class_loss = 0.0
        valid_batches = 0
        for b in range(len(key_embeds)):
            triplet = self.compute_batch_hard_triplet_loss(
                key_embeds[b], ref_embeds[b], key_ids[b], ref_ids[b]
            )
            if torch.isnan(triplet):
                print(f"[NaN Detected] triplet loss in batch {b}")

            if key_labels is not None and ref_labels is not None:
                class_loss = self.compute_supcon_loss(
                    key_embeds[b], ref_embeds[b], key_labels[b], ref_labels[b]
                )
            else:
                class_loss = torch.tensor(0.0, device=key_embeds[b].device)

            if torch.isnan(class_loss):
                print(f"[NaN Detected] class loss in batch {b}")

            # Only accumulate if not empty
            if triplet.numel() > 0:
                total_triplet = total_triplet + triplet
            if class_loss.numel() > 0:
                total_class_loss = total_class_loss + class_loss
            valid_batches += 1

        if valid_batches > 0:
            total_triplet = total_triplet / valid_batches
            total_class_loss = total_class_loss / valid_batches
        else:
            # fallback to 0 loss
            device = key_embeds[0].device
            total_triplet = torch.tensor(0.0, device=device, requires_grad=True)
            total_class_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_triplet, total_class_loss