import torch
import torch.nn.functional as F
from typing import List

class PromptConditionedSuppression:
    def __init__(self,
                 topk: int = None,
                 threshold: float = 0.25,
                 sim_threshold: float = 0.2,
                 category_path=None):
        self.topk = topk
        self.threshold = threshold
        self.sim_threshold = sim_threshold
        self.class_embeds = None  # defer assignment to runtime

        if category_path:
            import json
            with open(category_path, 'r') as f:
                cat_data = json.load(f)
                categories = cat_data['categories']

            self.common_ids = [cat['id'] for cat in categories if cat['frequency'] == 'c']
            self.frequent_ids = [cat['id'] for cat in categories if cat['frequency'] == 'f']
            self.rare_ids = [cat['id'] for cat in categories if cat['frequency'] == 'r']
        else:
            self.common_ids = []
            self.frequent_ids = []
            self.rare_ids = []
        print('category init')

    def class_aware_keep_mask(self, pred_labels, max_scores, active_cls_ids, sim_scores=None):
        device = pred_labels.device
        cat_ids = pred_labels + 1  # Convert 0-based label index to category ID

        # Convert active class list to tensor
        active_cls_ids_tensor = torch.tensor(active_cls_ids, device=device)

        # Create lookup masks for category frequencies
        id_range = torch.arange(1, 1204, device=device)  # category IDs from 1 to 1203
        is_freq = torch.tensor([i in self.frequent_ids for i in id_range.tolist()], device=device)
        is_comm = torch.tensor([i in self.common_ids for i in id_range.tolist()], device=device)
        is_rare = torch.tensor([i in self.rare_ids for i in id_range.tolist()], device=device)

        is_freq_mask = is_freq[cat_ids - 1]
        is_comm_mask = is_comm[cat_ids - 1]
        is_rare_mask = is_rare[cat_ids - 1]
        is_active = torch.isin(cat_ids, active_cls_ids_tensor)
        # ========== Updated Rules ==========
        # ===== 1. Soft Confidence Thresholds =====
        conf_thresh = torch.zeros_like(max_scores)
        conf_thresh[is_freq_mask] = 0.5
        conf_thresh[is_comm_mask] = 0.3
        conf_thresh[is_rare_mask] = 0.1

        # ===== 2. Optional: Fuse Similarity + Score =====
        if sim_scores is not None:
            final_score = 0.6 * max_scores + 0.4 * sim_scores  # Tunable fusion
        else:
            final_score = max_scores

        # ===== 3. Keep Criteria =====
        keep_mask = (
                is_active |
                (is_rare_mask & (final_score > 0.1)) |
                (is_freq_mask & (final_score > 0.5)) |
                (is_comm_mask & (final_score > 0.3))
        )

        # ===== 4. Background Suppression (optional) =====
        bg_mask = (final_score < 0.1) & (~is_active)
        keep_mask = keep_mask & (~bg_mask)

        return keep_mask

    def get_active_classes(self, global_feat: torch.Tensor,
                           threshold: float = 0.25,
                           topk: int = None) -> List[int]:
        """
        Estimate active classes in an image using global features.
        Args:
            global_feat: (1, D), pooled image feature
            threshold: similarity threshold
            topk: optional, return top-k instead of threshold filtering
        Returns:
            List[int]: indices of active classes
        """
        threshold = threshold if threshold is not None else self.threshold
        topk = topk if topk is not None else self.topk

        global_feat = F.normalize(global_feat, dim=1)
        sim_scores = torch.matmul(global_feat, self.class_embeds.T).squeeze(0)  # (C,)

        if topk:
            _, indices = sim_scores.topk(topk)
            return indices.tolist()
        else:
            return (sim_scores > threshold).nonzero(as_tuple=True)[0].tolist()

    def apply_pcs_filter(self, box_embeds: torch.Tensor,
                         active_class_indices: List[int],
                         sim_threshold: float = 0.2,
                         contrastive_head=None) -> torch.Tensor:
        """
        PCS filter using ContrastiveHead scaling and bias.
        """
        sim_threshold = sim_threshold if sim_threshold is not None else self.sim_threshold

        if len(active_class_indices) == 0:
            return torch.zeros((box_embeds.size(0),), dtype=torch.bool, device=box_embeds.device)

        active_class_embeds = self.class_embeds[active_class_indices]  # (K, D)

        # Normalize embeddings
        box_embeds = F.normalize(box_embeds, dim=1)
        active_class_embeds = F.normalize(active_class_embeds, dim=1)

        # Compute similarity using the trained logit_scale and bias from ContrastiveHead
        sim = torch.matmul(box_embeds, active_class_embeds.T)  # (N, K)

###########################################
        max_sim, max_idx = sim.max(dim=1)  # max over classes for each box
        mean_sim = sim.mean(dim=1)  # mean over classes for each box

        topk_values, topk_indices = torch.topk(sim, k=5, dim=1)  # top-5 sims per box

        # Optional: print shapes and examples
        print("Max Sim:", max_sim.shape, max_sim[:5])  # Print first 5 for brevity
        print("Mean Sim:", mean_sim.shape, mean_sim[:5])
        print("Top-5 Sim Values:", topk_values.shape, topk_values[:5])
        print("Top-5 Sim Indices:", topk_indices.shape, topk_indices[:5])
        ###########################################


        if contrastive_head is not None:
            sim = sim * contrastive_head.logit_scale.exp() + contrastive_head.bias
        max_sim, pred_labels = sim.max(dim=1)

        # print for inspection
        print("Mean similarity:", max_sim.mean().item())
        print("Kept boxes:", (max_sim > 0.2).sum().item())
        keep_mask = (max_sim > sim_threshold)
        num_kept = keep_mask.sum().item()

        return keep_mask
