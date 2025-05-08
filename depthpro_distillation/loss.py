import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEPTH_LOSS_WEIGHT, DISTILL_WEIGHT


class DistillationLoss(nn.Module):
    """
    Combined loss module for depth supervision and feature-based distillation,
    with learned 1×1 channel projectors to align teacher → student features.
    """
    def __init__(
        self,
        student_channels: list[int],
        teacher_channels: list[int],
    ):
        super().__init__()
        # 1×1 conv projectors from teacher channels to student channels
        self.projectors = nn.ModuleList([
            nn.Conv2d(ct, cs, kernel_size=1, bias=False)
            for ct, cs in zip(teacher_channels, student_channels)
        ])
        self.depth_crit = nn.MSELoss()

    def compute_depth_loss(
        self,
        pred_depth: torch.Tensor,
        gt_depth:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Resizes teacher depth to match student resolution if needed,
        then computes MSE.
        """
        if gt_depth.shape[2:] != pred_depth.shape[2:]:
            gt_depth = F.interpolate(
                gt_depth,   
                size=pred_depth.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        assert pred_depth.shape == gt_depth.shape
        return self.depth_crit(pred_depth, gt_depth) * DEPTH_LOSS_WEIGHT

    def compute_feature_loss(
        self,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Projects teacher features via 1×1 conv, resizes spatially,
        and applies MSE to each student-teacher pair.
        """
        feat_loss = torch.tensor(0.0, device=student_feats[0].device)
        # iterate through each level
        for idx, (S, T) in enumerate(zip(student_feats, teacher_feats)):
            # 1) detach teacher
            T = T.detach()
            # 2) channel project teacher -> student dims
            T = self.projectors[idx](T)
            # 3) dtype align
            T = T.to(S.dtype)
            # 4) spatial align if needed
            if T.shape[2:] != S.shape[2:]:
                T = F.interpolate(
                    T,
                    size=S.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            # 5) MSE
            feat_loss = feat_loss + F.mse_loss(S, T)
        return feat_loss * DISTILL_WEIGHT

    def forward(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes combined depth and feature distillation loss.

        Returns:
            total: combined loss tensor
            logs: dict of individual loss components
        """
        dloss = self.compute_depth_loss(pred_depth, gt_depth)
        #floss = self.compute_feature_loss(student_feats, teacher_feats)
        total = dloss # + floss
        return total, {"depth_loss": dloss.item()} #, "feat_loss": floss.item()}
