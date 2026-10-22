"""
SOLOv2 + BCE mask loss (2026-10-14, patch 10) - same model as detectors/solov2, only the mask loss differs.

Why: trained from scratch on FL36-Seg, the official SOLOv2 (dice only) collapsed early to "every pixel 0"
(mask max 0.000 at evaluation, AP 0 from epoch 5; category branch fine, cls_acc 83%). Letters cover only
0.1-2 % of the stride-4 mask, so pushing all predictions to 0 is the fastest early dice decrease, and dice is
computed on sigmoid(logits): once the logits are very negative, sigmoid' ~ 0 and the mask branch gets no
gradient back. BCE on the LOGITS keeps a gradient of (p - 1) ~ -1 on every letter pixel however negative the
logit is, so the collapse can be left.

[Unified] mask loss = dice x 3 (official, unchanged) + BCE_W x per-instance mean BCE-with-logits, on the same
positive cells and the same stride-4 gt masks. Everything else (backbone, heads, targets, category focal loss,
inference, training recipe, from-scratch protocol) is detectors/solov2 unchanged.
"""

import torch.nn.functional as F

from detectors.solov2.build import SOLOv2

BCE_W = 1.0


class SOLOv2BCE(SOLOv2):
    def _mask_loss(self, logits, gt):
        dice = super()._mask_loss(logits, gt)
        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none").mean((1, 2)).mean()  # [Unified]
        return dice + BCE_W * bce


def build_model(task, num_classes):
    return SOLOv2BCE(num_classes)
