"""
CondInst (ResNet-50-FPN) builder - instance segmentation with dynamic (conditional) mask heads. Seg task.

Paper: Tian, Shen & Chen, "Conditional Convolutions for Instance Segmentation" (ECCV 2020, arXiv 2003.05664).
Code : AdelaiDet (aim-uofa, the authors' code; detectron2-based, approved source family) adet/modeling/condinst/
       condinst.py, dynamic_mask_head.py, mask_branch.py, configs/CondInst/Base-CondInst.yaml; reimplemented here
       on top of the detection stage's FCOS (detectors/fcos, unchanged box part).

    detector     FCOS R50-FPN P3-P7 (detectors/fcos.FCOS: GN head, centerness, center sampling, focal + GIoU)
    controller   conv 3x3 on the FCOS box tower -> 169 parameters per location (std 0.01 init, bias 0)
    mask branch  P3, P4, P5 -> conv 3x3 128 + BN + ReLU each, upsampled to P3 (aligned bilinear) and summed
                 -> 4 x (conv 3x3 128 + BN + ReLU) -> conv 1x1 8 channels (stride 8)
    mask head    per instance: [8 mask features, 2 relative coordinates / size of interest] -> 1x1 conv 8 + ReLU
                 -> 1x1 conv 8 + ReLU -> 1x1 conv 1 (weights 80 + 64 + 8, biases 8 + 8 + 1 = 169), sizes of
                 interest 64 / 128 / 256 / 512 / 1024 for P3-P7; logits upsampled 2x (aligned bilinear) to stride 4
    loss         FCOS losses + dice loss on sigmoid masks vs ground truth sampled at stride 4 (offset 2),
                 averaged over the positive locations (max 500 sampled per batch, MAX_PROPOSALS), weight 1
    inference    the controller parameters of each kept detection build its mask (dynamic head on the shared
                 mask features), sigmoid, cropped with the box, threshold 0.5 (tasks/seg_task.py)

Raw contract: FCOS rows [B, R, 5 + C] (one per location); aux = mask features + parameters + locations.

[Unified] as FCOS (no ImageNet weights, trainable BN + PreciseBN, common init, no mean / std, 608 input, Adam 1e-4,
common post-processing); semantic auxiliary loss off (Base-CondInst default); instance locations are the FCOS
anchor centres of the framework (detectron2 anchor offset 0), mask-feature locations x * 8 + 4 (AdelaiDet
compute_locations); the predicted mask is also cropped with its box at inference (common seg post-processing).
"""


CH, LAYERS, IN_CH = 8, 3, 10
WEIGHT_NUMS, BIAS_NUMS = [IN_CH * CH, CH * CH, CH], [CH, CH, 1]
MASK_STRIDE, MASK_OUT_STRIDE, MAX_PROPOSALS = 8, 4, 500


def dice_loss(x, target, eps=1e-5):
    """AdelaiDet dice_coefficient: 1 - 2 |x t| / (|x|^2 + |t|^2), per instance."""
    x, target = x.flatten(1), target.flatten(1)
    return 1.0 - 2 * (x * target).sum(1) / ((x * x).sum(1) + (target * target).sum(1) + eps)
