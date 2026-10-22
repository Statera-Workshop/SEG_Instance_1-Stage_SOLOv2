"""
SOLOv2 (ResNet-50-FPN) builder - box-free instance segmentation by location. Seg task.

Paper: Wang, Zhang, Kong, Li & Shen, "SOLOv2: Dynamic and Fast Instance Segmentation" (NeurIPS 2020, arXiv 2003.10152).
Code : WXinlong/SOLO (the authors' code, mmdetection-based): mmdet/models/anchor_heads/solov2_head.py,
       mask_feature_heads/mask_feat_head.py, configs/solov2/solov2_r50_fpn_8gpu_1x.py; reimplemented here.

    backbone     ResNet-50 + FPN P2-P6 (torchvision resnet_fpn_backbone, as detectors/faster_rcnn); P2 is
                 downsampled 2x and P6 upsampled to P5 (split_feats) -> strides 8, 8, 16, 32, 32
    grids        S = 40 / 36 / 24 / 16 / 12, scale ranges sqrt(wh) (1,96) (48,192) (96,384) (192,768) (384,2048)
    head         per level, feature + 2 coordinate channels resized to S x S;
                 kernel branch 4 x (conv 3x3 512 + GN + ReLU) -> conv 3x3 -> E = 256 (1x1 dynamic kernels),
                 category branch (same feature without coordinates) 4 x (conv 3x3 512 + GN + ReLU) -> conv 3x3 C
                 (focal prior 0.01)
    mask feature P2-P5 -> conv 3x3 128 + GN + ReLU and 2x upsampling to stride 4 (P5 with coordinates), summed
                 -> conv 1x1 256 + GN + ReLU  (unified mask feature F)
    targets      per object: level by sqrt(wh); grid cells inside the centre region (centre of mass +-
                 sigma 0.2 x half size, at most the 3 x 3 cells around the centre cell) take its class and mask
    loss         dice (weight 3) on sigmoid(kernel * F) of the positive cells vs the mask at stride 4 +
                 sigmoid focal loss (alpha 0.25, gamma 2) on the categories / (number of positives + 1)
    inference    category maps with points-NMS (2 x 2 max pooling, local maxima kept); mask = sigmoid(kernel * F);
                 score x maskness (mean probability inside the 0.5 mask)

Raw contract: one row per grid cell (3,872 at 608); the box of a row is the tight box of its predicted mask,
computed for the 300 best cells (others get score 0); decode_masks kind "full" (stride 4).

[Unified] no ImageNet weights, trainable BN + PreciseBN, common init, no mean / std, 608 input, Adam 1e-4,
YOLO-paper augmentation, common post-processing with box NMS on the mask boxes instead of Matrix NMS (the
contract is box based); the official mask-area filter (mask area > stride) is replaced by dropping empty masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import sigmoid_focal_loss

from detectors.anchor_free.common import HeadStats, init_prior_bias
from detectors.condinst.build import dice_loss
from detectors.yolo11_seg.build import split_seg_targets

GRIDS = (40, 36, 24, 16, 12)
SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
SIGMA, E, TOWER, MASK_CH = 0.2, 256, 512, 128
DICE_W, TOPK_CELLS = 3.0, 300


def gn_relu(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.GroupNorm(32, cout),
                         nn.ReLU(inplace=True))


def coords_like(x):
    b, _, h, w = x.shape
    xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
    ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
    return torch.cat([xs, ys], 1)


class MaskFeatHead(nn.Module):
    def __init__(self, cin=256):
        super().__init__()
        self.levels = nn.ModuleList()
        for lvl in range(4):
            if lvl == 0:
                self.levels.append(gn_relu(cin, MASK_CH))
                continue
            mods = []
            for j in range(lvl):
                mods += [gn_relu((cin + 2 if lvl == 3 else cin) if j == 0 else MASK_CH, MASK_CH),
                         nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)]
            self.levels.append(nn.Sequential(*mods))
        self.pred = gn_relu(MASK_CH, E, k=1)

    def forward(self, feats):                     # P2-P5
        x = self.levels[0](feats[0])
        for lvl in range(1, 4):
            f = feats[lvl]
            if lvl == 3:
                f = torch.cat([f, coords_like(f)], 1)
            y = self.levels[lvl](f)
            x = x + y[:, :, :x.shape[2], :x.shape[3]]
        return self.pred(x)


class SOLOv2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None, norm_layer=nn.BatchNorm2d,
                                            trainable_layers=5)
        self.kernel_tower = nn.Sequential(*[gn_relu(256 + 2 if i == 0 else TOWER, TOWER) for i in range(4)])
        self.cate_tower = nn.Sequential(*[gn_relu(256 if i == 0 else TOWER, TOWER) for i in range(4)])
        self.kernel_pred = nn.Conv2d(TOWER, E, 3, padding=1)
        self.cate_pred = nn.Conv2d(TOWER, num_classes, 3, padding=1)
        init_prior_bias(self.cate_pred, 0.01)
        self.mask_feat = MaskFeatHead()
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def _heads(self, x):
        p = list(self.backbone(x).values())                       # P2, P3, P4, P5, P6(pool)
        split = [F.interpolate(p[0], scale_factor=0.5, mode="bilinear", align_corners=False), p[1], p[2], p[3],
                 F.interpolate(p[4], size=p[3].shape[-2:], mode="bilinear", align_corners=False)]
        kernels, cates = [], []
        B = x.shape[0]
        for f, S in zip(split, GRIDS):
            k_in = F.interpolate(torch.cat([f, coords_like(f)], 1), size=(S, S), mode="bilinear", align_corners=False)
            kernels.append(self.kernel_pred(self.kernel_tower(k_in)).permute(0, 2, 3, 1).reshape(B, -1, E))
            cates.append(self.cate_pred(self.cate_tower(k_in[:, :-2])))
        return torch.cat(kernels, 1), cates, self.mask_feat(p[:4])

    @staticmethod
    def masks_from(kernels, mask_feat):
        """kernels [N, E], mask_feat [E, h, w] -> logits [N, h, w]."""
        E_, h, w = mask_feat.shape
        return (kernels @ mask_feat.view(E_, -1)).view(-1, h, w)

    def forward(self, x, targets=None):
        kernels, cates, mask_feat = self._heads(x)
        B, S_in = x.shape[0], x.shape[2]
        cate_flat = torch.cat([c.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes) for c in cates], 1)
        if targets is None:
            if self.training:                                     # PreciseBN forward
                return cate_flat
            return self.to_raw(kernels, cates, mask_feat, S_in)

        tgt = split_seg_targets(targets, B, S_in)
        cate_t, pos, gt_masks = self.targets(tgt, S_in, mask_feat.shape[-2:])
        num_pos = int(pos.sum())
        onehot = torch.zeros_like(cate_flat)
        if num_pos:
            onehot[pos] = F.one_hot(cate_t[pos], self.num_classes).float()
        loss_cate = sigmoid_focal_loss(cate_flat, onehot, alpha=0.25, gamma=2.0, reduction="sum") / (num_pos + 1)
        if num_pos:
            logits = []
            for b in range(B):
                if pos[b].any():
                    logits.append(self.masks_from(kernels[b, pos[b]], mask_feat[b]))
            loss_ins = self._mask_loss(torch.cat(logits), torch.cat(gt_masks))
        else:
            loss_ins = kernels.sum() * 0.0 + mask_feat.sum() * 0.0
        loss = loss_cate + loss_ins
        flat = cate_flat.detach().reshape(-1, self.num_classes)
        self.stats.update({"loss_cate": loss_cate, "loss_ins": loss_ins, "losses": loss},
                          flat.sigmoid().max(-1).values, flat, pos.view(-1), cate_t[pos])
        return loss, cate_flat.new_zeros(B, 0, 5 + self.num_classes)

    def _mask_loss(self, logits, gt):
        """Official mask loss: dice on sigmoid(logits) x DICE_W. logits / gt [N, h, w] (stride 4).
        Overridden by detectors/solov2_bce (patch 10)."""
        return dice_loss(logits.sigmoid(), gt).mean() * DICE_W

    @torch.no_grad()
    def targets(self, tgt, img_size, mask_hw):
        """solov2_target_single -> cate labels [B, R], positive mask [B, R], per-image gt masks of the positives."""
        B, R = len(tgt), sum(s * s for s in GRIDS)
        dev = tgt[0]["boxes"].device
        cate_t = torch.zeros(B, R, dtype=torch.long, device=dev)
        pos = torch.zeros(B, R, dtype=torch.bool, device=dev)
        gt_out = []
        for b, t in enumerate(tgt):
            cell_gt = torch.full((R,), -1, dtype=torch.long, device=dev)
            n = len(t["labels"])
            if n:
                boxes = t["boxes"]
                wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
                areas = (wh[:, 0] * wh[:, 1]).sqrt()
                m = t["masks"].float()
                ys = torch.arange(img_size, device=dev, dtype=torch.float32)
                mass = m.sum((1, 2)).clamp(min=1)
                cy = (m.sum(2) * ys).sum(1) / mass
                cx = (m.sum(1) * ys).sum(1) / mass
                off = 0
                for S, (lo, hi) in zip(GRIDS, SCALE_RANGES):
                    for g in torch.nonzero((areas >= lo) & (areas <= hi)).squeeze(1).tolist():
                        if m[g].sum() < 1:
                            continue
                        hh, hw = 0.5 * wh[g, 1].item() * SIGMA, 0.5 * wh[g, 0].item() * SIGMA
                        ch, cw = cy[g].item(), cx[g].item()
                        coord_h, coord_w = int(ch / img_size * S), int(cw / img_size * S)
                        top = max(0, int((ch - hh) / img_size * S), coord_h - 1)
                        down = min(S - 1, int((ch + hh) / img_size * S), coord_h + 1)
                        left = max(0, int((cw - hw) / img_size * S), coord_w - 1)
                        right = min(S - 1, int((cw + hw) / img_size * S), coord_w + 1)
                        for i in range(top, down + 1):
                            for j in range(left, right + 1):
                                r = off + i * S + j
                                cate_t[b, r] = t["labels"][g]
                                pos[b, r] = True
                                cell_gt[r] = g
                    off += S * S
            if pos[b].any():
                ds = (F.interpolate(t["masks"][None].float(), size=mask_hw, mode="bilinear",
                                    align_corners=False)[0] > 0.5).float()
                gt_out.append(ds[cell_gt[pos[b]]])
        return cate_t, pos, gt_out

    @torch.no_grad()
    def to_raw(self, kernels, cates, mask_feat, img_size):
        B, C = kernels.shape[0], self.num_classes
        probs = []
        for c in cates:                                        # points_nms
            p = c.sigmoid()
            keep = (F.max_pool2d(p, 2, stride=1, padding=1)[:, :, :-1, :-1] == p).float()
            probs.append((p * keep).permute(0, 2, 3, 1).reshape(B, -1, C))
        probs = torch.cat(probs, 1)                            # [B, R, C]
        R = probs.shape[1]
        raw = probs.new_zeros(B, R, 5 + C)
        h, w = mask_feat.shape[-2:]
        stride = img_size / h
        for b in range(B):
            top = probs[b].max(1).values.topk(min(TOPK_CELLS, R)).indices
            mp = self.masks_from(kernels[b, top], mask_feat[b]).sigmoid()
            seg = mp > 0.5
            area = seg.sum((1, 2)).float()
            maskness = (mp * seg).sum((1, 2)) / area.clamp(min=1)
            any_x, any_y = seg.any(1), seg.any(2)              # [K, w], [K, h]
            ar_w = torch.arange(w, device=mp.device).float()
            ar_h = torch.arange(h, device=mp.device).float()
            x0 = torch.where(any_x, ar_w, ar_w.new_tensor(1e9)).min(1).values
            x1 = torch.where(any_x, ar_w, ar_w.new_tensor(-1.0)).max(1).values + 1
            y0 = torch.where(any_y, ar_h, ar_h.new_tensor(1e9)).min(1).values
            y1 = torch.where(any_y, ar_h, ar_h.new_tensor(-1.0)).max(1).values + 1
            valid = area > 0
            box = torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], 1) * stride
            raw[b, top[valid], :4] = box[valid]
            raw[b, top[valid], 4] = 1.0
            raw[b, top[valid], 5:] = probs[b, top[valid]] * maskness[valid, None]
        return raw, {"kernels": kernels, "mask_feat": mask_feat}

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        return {"kind": "full", "prob": self.masks_from(aux["kernels"][b, rows], aux["mask_feat"][b]).sigmoid()}

    def load_darknet_weights(self, path):
        raise NotImplementedError("SOLOv2 has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    return SOLOv2(num_classes)
