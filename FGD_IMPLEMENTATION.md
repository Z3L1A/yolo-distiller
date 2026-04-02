# FGD (Focal and Global Distillation) — Implementation Details

## Paper Reference

## Focal and Global Knowledge Distillation for Detectors

- Authors: Zhendong Yang, Zhe Li, Xiaohu Jiang, Yuan Gong, Zehuan Yuan, Danpei Zhao, Chun Yuan
- Venue: **CVPR 2022**
- arXiv: [2111.11837](https://arxiv.org/abs/2111.11837)
- Official code: [github.com/yzd-v/FGD](https://github.com/yzd-v/FGD)

## Motivation

Standard feature-based knowledge distillation (e.g., FitNets, CWD) treats all spatial locations equally when transferring knowledge from teacher to student. However, in object detection:

1. **Foreground and background regions have very different feature distributions** — the teacher's features in foreground (object) areas carry crucial detection knowledge, while background features are mostly noise.
2. **Uniform distillation creates conflicting gradients** — forcing the student to match the teacher everywhere dilutes the signal from important regions.
3. **Global structural relationships** between different spatial positions (context) are lost when only local feature matching is performed.

FGD addresses both issues with two complementary components:

- **Focal Distillation**: Separates foreground/background using GT bounding boxes and weights the feature matching by teacher's own attention (spatial + channel), forcing the student to focus on what the teacher considers important.
- **Global Distillation**: Uses GcBlock-style spatial pooling to capture global context relationships and transfers them from teacher to student.

## Algorithm Overview

### Input

- Student feature maps: `F_s` of shape `(N, C, H, W)` per feature level
- Teacher feature maps: `F_t` of shape `(N, C, H, W)` per feature level
- Ground-truth bounding boxes per image (pixel coordinates)
- Input image dimensions `(H_img, W_img)`

### Step 1 — Attention Computation

For both student and teacher features, compute:

**Spatial Attention** `S(F)`:

```
fea_map = mean(|F|, dim=C)                    → (N, 1, H, W)
S(F) = H × W × softmax(fea_map / τ)           → (N, H, W)
```

Highlights which spatial positions are most activated.

**Channel Attention** `C(F)`:

```
channel_map = mean(|F|, dim=(H,W))             → (N, C)
C(F) = C × softmax(channel_map / τ)            → (N, C)
```

Highlights which channels carry the most information.

Temperature `τ` (default 0.5) controls the sharpness of attention distributions.

### Step 2 — Foreground/Background Mask Construction

From GT bounding boxes, build per-feature-level masks:

- **`Mask_fg`** `(N, H, W)`: For each GT box, scale coordinates to feature map resolution. Fill the box region with inverse-area weight `1 / (box_h × box_w)`. Overlapping boxes take the maximum weight.
- **`Mask_bg`** `(N, H, W)`: Binary complement of `Mask_fg` (1 where no GT boxes exist), normalized to sum to 1.

### Step 3 — Focal Loss (Foreground + Background)

Weight both student and teacher features by the **teacher's** spatial and channel attention, then apply fg/bg masks:

```
fea = F × √(S_t) × √(C_t)          # attention-weighted features
fg_fea = fea × √(Mask_fg)           # foreground-masked
bg_fea = fea × √(Mask_bg)           # background-masked

fg_loss = MSE(fg_fea_s, fg_fea_t) / N
bg_loss = MSE(bg_fea_s, bg_fea_t) / N
```

Key insight: Using the **teacher's** attention (not the student's) as weights ensures the student learns to attend to the same regions the teacher considers important.

### Step 4 — Attention Alignment Loss

Direct L1 matching of student and teacher attention maps:

```
mask_loss = L1(C_s, C_t) / N + L1(S_s, S_t) / N
```

This explicitly teaches the student to develop similar attention patterns.

### Step 5 — Global Relation Loss (GcBlock)

Captures global context via spatial pooling:

```
context = softmax(conv_1x1(F)) ⊗ F     # weighted spatial pool → (N, C, 1, 1)
out = F + conv_bottleneck(context)       # add global context back
rela_loss = MSE(out_s, out_t) / N
```

- `conv_mask_s/t`: Learnable 1×1 conv to produce spatial attention weights for pooling
- `channel_add_conv_s/t`: Bottleneck (C → C/2 → C) with LayerNorm + ReLU to project pooled context

This loss transfers the teacher's understanding of long-range spatial relationships.

### Step 6 — Total Loss

```
L_FGD = α × fg_loss + β × bg_loss + γ × mask_loss + λ × rela_loss
```

Default weights from the paper:

| Parameter | Symbol | Default | Role |
|-----------|--------|---------|------|
| `alpha_fgd` | α | 0.001 | Foreground feature loss weight |
| `beta_fgd` | β | 0.0005 | Background feature loss weight |
| `gamma_fgd` | γ | 0.001 | Attention alignment loss weight |
| `lambda_fgd` | λ | 0.000005 | Global relation loss weight |
| `temp` | τ | 0.5 | Attention temperature |

## Implementation Architecture

### Files Modified

1. **`ultralytics/engine/trainer.py`** — Core implementation:
   - `FGDLoss(nn.Module)` — The loss class implementing all 4 sub-losses
   - `FeatureLoss` — Updated to support `distiller='fgd'` and pass GT data
   - `DistillationLoss.get_loss()` — Updated to extract GT bboxes from batch and pass to FeatureLoss
   - Training loop — Updated to pass `batch` to `distillation_loss.get_loss(batch=batch)`

2. **`ultralytics/cfg/default.yaml`** — Added `'fgd'` to valid `distillation_loss` options

### Class Hierarchy

```
DistillationLoss (coordinator)
  ├── register_hook()     — Attaches forward hooks to capture intermediate features
  ├── get_loss(batch)     — Extracts GT bboxes, calls FeatureLoss
  └── remove_handle_()    — Cleans up hooks
      │
      └── FeatureLoss(nn.Module) (alignment + dispatch)
            ├── align_module (ModuleList of 1x1 Conv + BN per level)
            └── feature_loss = FGDLoss (or CWDLoss / MGDLoss)
                  │
                  └── FGDLoss(nn.Module)
                        ├── conv_mask_s/t    (ModuleList, per-level 1x1→1 conv for spatial pooling)
                        ├── channel_add_conv_s/t (ModuleList, per-level bottleneck for relation)
                        ├── get_attention()  — Spatial + channel attention
                        ├── get_fea_loss()   — FG/BG masked MSE
                        ├── get_mask_loss()  — Attention alignment L1
                        └── get_rela_loss()  — GcBlock relation MSE
```

### GT Bounding Box Flow

```
batch['bboxes']          (M, 4) normalized xywh (all images concatenated)
batch['batch_idx']       (M,) image index per bbox
         │
         ▼
DistillationLoss.get_loss(batch)
    ├── xywh2xyxy conversion
    ├── Scale to pixel coordinates: × img_w, × img_h
    └── Split per image → list of (n_i, 4) tensors
         │
         ▼
FeatureLoss.forward(y_s, y_t, gt_bboxes=..., img_shape=...)
         │
         ▼
FGDLoss.forward(y_s, y_t, gt_bboxes=..., img_shape=...)
    └── Per feature level: scale GT boxes to (H_feat, W_feat) resolution
        → build Mask_fg, Mask_bg
```

### Learnable Parameters

FGD introduces additional learnable parameters compared to CWD:

| Module | Per Level | Total (6 levels) | Purpose |
|--------|-----------|-------------------|---------|
| `conv_mask_s` | C×1 conv | 6 convs | Student spatial pool |
| `conv_mask_t` | C×1 conv | 6 convs | Teacher spatial pool |
| `channel_add_conv_s` | C→C/2→C bottleneck | 6 bottlenecks | Student relation |
| `channel_add_conv_t` | C→C/2→C bottleneck | 6 bottlenecks | Teacher relation |

These parameters are automatically registered via `nn.ModuleList` and added to the optimizer alongside the channel alignment modules.

### Initialization

Following the original paper:

- `conv_mask_s/t`: Kaiming normal initialization
- `channel_add_conv_s/t`: Last layer zero-initialized (residual connection starts as identity)

## Key Differences from Original Implementation

1. **No separate alignment conv in FGDLoss** — Channel alignment (student→teacher dims) is handled by the parent `FeatureLoss.align_module`, so FGDLoss always receives feature pairs with matching channel dimensions.

2. **YOLO bbox format** — The original uses mmdet-style pixel xyxy with `img_metas`. Our implementation converts from YOLO's normalized xywh format (from `batch['bboxes']`) to pixel xyxy, and scales to feature map resolution inside FGDLoss.

3. **Per-level ModuleLists** — The original creates one `FeatureLoss` per FPN level. Our implementation uses `nn.ModuleList` to hold per-level modules within a single `FGDLoss` instance, matching the existing architecture pattern of CWD/MGD.

4. **Feature hook targets** — Hooks are registered on layers `[6, 8, 13, 16, 19, 22]` (cv2 conv outputs) of both teacher and student, consistent with the existing CWD/MGD distillation infrastructure.

5. **Distillation warmup** — The existing `distill_warmup_epochs` mechanism applies to FGD as well, gradually ramping up the FGD loss after the main training warmup period.

## References

- [1] Yang et al., "Focal and Global Knowledge Distillation for Detectors", CVPR 2022
- [2] Shu et al., "Channel-wise Knowledge Distillation for Dense Prediction", ICCV 2021 (CWD — existing implementation)
- [3] Yang et al., "Masked Generative Distillation", ECCV 2022 (MGD — existing implementation)
- [4] Cao et al., "GCNet: Non-Local Networks Meet Squeeze-Excitation Networks and Beyond", ICCVW 2019 (GcBlock used in global distillation)
