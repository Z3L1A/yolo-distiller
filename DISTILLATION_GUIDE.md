# Knowledge Distillation Guide — YOLO Distiller

Practical recommendations for using knowledge distillation with YOLO models. Covers all supported distillation methods: **CWD**, **MGD**, **FGD**, and **logit**.

---

## Quick Start

```bash
# Basic distillation command
yolo train model=yolov8n.pt data=coco.yaml epochs=100 \
    teacher=yolov8l.pt \
    distillation_loss=fgd \
    distill_weight=1.0 \
    distill_warmup_epochs=3
```

---

## Method Comparison

| Method | Type | What it Transfers | GT Boxes Required | Extra Learnable Params | Best For |
|--------|------|-------------------|-------------------|----------------------|----------|
| **CWD** | Feature-level | Channel-wise probability distributions (KL divergence) | No | Alignment convs only | General-purpose; semantically rich features |
| **MGD** | Feature-level | Masked feature reconstruction (MSE) | No | Alignment convs + generation convs | When teacher/student architectures differ significantly |
| **FGD** | Feature-level | Focal (fg/bg-aware) + global (relation) features | Yes | Alignment convs + attention convs + relation bottlenecks | Dense detection; datasets with many small objects |
| **logit** | Head-level | Detection head outputs (class + bbox distributions) | No (uses TAL fg_mask) | None | Fine-tuning; when feature access is limited |

### When to Use Each Method

- **CWD** — Good default choice. Works well across architectures. Low overhead. Start here if unsure.
- **MGD** — Better than CWD when there's a large capacity gap between teacher and student (e.g., YOLOv8x → YOLOv8n). The masked reconstruction forces the student to learn more robust features.
- **FGD** — Best when GT bounding boxes are reliable and the dataset has complex scenes with many objects. The foreground/background separation prevents the student from wasting capacity on background noise. Particularly effective for dense detection scenarios.
- **logit** — Lightweight option that only distills detection head outputs. Best for fine-tuning a student that already has a decent backbone. Avoids feature-level hooks entirely.

---

## Key Training Parameters

### `distillation_loss` (str)
The distillation method to use. Options: `'cwd'`, `'mgd'`, `'fgd'`, `'logit'`.

### `distill_weight` (float, default: 1.0)
Multiplier for the distillation loss before adding to the main detection loss.

**Recommendations by method:**

| Method | Recommended `distill_weight` | Notes |
|--------|------------------------------|-------|
| CWD | 0.5 – 2.0 | Start with 1.0; increase if student underfits |
| MGD | 0.5 – 2.0 | Start with 1.0 |
| FGD | 0.5 – 2.0 | Start with 1.0; FGD's internal sub-loss weights (α, β, γ, λ) are already calibrated |
| logit | 0.5 – 1.5 | Often needs lower weight to avoid overpowering main loss |

**Tuning strategy**: Monitor the `d_ratio` metric in training logs. This shows the ratio of distillation loss to main detection loss. Ideally it should be **0.1–0.5** (distillation is a regularizer, not the primary objective). If `d_ratio > 1.0`, reduce `distill_weight`. If `d_ratio < 0.01`, increase it.

### `distill_warmup_epochs` (int, default: 3)
Number of epochs to gradually ramp up distillation loss after the main warmup ends. This prevents the distillation signal from destabilizing early training when the student's features are still random.

**Recommendations:**
- Short training (≤50 epochs): Use 1–2 epochs
- Standard training (100 epochs): Use 3–5 epochs  
- Long training (300+ epochs): Use 5–10 epochs

### `teacher` (str or model)
Path to the teacher model weights. Must be a YOLO model of the **same task** (detection, segmentation, etc.) and trained on a **compatible dataset** (same class set).

---

## Teacher–Student Pairing

### Architecture Compatibility

The teacher and student must use the **same task head** (Detect, Segment, Pose, etc.). The backbone and neck can differ in depth/width.

**Recommended pairings:**

| Teacher | Student | Method | Expected mAP Gain |
|---------|---------|--------|-------------------|
| YOLOv8l | YOLOv8n | CWD/FGD | +1.5 – 3.0 |
| YOLOv8x | YOLOv8s | CWD/FGD | +1.0 – 2.5 |
| YOLOv8l | YOLOv8s | CWD/MGD | +1.0 – 2.0 |
| YOLOv8m | YOLOv8n | CWD/FGD | +1.0 – 2.0 |
| YOLOv8x | YOLOv8n | MGD | +1.5 – 3.0 |
| YOLOv8l | YOLOv8l (pruned) | logit | +0.5 – 1.5 |

**Guidelines:**
- Larger capacity gaps benefit more from feature-level distillation (CWD, MGD, FGD)
- Same-family models (e.g., both YOLOv8) work best — architecture alignment at hook layers is more natural
- Cross-family distillation (e.g., YOLOv8x teacher → YOLOv5n student) may work but requires careful hook layer matching
- The teacher should be **significantly better** than the student (≥3-5 mAP points) for distillation to help

### Teacher Quality Matters

- **Always use the best teacher you can afford** — distillation can only transfer what the teacher knows
- A poorly trained teacher will transfer bad knowledge and can **hurt** student performance
- Prefer teachers trained to convergence on the same dataset

---

## Pretrained vs. From-Scratch Training

### Pretrained Student (Recommended)

```bash
# Start from pretrained weights (default, recommended)
yolo train model=yolov8n.pt data=coco.yaml epochs=100 \
    teacher=yolov8l.pt distillation_loss=fgd
```

**Why pretrained is better for distillation:**
- The student already has reasonable features → distillation fine-tunes them toward the teacher
- Faster convergence (fewer epochs needed)
- The pretrained backbone provides a good initialization that distillation can improve upon
- Less sensitive to `distill_weight` and `distill_warmup_epochs`

### From-Scratch Student

```bash
# Train from scratch (use .yaml config, not .pt weights)
yolo train model=yolov8n.yaml data=coco.yaml epochs=300 \
    teacher=yolov8l.pt distillation_loss=fgd \
    distill_warmup_epochs=10 distill_weight=0.5
```

**When training from scratch:**
- Use **longer `distill_warmup_epochs`** (5–10) — the student's random features need time to stabilize before distillation helps
- Use **lower `distill_weight`** (0.3–0.8) initially — don't overwhelm the main detection loss
- Train for **more epochs** (200–300 vs. 100 for pretrained)
- **FGD** and **CWD** work best from scratch because they provide structured feature-level guidance
- **logit** distillation is less effective from scratch (the student needs reasonable features before head-level matching helps)

### Fine-Tuning with Distillation

```bash
# Fine-tune on custom dataset with distillation
yolo train model=yolov8n.pt data=custom.yaml epochs=50 \
    teacher=yolov8l_custom.pt distillation_loss=cwd \
    distill_weight=1.0 distill_warmup_epochs=2
```

Fine-tuning is the sweet spot for distillation:
- The teacher is trained on your target domain → maximally relevant knowledge
- Short training schedule → distillation has the biggest relative impact
- All methods work well; CWD and FGD are most reliable

---

## FGD-Specific Recommendations

### When FGD Shines
- **Dense object scenes** — many objects per image (COCO, VisDrone, DOTA)
- **Small objects** — FGD's focal mechanism ensures small object features aren't drowned out by background
- **Well-annotated datasets** — FGD relies on GT bounding boxes for fg/bg separation; noisy labels will hurt

### When to Avoid FGD
- **Datasets with noisy/missing annotations** — incorrect GT boxes create wrong fg/bg masks → misleading distillation signal
- **Very few objects per image** — the fg/bg separation provides less benefit when most of the image is background anyway
- **Memory-constrained training** — FGD has more learnable parameters (conv_mask, channel_add_conv per level) than CWD

### FGD Internal Parameters
The default sub-loss weights are calibrated from the paper and generally don't need tuning:

| Parameter | Default | Increase When... | Decrease When... |
|-----------|---------|-------------------|------------------|
| `alpha_fgd` (fg_loss) | 0.001 | Foreground precision matters most | Background objects are important too |
| `beta_fgd` (bg_loss) | 0.0005 | Background suppression is critical | Focusing only on foreground |
| `gamma_fgd` (mask_loss) | 0.001 | Student's attention is very different from teacher's | Attention is already well-aligned |
| `lambda_fgd` (rela_loss) | 0.000005 | Global context matters (large images, many objects) | Simple scenes |
| `temp` | 0.5 | Want softer/smoother attention maps | Want sharper/more focused attention |

These are hardcoded in `FGDLoss.__init__()`. To customize, modify the defaults in the class or subclass it.

---

## Training Tips

### Monitoring Distillation Health

Watch these metrics during training:

1. **`d_loss`** — Distillation loss value (logged per epoch). Should decrease over time.
2. **`d_ratio`** — Ratio of distillation loss to main detection loss. Target: 0.1–0.5.
3. **Main mAP** — Compare against a no-distillation baseline to measure actual improvement.

### Common Issues and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| mAP worse than baseline | `distill_weight` too high | Reduce to 0.3–0.5 |
| `d_loss` not decreasing | Features too mismatched | Use longer `distill_warmup_epochs`; try MGD (more forgiving) |
| `d_loss` is 0 or NaN | Hook registration failure | Check that teacher/student have matching architecture family |
| Training OOM | FGD extra params + feature storage | Reduce batch size; try CWD (fewer params) |
| `d_ratio` > 2.0 | Distillation overwhelming main loss | Reduce `distill_weight` or increase `distill_warmup_epochs` |
| mAP plateaus early | Student saturated | Teacher may not be good enough; try a larger teacher |
| Slow convergence | Feature-level mismatch | Switch to logit distillation for head-only matching |

### Combining with Other Techniques

- **Mosaic augmentation**: Works well with all distillation methods. The `close_mosaic` parameter disables mosaic for the last N epochs — distillation continues normally.
- **Multi-GPU (DDP)**: Fully supported. Teacher is automatically wrapped in DDP.
- **Mixed precision (AMP)**: Fully supported. Feature hooks capture at native precision.
- **Pruning + Distillation**: Train the pruned student with the unpruned model as teacher. Use logit distillation first, then CWD/FGD for maximum recovery.

---

## Example Commands

### CWD — Default feature distillation
```bash
yolo train model=yolov8n.pt data=coco.yaml epochs=100 \
    teacher=yolov8l.pt distillation_loss=cwd \
    distill_weight=1.0 distill_warmup_epochs=3
```

### MGD — Masked feature distillation
```bash
yolo train model=yolov8n.pt data=coco.yaml epochs=100 \
    teacher=yolov8l.pt distillation_loss=mgd \
    distill_weight=1.0 distill_warmup_epochs=3
```

### FGD — Focal and global distillation
```bash
yolo train model=yolov8n.pt data=coco.yaml epochs=100 \
    teacher=yolov8l.pt distillation_loss=fgd \
    distill_weight=1.0 distill_warmup_epochs=3
```

### Logit — Head-level distillation
```bash
yolo train model=yolov8n.pt data=coco.yaml epochs=100 \
    teacher=yolov8l.pt distillation_loss=logit \
    distill_weight=0.5 distill_warmup_epochs=2
```

### From-Scratch with FGD
```bash
yolo train model=yolov8n.yaml data=coco.yaml epochs=300 \
    teacher=yolov8l.pt distillation_loss=fgd \
    distill_weight=0.5 distill_warmup_epochs=10
```

### Fine-tune Custom Dataset
```bash
# First train the teacher
yolo train model=yolov8l.pt data=custom.yaml epochs=100

# Then distill to student
yolo train model=yolov8n.pt data=custom.yaml epochs=50 \
    teacher=runs/detect/train/weights/best.pt \
    distillation_loss=fgd distill_weight=1.0
```
