# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import csv
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_. The loss is calculated as the KL divergence between the student and teacher feature maps, normalized by the number of channels and batch size. The temperature parameter `tau` is used to soften the probability distributions, which can help improve the distillation process.
    """

    def __init__(self, channels_s, channels_t, tau=1.0):
        super().__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            # normalize in channel dimension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)
        return loss

class MGDLoss(nn.Module):
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)
            ) for channel in teacher_channels
        ])

    def forward(self, y_s, y_t, layer=None):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # print(s.shape)
            # print(t.shape)
            # assert s.shape == t.shape
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss


class FGDLoss(nn.Module):
    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
    <https://arxiv.org/abs/2111.11837> (CVPR 2022).

    Combines focal distillation (foreground/background-aware feature matching weighted
    by teacher attention) with global distillation (GcBlock-based relation transfer).

    Four sub-losses:
        - fg_loss:   MSE on foreground features weighted by teacher spatial+channel attention
        - bg_loss:   MSE on background features weighted by teacher spatial+channel attention
        - mask_loss: L1 between student/teacher spatial and channel attention maps
        - rela_loss: GcBlock spatial-pooling relation loss (MSE)
    """

    def __init__(self, channels_t, temp=0.5, alpha_fgd=0.001, beta_fgd=0.0005,
                 gamma_fgd=0.001, lambda_fgd=0.000005):
        super(FGDLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        # Per-level learnable modules for spatial pooling (relation loss)
        self.conv_mask_s = nn.ModuleList()
        self.conv_mask_t = nn.ModuleList()
        self.channel_add_conv_s = nn.ModuleList()
        self.channel_add_conv_t = nn.ModuleList()

        for ch in channels_t:
            self.conv_mask_s.append(nn.Conv2d(ch, 1, kernel_size=1))
            self.conv_mask_t.append(nn.Conv2d(ch, 1, kernel_size=1))
            self.channel_add_conv_s.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, kernel_size=1),
                nn.LayerNorm([ch // 2, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, ch, kernel_size=1),
            ))
            self.channel_add_conv_t.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, kernel_size=1),
                nn.LayerNorm([ch // 2, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 2, ch, kernel_size=1),
            ))

        self.reset_parameters()

    def forward(self, y_s, y_t, gt_bboxes=None, img_shape=None):
        """Forward computation.
        Args:
            y_s (list): Student feature maps, each (N, C, H, W).
            y_t (list): Teacher feature maps, each (N, C, H, W).
            gt_bboxes (list[Tensor]): Per-image GT boxes in pixel xyxy, len=N.
            img_shape (tuple): (H, W) of the input image.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            N, C, H, W = s.shape

            S_attention_t, C_attention_t = self.get_attention(t, self.temp)
            S_attention_s, C_attention_s = self.get_attention(s, self.temp)

            # Build foreground/background masks from GT bboxes
            Mask_fg = torch.zeros(N, H, W, device=s.device, dtype=s.dtype)
            Mask_bg = torch.ones(N, H, W, device=s.device, dtype=s.dtype)

            if gt_bboxes is not None and img_shape is not None:
                img_h, img_w = img_shape
                for i in range(N):
                    if gt_bboxes[i].numel() == 0:
                        continue
                    bboxes = gt_bboxes[i]  # (nt, 4) in pixel xyxy
                    # Scale bboxes to feature map resolution
                    wmin = torch.floor(bboxes[:, 0] / img_w * W).int().clamp(0, W - 1)
                    wmax = torch.ceil(bboxes[:, 2] / img_w * W).int().clamp(0, W - 1)
                    hmin = torch.floor(bboxes[:, 1] / img_h * H).int().clamp(0, H - 1)
                    hmax = torch.ceil(bboxes[:, 3] / img_h * H).int().clamp(0, H - 1)

                    # Inverse-area weighting per GT box
                    area = 1.0 / ((hmax - hmin + 1).float() * (wmax - wmin + 1).float())
                    for j in range(len(bboxes)):
                        Mask_fg[i, hmin[j]:hmax[j]+1, wmin[j]:wmax[j]+1] = torch.maximum(
                            Mask_fg[i, hmin[j]:hmax[j]+1, wmin[j]:wmax[j]+1], area[j]
                        )

                    Mask_bg[i] = torch.where(Mask_fg[i] > 0, torch.tensor(0.0, device=s.device), torch.tensor(1.0, device=s.device))
                    if Mask_bg[i].sum() > 0:
                        Mask_bg[i] /= Mask_bg[i].sum()

            fg_loss, bg_loss = self.get_fea_loss(s, t, Mask_fg, Mask_bg,
                                                  C_attention_s, C_attention_t,
                                                  S_attention_s, S_attention_t)
            mask_loss = self.get_mask_loss(C_attention_s, C_attention_t,
                                           S_attention_s, S_attention_t)
            rela_loss = self.get_rela_loss(s, t, idx)

            loss = (self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss
                    + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss)
            losses.append(loss)

        return sum(losses)

    def get_attention(self, preds, temp):
        """Compute spatial and channel attention maps.
        Args:
            preds: (N, C, H, W) feature tensor.
            temp: Temperature for softmax.
        Returns:
            S_attention: (N, H, W) spatial attention.
            C_attention: (N, C) channel attention.
        """
        N, C, H, W = preds.shape
        value = torch.abs(preds)

        # Spatial attention: mean over channels → softmax over spatial
        fea_map = value.mean(dim=1, keepdim=True)  # (N, 1, H, W)
        S_attention = (H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # Channel attention: mean over spatial → softmax over channels
        channel_map = value.mean(dim=2).mean(dim=2)  # (N, C)
        C_attention = C * F.softmax(channel_map / temp, dim=1)

        return S_attention, C_attention

    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg,
                     C_s, C_t, S_s, S_t):
        """Compute foreground and background feature losses.

        Normalized by N*C to keep spatial summation (important for
        foreground/background weighting) while being robust to different
        feature map resolutions across architecture variants.
        """
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_S.shape

        Mask_fg = Mask_fg.unsqueeze(dim=1)  # (N, 1, H, W)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(-1).unsqueeze(-1)  # (N, C, 1, 1)
        S_t = S_t.unsqueeze(dim=1)  # (N, 1, H, W)

        # Weight features by teacher attention
        fea_t = torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t) / (N * C)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t) / (N * C)

        return fg_loss, bg_loss

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        """L1 loss between student/teacher attention maps."""
        mask_loss = (torch.sum(torch.abs(C_s - C_t)) / C_s.numel()
                     + torch.sum(torch.abs(S_s - S_t)) / S_s.numel())
        return mask_loss

    def spatial_pool(self, x, idx, in_type):
        """GcBlock-style spatial pooling for relation loss."""
        batch, channel, height, width = x.size()
        input_x = x.view(batch, channel, height * width).unsqueeze(1)  # (N, 1, C, H*W)

        if in_type == 0:
            context_mask = self.conv_mask_s[idx](x)
        else:
            context_mask = self.conv_mask_t[idx](x)

        context_mask = context_mask.view(batch, 1, height * width)  # (N, 1, H*W)
        context_mask = F.softmax(context_mask, dim=2).unsqueeze(-1)  # (N, 1, H*W, 1)
        context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)  # (N, C, 1, 1)
        return context

    def get_rela_loss(self, preds_S, preds_T, idx):
        """GcBlock relation loss between student and teacher features."""
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_S.shape

        context_s = self.spatial_pool(preds_S, idx, 0)
        context_t = self.spatial_pool(preds_T, idx, 1)

        out_s = preds_S + self.channel_add_conv_s[idx](context_s)
        out_t = preds_T + self.channel_add_conv_t[idx](context_t)

        rela_loss = loss_mse(out_s, out_t) / (N * C)
        return rela_loss

    def reset_parameters(self):
        for i in range(len(self.conv_mask_s)):
            nn.init.kaiming_normal_(self.conv_mask_s[i].weight, mode='fan_in')
            nn.init.kaiming_normal_(self.conv_mask_t[i].weight, mode='fan_in')
            # Zero-init last layer of channel_add convs
            if isinstance(self.channel_add_conv_s[i], nn.Sequential):
                nn.init.constant_(self.channel_add_conv_s[i][-1].weight, 0)
                nn.init.constant_(self.channel_add_conv_s[i][-1].bias, 0)
            if isinstance(self.channel_add_conv_t[i], nn.Sequential):
                nn.init.constant_(self.channel_add_conv_t[i][-1].weight, 0)
                nn.init.constant_(self.channel_add_conv_t[i][-1].bias, 0)


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        
        # Channel alignment: student → teacher dims (no .to(device) — inherits from parent)
        self.align_module = nn.ModuleList()
        for s_chan, t_chan in zip(channels_s, channels_t):
            if distiller == 'fgd':
                # FGD requires raw feature scale for attention-weighted matching — no BN
                align = nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0)
            else:
                align = nn.Sequential(
                    nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(t_chan, affine=False)
                )
            self.align_module.append(align)

        self.needs_gt = (distiller == 'fgd')

        if distiller == 'mgd':
            # After alignment, student features have teacher_channels dims
            self.feature_loss = MGDLoss(channels_t, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_t, channels_t)
        elif distiller == 'fgd':
            self.feature_loss = FGDLoss(channels_t)
        else:
            raise NotImplementedError

    def forward(self, y_s, y_t, gt_bboxes=None, img_shape=None):
        if len(y_s) != len(y_t):
            y_t = y_t[len(y_t) // 2:]

        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # Match input dtype to module dtype
            s = s.type(next(self.align_module[idx].parameters()).dtype)
            t = t.type(next(self.align_module[idx].parameters()).dtype)
            
            # Always align student channels to teacher dimensions
            s = self.align_module[idx](s)
            stu_feats.append(s)
            tea_feats.append(t.detach())

        if self.needs_gt:
            loss = self.feature_loss(stu_feats, tea_feats, gt_bboxes=gt_bboxes, img_shape=img_shape)
        else:
            loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss


class DistillationLoss:
    def __init__(self, models, modelt, distiller="CWDLoss"):
        self.distiller = distiller
        self.layers = ["6", "8", "13", "16", "19", "22"]
        self.models = models 
        self.modelt = modelt

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # ini warm up
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 640, 640)
            _ = self.models(dummy_input.to(device))
            _ = self.modelt(dummy_input.to(device))
        
        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        self.remove_handle = []
        
        self._find_layers()
        
        self.distill_loss_fn = FeatureLoss(
            channels_s=self.channels_s, 
            channels_t=self.channels_t, 
            distiller=distiller[:3]
        )
        
    def _find_layers(self):

        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        
        for name, ml in self.modelt.named_modules():
            if name is not None:
                name = name.split(".")
                # print(name)
                
                if name[0] != "model":
                    continue
                if len(name) >= 3:
                    if name[1] in self.layers:
                        if "cv2" in name[2]:
                            if hasattr(ml, 'conv'):
                                self.channels_t.append(ml.conv.out_channels)
                                self.teacher_module_pairs.append(ml)
        # print()
        for name, ml in self.models.named_modules():
            if name is not None:
                name = name.split(".")
                # print(name)
                if name[0] != "model":
                    continue
                if len(name) >= 3:
                    if name[1] in self.layers:
                        if "cv2" in name[2]:
                            if hasattr(ml, 'conv'):
                                self.channels_s.append(ml.conv.out_channels)
                                self.student_module_pairs.append(ml)

        nl = min(len(self.channels_s), len(self.channels_t))
        self.channels_s = self.channels_s[-nl:]
        self.channels_t = self.channels_t[-nl:]
        self.teacher_module_pairs = self.teacher_module_pairs[-nl:]
        self.student_module_pairs = self.student_module_pairs[-nl:]

    def register_hook(self):
        # Remove the existing hook if they exist
        self.remove_handle_()
        
        self.teacher_outputs = []
        self.student_outputs = []

        def make_student_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    out = output.clone()  # Clone to ensure we don't modify the original
                    l.append(out)
                else:
                    l.append([o.clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        def make_teacher_hook(l):
            def forward_hook(m, input, output):
                if isinstance(output, torch.Tensor):
                    l.append(output.detach().clone())  # Detach and clone teacher outputs
                else:
                    l.append([o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        for ml, ori in zip(self.teacher_module_pairs, self.student_module_pairs):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_hook(self.student_outputs)))

    def get_loss(self, batch=None):
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0, requires_grad=True)
        
        if len(self.teacher_outputs) != len(self.student_outputs):
            print(f"Warning: Mismatched outputs - Teacher: {len(self.teacher_outputs)}, Student: {len(self.student_outputs)}")
            return torch.tensor(0.0, requires_grad=True)

        # Extract GT bboxes for FGD (ignored by CWD/MGD)
        gt_bboxes = None
        img_shape = None
        if batch is not None and self.distill_loss_fn.needs_gt:
            from ultralytics.utils.ops import xywh2xyxy
            img_h, img_w = batch['img'].shape[2:]
            img_shape = (img_h, img_w)
            batch_idx = batch['batch_idx']  # (M,)
            bboxes_xywh = batch['bboxes']   # (M, 4) normalized xywh
            # Convert to pixel xyxy
            bboxes_xyxy = xywh2xyxy(bboxes_xywh)
            bboxes_xyxy[:, [0, 2]] *= img_w
            bboxes_xyxy[:, [1, 3]] *= img_h
            # Split per image
            bs = batch['img'].shape[0]
            gt_bboxes = []
            for i in range(bs):
                mask = batch_idx == i
                gt_bboxes.append(bboxes_xyxy[mask])

        if gt_bboxes is not None:
            quant_loss = self.distill_loss_fn(
                y_s=self.student_outputs, y_t=self.teacher_outputs,
                gt_bboxes=gt_bboxes, img_shape=img_shape
            )
        else:
            quant_loss = self.distill_loss_fn(
                y_s=self.student_outputs, y_t=self.teacher_outputs
            )

        self.teacher_outputs.clear()
        self.student_outputs.clear()
        
        return quant_loss

    def remove_handle_(self):
        for rm in self.remove_handle:
            rm.remove()
        self.remove_handle.clear()


class LogitDistillationLoss:
    """GT-guided logit-level knowledge distillation for YOLO detection models.

    Distills knowledge from the teacher's detection head outputs (class logits and bbox
    DFL distributions) directly to the student. Uses fg_mask from TAL (Task-Aligned
    Assigner) to distill ONLY on foreground positions assigned to ground-truth objects,
    avoiding noise from background positions.

    Unlike feature-level distillation (CWD/MGD), this approach:
    - Requires no learnable alignment modules (no extra optimizer params)
    - Operates on task-aligned signals (detection outputs, not abstract features)
    - Naturally handles different backbone widths (detection head output format
      is identical across model scales: nc + reg_max*4 channels)
    """

    def __init__(self, models, modelt, tau=2.0):
        """Initialize logit-level distillation.

        Args:
            models: Student model (may be DDP-wrapped).
            modelt: Teacher model (may be DDP-wrapped).
            tau: Temperature for softening distributions. Higher = softer.
        """
        self.remove_handle = []

        # Get Detect modules (unwrap DDP if needed)
        s_model = models.module if hasattr(models, 'module') else models
        t_model = modelt.module if hasattr(modelt, 'module') else modelt
        self.s_model = s_model  # reference for accessing criterion.last_fg_mask
        self.student_detect = s_model.model[-1]
        self.teacher_detect = t_model.model[-1]

        self.nc = self.student_detect.nc
        self.reg_max = self.student_detect.reg_max
        self.no = self.nc + self.reg_max * 4
        self.tau = tau

        # Verify compatibility
        assert self.nc == self.teacher_detect.nc, \
            f"Student nc={self.nc} != Teacher nc={self.teacher_detect.nc}"
        assert self.reg_max == self.teacher_detect.reg_max, \
            f"Student reg_max={self.reg_max} != Teacher reg_max={self.teacher_detect.reg_max}"

        self.student_outputs = []
        self.teacher_outputs = []

    def register_hook(self):
        """Register forward hooks on student and teacher Detect modules."""
        self.remove_handle_()
        self.student_outputs = []
        self.teacher_outputs = []

        def make_student_hook(storage):
            def hook(m, inp, out):
                if isinstance(out, list):
                    # Training mode: Detect returns list of [B, no, H, W]
                    storage.extend(out)
                elif isinstance(out, tuple) and len(out) == 2:
                    # Eval mode: Detect returns (y, x)
                    raw = out[1]
                    if isinstance(raw, list):
                        storage.extend(raw)
            return hook

        def make_teacher_hook(storage):
            def hook(m, inp, out):
                if isinstance(out, tuple) and len(out) == 2:
                    raw = out[1]
                    if isinstance(raw, list):
                        storage.extend([o.detach() for o in raw])
                elif isinstance(out, list):
                    storage.extend([o.detach() for o in out])
            return hook

        self.remove_handle.append(
            self.student_detect.register_forward_hook(make_student_hook(self.student_outputs))
        )
        self.remove_handle.append(
            self.teacher_detect.register_forward_hook(make_teacher_hook(self.teacher_outputs))
        )

    def get_loss(self, batch=None):
        """Compute GT-guided logit distillation loss from captured head outputs.

        Uses fg_mask from TAL assignment to distill ONLY on foreground (positive)
        anchor positions. This avoids the noise from background positions where
        teacher predictions are unreliable and conflict with ground truth targets.

        Returns:
            torch.Tensor: Scalar distillation loss (with gradient for student path).
        """
        if not self.student_outputs or not self.teacher_outputs:
            self.student_outputs.clear()
            self.teacher_outputs.clear()
            device = self.student_detect.stride.device
            return torch.tensor(0.0, device=device, requires_grad=True)

        if len(self.student_outputs) != len(self.teacher_outputs):
            LOGGER.warning(
                f"Logit KD: Mismatched outputs - Student: {len(self.student_outputs)}, "
                f"Teacher: {len(self.teacher_outputs)}"
            )
            self.student_outputs.clear()
            self.teacher_outputs.clear()
            device = self.student_detect.stride.device
            return torch.tensor(0.0, device=device, requires_grad=True)

        device = self.student_detect.stride.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=False)
        n_levels = len(self.student_outputs)
        reg_max = self.reg_max
        tau = self.tau

        # --- Get fg_mask from TAL (computed during main loss) ---
        fg_masks_per_level = None
        criterion = getattr(self.s_model, 'criterion', None)
        if criterion is not None and hasattr(criterion, 'last_fg_mask'):
            fg_mask = criterion.last_fg_mask  # (B, total_anchors) bool
            level_sizes = criterion.last_anchor_splits  # [H1*W1, H2*W2, H3*W3]
            if len(level_sizes) == n_levels:
                fg_masks_per_level = fg_mask.split(level_sizes, dim=1)  # list of (B, Hi*Wi)

        for idx, (s, t) in enumerate(zip(self.student_outputs, self.teacher_outputs)):
            B, C, H, W = s.shape
            N = B * H * W  # total grid cells at this level

            # Reshape: (B, C, H, W) -> (B*H*W, C)
            s_flat = s.permute(0, 2, 3, 1).reshape(N, C).float()
            t_flat = t.permute(0, 2, 3, 1).reshape(N, C).float()

            # --- Apply GT-guided mask (only distill on foreground positions) ---
            if fg_masks_per_level is not None:
                level_mask = fg_masks_per_level[idx].reshape(N)  # (B*H*W,)
                n_pos = level_mask.sum().item()
                if n_pos == 0:
                    continue  # no positive anchors at this level
                s_flat = s_flat[level_mask]  # (n_pos, C)
                t_flat = t_flat[level_mask]  # (n_pos, C)

            # Split into bbox DFL logits and class logits
            s_box, s_cls = s_flat[:, :reg_max * 4], s_flat[:, reg_max * 4:]
            t_box, t_cls = t_flat[:, :reg_max * 4], t_flat[:, reg_max * 4:]

            # === Classification distillation ===
            # Mean over selected (foreground) positions
            t_cls_soft = torch.sigmoid(t_cls / tau).detach()
            cls_loss = F.binary_cross_entropy_with_logits(
                s_cls / tau, t_cls_soft, reduction='none'
            ).mean(dim=1)  # (n_pos,)
            cls_loss = cls_loss.mean() * (tau ** 2)

            # === Bbox DFL distribution distillation ===
            n_pos_actual = s_flat.shape[0]
            s_dfl = s_box.view(n_pos_actual, 4, reg_max)
            t_dfl = t_box.view(n_pos_actual, 4, reg_max)

            t_dfl_soft = F.softmax(t_dfl / tau, dim=2).detach()
            s_dfl_log = F.log_softmax(s_dfl / tau, dim=2)

            # KL(teacher || student) per position
            dfl_loss = F.kl_div(s_dfl_log, t_dfl_soft, reduction='none').mean(dim=2).mean(dim=1)  # (n_pos,)
            dfl_loss = dfl_loss.mean() * (tau ** 2)

            total_loss = total_loss + cls_loss + dfl_loss

        self.student_outputs.clear()
        self.teacher_outputs.clear()

        return total_loss / max(n_levels, 1)

    def remove_handle_(self):
        """Remove all registered hooks."""
        for h in self.remove_handle:
            h.remove()
        self.remove_handle.clear()


class BaseTrainer:
    """
    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        
        if overrides:
            self.teacher = overrides.get("teacher", None)
            self.loss_type = overrides.get("distillation_loss", None)
            if "teacher" in overrides:
                overrides.pop("teacher")
            if "distillation_loss" in overrides:
                overrides.pop("distillation_loss")
        else:
            self.loss_type = None
            self.teacher = None
        
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # HUB
        self.hub_session = None

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch<1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""
        
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        
        # Load teacher model to device
        if self.teacher is not None:
            for k, v in self.teacher.named_parameters():
                v.requires_grad = False  # Teacher must be frozen
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()  # Teacher always in eval mode
                
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)
            
            if self.teacher is not None:
                self.teacher = nn.parallel.DistributedDataParallel(self.teacher, device_ids=[RANK])

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(
                model=self.model,
                imgsz=self.args.imgsz,
                amp=self.amp,
                batch=self.batch_size,
            )

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            teacher=self.teacher,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
            
        # make loss
        if self.teacher is not None:
            if self.loss_type == 'logit':
                distillation_loss = LogitDistillationLoss(self.model, self.teacher)
                LOGGER.info("Using logit-level knowledge distillation (tau=2.0, objectness-weighted)")
            else:
                distillation_loss = DistillationLoss(self.model, self.teacher, distiller=self.loss_type)
                LOGGER.info(f"Using {self.loss_type}-level feature distillation")
                # Move FeatureLoss (align_module, etc.) to training device
                distillation_loss.distill_loss_fn = distillation_loss.distill_loss_fn.to(self.device)
                # Add FeatureLoss learnable params (align_module) to optimizer
                distill_params = list(distillation_loss.distill_loss_fn.parameters())
                if distill_params:
                    init_lr = self.optimizer.param_groups[0]["lr"]
                    self.optimizer.add_param_group({
                        "params": distill_params,
                        "weight_decay": 0.0,
                        "lr": init_lr,
                        "initial_lr": init_lr,
                    })
            # Now that distill param groups match, load the deferred optimizer state
            if hasattr(self, "_deferred_optimizer_state"):
                try:
                    self.optimizer.load_state_dict(self._deferred_optimizer_state)
                    LOGGER.info("Deferred optimizer state restored successfully after distillation params were added.")
                except Exception as e:
                    LOGGER.warning(f"Could not restore deferred optimizer state: {e}")
                del self._deferred_optimizer_state
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.td_loss = None  # running average of distillation loss per epoch
            self.td_ratio = None  # running average of distill/main loss ratio per epoch
            
            if self.teacher is not None:
                distillation_loss.register_hook()
            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                else:
                    # After warmup: sync distill param group LR with scheduler
                    n_pg = len(self.optimizer.param_groups)
                    if n_pg > 3:  # distill param group exists
                        self.optimizer.param_groups[-1]["lr"] = (
                            self.optimizer.param_groups[-1]["initial_lr"] * self.lf(epoch)
                        )

                # Forward
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                    
                # Distillation logic — skip during warmup
                if self.teacher is not None and ni > nw:
                    with torch.no_grad():
                        pred = self.teacher(batch['img'])
                        
                    self.d_loss = distillation_loss.get_loss(batch=batch)
                    # Gradual distill warmup over distill_warmup_iters after main warmup ends
                    distill_warmup_iters = nb * self.args.distill_warmup_epochs
                    if distill_warmup_iters > 0:
                        distill_progress = min(1.0, (ni - nw) / distill_warmup_iters)
                    else:
                        distill_progress = 1.0
                    self.d_loss *= self.args.distill_weight * distill_progress
                    # Batch-scale d_loss to match self.loss convention:
                    # v8DetectionLoss returns loss.sum() * batch_size, so d_loss must also be batch-scaled.
                    batch_size = batch['img'].shape[0]
                    self.d_loss = self.d_loss * batch_size
                    # Ratio: compare per-item scales (d_loss/batch_size vs loss_items.sum())
                    main_loss_logged_scale = self.loss_items.detach().sum().abs() + 1e-9
                    d_ratio_val = (self.d_loss.detach().abs() / batch_size) / main_loss_logged_scale
                    self.loss += self.d_loss
                    # Track running average of distillation loss (per-item scale for CSV logging)
                    d_loss_logged = self.d_loss.detach() / batch_size
                    self.td_loss = (
                        (self.td_loss * i + d_loss_logged) / (i + 1) if self.td_loss is not None else d_loss_logged
                    )
                    self.td_ratio = (
                        (self.td_ratio * i + d_ratio_val) / (i + 1) if self.td_ratio is not None else d_ratio_val
                    )
                elif self.teacher is not None and ni <= nw:
                    # During warmup: run teacher forward to fill hooks, but discard loss
                    with torch.no_grad():
                        pred = self.teacher(batch['img'])
                    distillation_loss.get_loss(batch=batch)  # clear hook buffers

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch["cls"].shape[0],  # batch size, i.e. 8
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                
            # More distillation logic
            if self.teacher is not None:
                distillation_loss.remove_handle_()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                distill_metrics = {
                    "train/distill_loss": round(float(self.td_loss), 5) if self.td_loss is not None else 0.0,
                    "train/distill_ratio": round(float(self.td_ratio), 5) if self.td_ratio is not None else 0.0,
                } if self.teacher is not None else {}
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **distill_metrics, **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            seconds = time.time() - self.train_time_start
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        
        # Distill logic
        if self.teacher is not None:
            distillation_loss.remove_handle_()
        self.run_callbacks("teardown")

    def _get_memory(self):
        """Get accelerator memory utilization in GB."""
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
        elif self.device.type == "cpu":
            memory = 0
        else:
            memory = torch.cuda.memory_reserved()
        return memory / 1e9

    def _clear_memory(self):
        """Clear accelerator memory on different platforms."""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dict using pandas."""
        import pandas as pd  # scope for faster 'import ultralytics'
        try:
            return pd.read_csv(self.csv).to_dict(orient="list")
        except pd.errors.ParserError:
            LOGGER.warning(
                f"WARNING ⚠️ Failed to parse {self.csv.name} due to malformed rows. "
                "Retrying with tolerant parser and skipping bad lines."
            )
            return pd.read_csv(self.csv, engine="python", on_bad_lines="skip").to_dict(orient="list")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        self.data = data
        return data["train"], data.get("val") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        t = time.time() - self.train_time_start
        row_values = [self.epoch + 1, t] + vals
        row_keys = ["epoch", "time", *keys]

        def _fmt(value):
            try:
                return f"{float(value):.6g}"
            except (TypeError, ValueError):
                return str(value)

        row = {k: _fmt(v) for k, v in zip(row_keys, row_values)}

        # Keep CSV schema consistent across resumed runs (e.g. distillation adds new metric columns).
        if self.csv.exists() and self.csv.stat().st_size > 0:
            with open(self.csv, newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])

            if header:
                missing = [k for k in row_keys if k not in header]
                if missing:
                    merged_header = header + missing
                    with open(self.csv, newline="") as f:
                        old_rows = list(csv.DictReader(f))
                    with open(self.csv, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=merged_header)
                        writer.writeheader()
                        writer.writerows(old_rows)
                        writer.writerow({k: row.get(k, "") for k in merged_header})
                    LOGGER.warning(
                        f"WARNING ⚠️ Expanded {self.csv.name} columns to match current metrics: "
                        f"{', '.join(missing)}"
                    )
                    return

                # Existing schema already contains all keys (order may differ), append in header order.
                with open(self.csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([row.get(k, "") for k in header])
                return

        with open(self.csv, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(row_keys)
            writer.writerow([row[k] for k in row_keys])

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        ckpt = {}
        for f in self.last, self.best:
            if f.exists():
                if f is self.last:
                    ckpt = strip_optimizer(f)
                elif f is self.best:
                    k = "train_results"  # update best.pt train_metrics from last.pt
                    strip_optimizer(f, updates={k: ckpt[k]} if k in ckpt else None)
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            except ValueError as e:
                # Param group count mismatch can happen when resuming distillation training:
                # the checkpoint optimizer has an extra group for distillation (FeatureLoss) params
                # that haven't been added yet at this point. Store the state dict and defer loading
                # until distillation params are re-added in _do_train.
                LOGGER.warning(
                    f"⚠️ Optimizer state dict mismatch ({e}), deferring optimizer state restore "
                    f"until distillation param groups are rebuilt."
                )
                self._deferred_optimizer_state = ckpt["optimizer"]
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, teacher=None, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            teacher (torch.nn.Module): the teacher model that will help the model to improve.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        # Note: teacher params are NOT added to optimizer — teacher must stay frozen

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer
