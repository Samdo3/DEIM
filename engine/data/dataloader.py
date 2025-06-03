"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

from ..core import register
torchvision.disable_beta_transforms_warning()
from copy import deepcopy
from PIL import Image, ImageDraw
import os
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat




__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('')


def generate_scales(base_size, base_size_repeat):
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


@register() 
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self, 
        stop_epoch=None, 
        ema_restart_decay=0.9999,
        base_size=640,
        base_size_repeat=None,
        mixup_prob=0.0,
        mixup_epochs=[0, 0],
        data_vis=False,
        vis_save='./vis_dataset/'
    ) -> None:
        super().__init__()
        self.base_size = base_size
        self.scales = generate_scales(base_size, base_size_repeat) if base_size_repeat is not None else None
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay
        # FIXME Mixup
        self.mixup_prob, self.mixup_epochs = mixup_prob, mixup_epochs
        if self.mixup_prob > 0:
            self.data_vis, self.vis_save = data_vis, vis_save
            os.makedirs(self.vis_save, exist_ok=True) if self.data_vis else None
            print("     ### Using MixUp with Prob@{} in {} epochs ### ".format(self.mixup_prob, self.mixup_epochs))
        if stop_epoch is not None:
            print("     ### Multi-scale Training until {} epochs ### ".format(self.stop_epoch))
            print("     ### Multi-scales@ {} ###        ".format(self.scales))
        self.print_info_flag = True
        # self.interpolation = interpolation

    def apply_mixup(self, images, targets):
        """
        가정: 
        - 현재 'boxes' 필드가 'cxcywh' + normalize=True 형태로 들어옴 (0~1)
        - Mixup 단계에서 편의상 '픽셀 XYXY'로 변환 -> roll -> concat -> filter
        - 그리고 최종에는 다시 'cxcywh'(정규화)로 돌려서 저장한다.
        - 이미지 크기는 고정 512 x 512 라고 가정.
        """
        # 1) mixup on/off
        if not (random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]):
            return images, targets

        # 2) ratio
        beta = round(random.uniform(0.45, 0.55), 6)

        # 3) mix image
        images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

        shifted_targets = targets[-1:] + targets[:-1]
        updated_targets = deepcopy(targets)

        for i in range(len(targets)):
            # 만약 boxes가 존재하지 않으면 skip
            if "boxes" not in updated_targets[i] or "boxes" not in shifted_targets[i]:
                continue

            boxesA = updated_targets[i]["boxes"]   # BoundingBoxes(cxcywh, normalize=True)
            boxesB = shifted_targets[i]["boxes"]

            # (A) cxcywh(normalized) -> 픽셀 xyxy
            # 1) as_subclass tensor
            A_cxcywh = boxesA.as_subclass(torch.Tensor)  # shape [N,4]
            B_cxcywh = boxesB.as_subclass(torch.Tensor)

            # 2) undo normalization => 픽셀 cxcywh
            #   (cx,cy,w,h) in [0,1] -> multiply by 512
            A_cxcywh_px = A_cxcywh.clone()
            B_cxcywh_px = B_cxcywh.clone()
            A_cxcywh_px[..., :4] *= 512.0   # (cx,cy,w,h) => now 0~512
            B_cxcywh_px[..., :4] *= 512.0

            # 3) cxcywh -> xyxy
            A_xyxy = box_convert(A_cxcywh_px, in_fmt="cxcywh", out_fmt="xyxy")
            B_xyxy = box_convert(B_cxcywh_px, in_fmt="cxcywh", out_fmt="xyxy")

            # concat
            merged_xyxy = torch.cat([A_xyxy, B_xyxy], dim=0)
            # print("[DEBUG] merged_xyxy(before valid):", merged_xyxy.shape, merged_xyxy[:5])

            # labels, area
            if "labels" in updated_targets[i] and "labels" in shifted_targets[i]:
                updated_targets[i]["labels"] = torch.cat([
                    updated_targets[i]["labels"],
                    shifted_targets[i]["labels"]
                ], dim=0)
            if "area" in updated_targets[i] and "area" in shifted_targets[i]:
                updated_targets[i]["area"] = torch.cat([
                    updated_targets[i]["area"],
                    shifted_targets[i]["area"]
                ], dim=0)

            # mixup ratio
            Ncur = A_xyxy.shape[0]
            Nshf = B_xyxy.shape[0]
            updated_targets[i]["mixup"] = torch.tensor([beta]*Ncur + [(1-beta)]*Nshf, dtype=torch.float32)

            # 4) sanitize in pixel coords
            x1, y1, x2, y2 = merged_xyxy.unbind(-1)
            #  (A) 만약 이미지크기가 512×512 라면 clamp로 이미지 범위 밖 제거 or 잘라내기
            x1 = x1.clamp(0, 512)
            x2 = x2.clamp(0, 512)
            y1 = y1.clamp(0, 512)
            y2 = y2.clamp(0, 512)

            # (B) reorder => x1 <= x2, y1 <= y2
            new_x1 = torch.min(x1, x2)
            new_x2 = torch.max(x1, x2)
            new_y1 = torch.min(y1, y2)
            new_y2 = torch.max(y1, y2)

            merged_xyxy = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

            # (C) 작은 박스 제거
            w = new_x2 - new_x1
            h = new_y2 - new_y1
            valid = (w>1e-3) & (h>1e-3)
            merged_xyxy = merged_xyxy[valid]

            # 최종 xyxy
            # print("[DEBUG] merged_xyxy(after valid):", merged_xyxy.shape, merged_xyxy[:5])

            # (C) labels, area 등도 valid만큼 필터
            if "labels" in updated_targets[i]:
                if updated_targets[i]["labels"].shape[0] == valid.shape[0]:
                    updated_targets[i]["labels"] = updated_targets[i]["labels"][valid]
                else:
                    updated_targets[i]["labels"] = updated_targets[i]["labels"].new_empty((0,))
            if "area" in updated_targets[i]:
                if updated_targets[i]["area"].shape[0] == valid.shape[0]:
                    updated_targets[i]["area"] = updated_targets[i]["area"][valid]
                else:
                    updated_targets[i]["area"] = updated_targets[i]["area"].new_empty((0,))
            if "mixup" in updated_targets[i]:
                if updated_targets[i]["mixup"].shape[0] == valid.shape[0]:
                    updated_targets[i]["mixup"] = updated_targets[i]["mixup"][valid]
                else:
                    updated_targets[i]["mixup"] = updated_targets[i]["mixup"].new_empty((0,))

            # 5) pixel xyxy -> normalized cxcywh
            #   (1) xyxy -> cxcywh in pixel
            merged_cxcywh_px = box_convert(merged_xyxy, "xyxy", "cxcywh")
            #   (2) pixel -> normalize
            merged_cxcywh_norm = merged_cxcywh_px.clone()
            merged_cxcywh_norm[..., :4] /= 512.0

            # 6) wrap to boundingBoxes
            updated_targets[i]["boxes"] = BoundingBoxes(
                merged_cxcywh_norm,
                format=BoundingBoxFormat.CXCYWH,
                canvas_size=(512,512) # or (H,W)
            )

        # 8) mixup ratio 저장
        #    (필요하다면, concat된 순서에 맞게 [beta]*(N_cur) + [(1-beta)]*(N_shift) )
        #    여기서는 단순히 “배치 전체가 beta”로 묶는다면 아래처럼
        for i in range(len(updated_targets)):
            mixup_tensor = torch.full((len(updated_targets[i]['boxes']),), fill_value=beta, dtype=torch.float32)
            # (원한다면 cur-part와 shifted-part를 구분해서 저장 가능)
            updated_targets[i]['mixup'] = mixup_tensor

        # 예시: MixUp 시각화 코드
        if self.data_vis:
            from PIL import Image, ImageDraw
            import os
            os.makedirs(self.vis_save, exist_ok=True)

            for i, t in enumerate(updated_targets):
                # images[i] => (C,H,W) float32 tensor in [0..1]
                image_tensor = images[i].detach().cpu()
                image_tensor_uint8 = (image_tensor * 255).clamp(0,255).byte()
                # (H,W,C)
                image_numpy = image_tensor_uint8.numpy().transpose((1, 2, 0))
                # PIL Image
                pilImage = Image.fromarray(image_numpy, mode='L')
                draw = ImageDraw.Draw(pilImage)

                # t['boxes'] => BoundingBoxes(...)  with format=XYWH (픽셀 좌표로 가정)
                boxes_xywh_draw = t['boxes'].as_subclass(torch.Tensor)  # shape=(N,4)
                if boxes_xywh_draw.numel() > 0:
                    # xywh -> (x1, y1, x2, y2)
                    x = boxes_xywh_draw[:, 0]
                    y = boxes_xywh_draw[:, 1]
                    w = boxes_xywh_draw[:, 2]
                    h = boxes_xywh_draw[:, 3]

                    x1 = x
                    y1 = y
                    x2 = x + w
                    y2 = y + h

                    # 이제 x1,y1,x2,y2는 픽셀 좌표라고 가정
                    for bx in range(len(x1)):
                        draw.rectangle(
                            [x1[bx].item(), y1[bx].item(), x2[bx].item(), y2[bx].item()],
                            outline=(255, 0, 0), width=2
                        )

                save_path = f"{self.vis_save}/mixup_epoch{self.epoch}_idx{i}_boxes{len(boxes_xywh_draw)}.jpg"
                pilImage.save(save_path)
                print(f"[MixUp DEBUG] saved: {save_path}")

        return images, updated_targets

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        # if self.epoch < 2 and random.random()<0.1:
        #     # (A) debug: batch별 박스 총합
        #     total_boxes = sum(t["boxes"].shape[0] for t in targets)
        #     print(f"[DEBUG] Collate => total GT boxes in this batch = {total_boxes}")

        # Mixup
        images, targets = self.apply_mixup(images, targets)

        # (B) debug: mixup 후 box 총합
        # total_boxes_after = sum(t["boxes"].shape[0] for t in targets)
        # if total_boxes_after>0:
        #     print(f"[DEBUG] After MixUp => total GT boxes = {total_boxes_after}")

        if self.scales is not None and self.epoch < self.stop_epoch:
            # sz = random.choice(self.scales)
            # sz = [sz] if isinstance(sz, int) else list(sz)
            # VF.resize(inpt, sz, interpolation=self.interpolation)

            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('')

        return images, targets
