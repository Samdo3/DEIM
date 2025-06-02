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
        B안: 평소 boxes = BoundingBoxes(CXCYWH, 정규화)
            MixUp 직전에만 XYXY 변환 → concat + sanitize → 다시 CXCYWH 복귀.

        Args:
            images (torch.Tensor): Batch of images of shape (B, C, H, W), in 0~1 float range.
            targets (list[dict]): List of target dictionaries. 각 target에는
                {
                'boxes': BoundingBoxes(CXCYWH, normalize=True),
                'labels': Tensor(...),
                'area': Tensor(...),
                (optional) 'mixup': Tensor(...),
                ...
                }
            가 포함되어 있다고 가정.

        Returns:
            (images, updated_targets):
                - images: mixup된 이미지 텐서 (shape 동일)
                - updated_targets: mixup된 타겟 리스트. 'boxes'는 다시 CXCYWH로 복귀.
        """

        # 1) MixUp 적용 여부 결정
        #    - epoch이 mixup_epochs 범위에 들어가는지
        #    - random() < self.mixup_prob 인지
        if not (random.random() < self.mixup_prob and self.mixup_epochs[0] <= self.epoch < self.mixup_epochs[-1]):
            return images, targets

        # 2) mixup ratio
        beta = round(random.uniform(0.45, 0.55), 6)  # 예) 0.45~0.55 범위 float

        # 3) 이미지 mixup (batch 내에서 roll)
        #    images.shape = (B, C, H, W)
        images = images.roll(shifts=1, dims=0).mul_(1.0 - beta).add_(images.mul(beta))

        # 4) 타겟 mixup
        shifted_targets = targets[-1:] + targets[:-1]  # roll
        updated_targets = deepcopy(targets)

        for i in range(len(targets)):
            # 원본 boxes (CXCYWH)
            if 'boxes' not in updated_targets[i]:
                continue  # 박스가 없는 경우 건너뛰기 (필요시)
            boxes_cur_cxcywh = updated_targets[i]['boxes']

            # 상대편 boxes
            if 'boxes' not in shifted_targets[i]:
                # 상대편에 아예 boxes가 없는 경우 => 그냥 skip
                continue
            boxes_shifted_cxcywh = shifted_targets[i]['boxes']

            # BoundingBoxes -> torch.Tensor로 접근
            #  (BoundingBoxes.as_subclass(torch.Tensor) or .to_tensor() 등 사용)
            boxes_cur_tensor = boxes_cur_cxcywh.as_subclass(torch.Tensor)
            boxes_shifted_tensor = boxes_shifted_cxcywh.as_subclass(torch.Tensor)

            # CXCYWH(정규화) -> XYXY(정규화)
            # box_convert()는 box_ops or torchvision.ops.box_convert를 사용
            # in_fmt="cxcywh" / out_fmt="xyxy"
            boxes_cur_xyxy = box_convert(boxes_cur_tensor, in_fmt="cxcywh", out_fmt="xyxy")
            boxes_shifted_xyxy = box_convert(boxes_shifted_tensor, in_fmt="cxcywh", out_fmt="xyxy")

            # concat
            merged_xyxy = torch.cat([boxes_cur_xyxy, boxes_shifted_xyxy], dim=0)

            # labels
            if 'labels' in updated_targets[i] and 'labels' in shifted_targets[i]:
                updated_targets[i]['labels'] = torch.cat([
                    updated_targets[i]['labels'],
                    shifted_targets[i]['labels']
                ], dim=0)
            # area
            if 'area' in updated_targets[i] and 'area' in shifted_targets[i]:
                updated_targets[i]['area'] = torch.cat([
                    updated_targets[i]['area'],
                    shifted_targets[i]['area']
                ], dim=0)

            # mixup ratio
            updated_targets[i]['mixup'] = torch.tensor(
                [beta]*boxes_cur_xyxy.shape[0] + [(1.0 - beta)]*boxes_shifted_xyxy.shape[0],
                dtype=torch.float32
            )

            # 5) sanitize
            #    x1, y1, x2, y2 = merged_xyxy.unbind(-1)
            #    w, h 계산
            #    valid 마스크로 filtering
            x1, y1, x2, y2 = merged_xyxy.unbind(-1)
            w = x2 - x1
            h = y2 - y1

            # 여기서 정규화된 좌표라고 가정. 
            # 0 ~ 1 범위 벗어나는지 check
            valid = (w > 1e-6) & (h > 1e-6) \
                    & (x1 < 1.) & (y1 < 1.) \
                    & (x2 > 0.) & (y2 > 0.)

            # valid 길이가 merged_xyxy와 같아야 함
            if valid.shape[0] != merged_xyxy.shape[0]:
                # 매우 이례적인 경우. 그냥 스킵하거나 empty로 처리
                updated_targets[i]['boxes'] = boxes_cur_cxcywh.new_empty((0,4)) # etc
                continue

            merged_xyxy = merged_xyxy[valid]

            # (C) labels, area 등도 valid만큼 필터
            if 'labels' in updated_targets[i]:
                if valid.shape[0] == updated_targets[i]['labels'].shape[0]:
                    updated_targets[i]['labels'] = updated_targets[i]['labels'][valid]
                else:
                    # 길이 mismatch => 강제로 empty
                    updated_targets[i]['labels'] = updated_targets[i]['labels'].new_empty((0,))
            if 'area' in updated_targets[i]:
                if valid.shape[0] == updated_targets[i]['area'].shape[0]:
                    updated_targets[i]['area'] = updated_targets[i]['area'][valid]
                else:
                    updated_targets[i]['area'] = updated_targets[i]['area'].new_empty((0,))
            if 'mixup' in updated_targets[i]:
                if valid.shape[0] == updated_targets[i]['mixup'].shape[0]:
                    updated_targets[i]['mixup'] = updated_targets[i]['mixup'][valid]
                else:
                    updated_targets[i]['mixup'] = updated_targets[i]['mixup'].new_empty((0,))

            # 6) XYXY -> CXCYWH 재변환
            merged_cxcywh = box_convert(merged_xyxy, in_fmt="xyxy", out_fmt="cxcywh")

            # 7) 다시 BoundingBoxes로
            #    (format=BoundingBoxFormat.CXCYWH, canvas_size=(H,W) 등)
            updated_targets[i]['boxes'] = BoundingBoxes(
                merged_cxcywh,
                format=BoundingBoxFormat.CXCYWH,
                canvas_size=boxes_cur_cxcywh.canvas_size  # 기존과 동일
            )

        # 8) mixup ratio 저장
        #    (필요하다면, concat된 순서에 맞게 [beta]*(N_cur) + [(1-beta)]*(N_shift) )
        #    여기서는 단순히 “배치 전체가 beta”로 묶는다면 아래처럼
        for i in range(len(updated_targets)):
            mixup_tensor = torch.full((len(updated_targets[i]['boxes']),), fill_value=beta, dtype=torch.float32)
            # (원한다면 cur-part와 shifted-part를 구분해서 저장 가능)
            updated_targets[i]['mixup'] = mixup_tensor

        # (Optional) 시각화
        if self.data_vis:
            from PIL import Image, ImageDraw
            import os
            os.makedirs(self.vis_save, exist_ok=True)

            for i, t in enumerate(updated_targets):
                # images[i] => (C,H,W)
                image_tensor = images[i].detach().cpu()
                image_tensor_uint8 = (image_tensor * 255).clamp(0,255).byte()
                # (H,W,C)
                image_numpy = image_tensor_uint8.numpy().transpose((1, 2, 0))
                # PIL Image
                pilImage = Image.fromarray(image_numpy, mode='RGB')
                draw = ImageDraw.Draw(pilImage)

                # t['boxes'] => BoundingBoxes(CXCYWH)
                # scale to image size and draw
                boxes_draw = t['boxes']
                if boxes_draw.numel() > 0:
                    boxes_xyxy_draw = box_convert(
                        boxes_draw.as_subclass(torch.Tensor), "cxcywh", "xyxy"
                    )
                    # scale to (H,W)
                    # x1, y1, x2, y2 = ...
                    x1, y1, x2, y2 = boxes_xyxy_draw.unbind(-1)
                    # multiply by image shape
                    x1 *= pilImage.width
                    x2 *= pilImage.width
                    y1 *= pilImage.height
                    y2 *= pilImage.height

                    for bx in range(len(x1)):
                        draw.rectangle(
                            [x1[bx].item(), y1[bx].item(), x2[bx].item(), y2[bx].item()],
                            outline=(255, 0, 0), width=2
                        )

                save_path = f"{self.vis_save}/mixup_epoch{self.epoch}_idx{i}_boxes{len(boxes_draw)}.jpg"
                pilImage.save(save_path)
                print(f"[MixUp DEBUG] saved: {save_path}")

        return images, updated_targets

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        # # --- 여기서 print ---
        # for i, t in enumerate(targets):
        #     print(f"[DEBUG] Before Mixup: target[{i}]['boxes'] = {t['boxes']}")

        # Mixup
        images, targets = self.apply_mixup(images, targets)

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
