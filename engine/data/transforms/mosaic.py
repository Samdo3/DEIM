"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register

from torchvision.transforms.v2.functional import to_pil_image, pil_to_tensor
from torchvision.tv_tensors import Image as TVImage
from torchvision.ops import box_convert



@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = T.RandomAffine(degrees=rotation_range,
                                               translate=translation_range,
                                               scale=scaling_range,
                                               fill=fill_value)
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop



    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    # @staticmethod
    def _clone(self, tensor_dict):
        cloned = {}
        for (key, value) in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            else:
                # e.g. str, int, float, list, etc.
                cloned[key] = value
        return cloned

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)

        sample_indices = random.choices(range(len(cache)), k=3)

        mosaic_samples = [
            dict(
                img=cache[idx]["img"].clone(),
                labels=self._clone(cache[idx]["labels"])
            )
            for idx in sample_indices
        ]
        # 현재 image도 clone
        mosaic_samples = [dict(img=image.clone(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        """
        1) merged_image: PIL Image("L")
        2) paste()로 4등분
        3) 모든 target['boxes']는 cxcywh -> xyxy 변환 + offset
        4) 한 번에 merged_target에 cat()
        5) broken box (w<=0,h<=0) 제거
        6) return (mosaic_image_tensor, merged_target)
        """
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        placement_offsets = [
            [0, 0],
            [max_width, 0],
            [0, max_height],
            [max_width, max_height]
        ]
        # 2배 사이즈 (가로 2×max_width, 세로 2×max_height)
        merged_image = Image.new(mode="L", size=(max_width*2, max_height*2), color=0)

        # 오프셋을 xyxy(4원소)로 더하기 쉽게 tensor 준비
        offsets_xyxy = torch.tensor([
            [0, 0, 0, 0],
            [max_width, 0, max_width, 0],
            [0, max_height, 0, max_height],
            [max_width, max_height, max_width, max_height]
        ], dtype=torch.float32)

        mosaic_targets = []
        # 1) 이미지 4장을 순회
        for i, sample in enumerate(mosaic_samples):
            img_t = sample["img"]  # (C,H,W) float or PIL
            tgt = sample["labels"] # dict with 'boxes'(cxcywh)...

            # (A) PIL 변환
            if not isinstance(img_t, Image.Image):
                # mode="L" assumed
                if img_t.shape[0] == 1:
                    pil_img = to_pil_image(img_t, mode="L")
                elif img_t.shape[0] == 3:
                    pil_img = to_pil_image(img_t, mode="RGB").convert("L")
                else:
                    raise ValueError(f"Unexpected channels: {img_t.shape[0]}")
            else:
                pil_img = img_t

            # paste
            merged_image.paste(pil_img, tuple(placement_offsets[i]))

            # (B) 박스 변환
            # cxcywh -> xyxy
            boxes_cxcywh = tgt["boxes"]
            cxcywh_t = boxes_cxcywh.as_subclass(torch.Tensor)  # shape(N,4)
            xyxy_t = box_convert(cxcywh_t, in_fmt="cxcywh", out_fmt="xyxy")

            # (C) offset
            xyxy_t += offsets_xyxy[i]

            # 디버그: 오프셋 더한 뒤 값
            # print(f"[Mosaic] sample={i}, AFTER offset => boxes_xyxy[:5]:", boxes_xyxy_t[:5])


            # (D) clamp (2배 canvas 넘어가면 잘라주기) - 선택사항
            #     예: x1,x2 in [0,2×max_width], y1,y2 in [0,2×max_height]
            x1, y1, x2, y2 = xyxy_t.unbind(-1)
            x1 = x1.clamp(0, max_width*2)
            x2 = x2.clamp(0, max_width*2)
            y1 = y1.clamp(0, max_height*2)
            y2 = y2.clamp(0, max_height*2)

            # reorder => x1 <= x2, y1 <= y2
            new_x1 = torch.min(x1, x2)
            new_x2 = torch.max(x1, x2)
            new_y1 = torch.min(y1, y2)
            new_y2 = torch.max(y1, y2)
            xyxy_t = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

            # (E) min_size 필터 or w>1e-6
            w = new_x2 - new_x1
            h = new_y2 - new_y1
            valid = (w>1e-4) & (h>1e-4)
            xyxy_t = xyxy_t[valid]

            # labels, area 등도 있으면 동등하게 valid로 필터
            if "labels" in tgt:
                tgt["labels"] = tgt["labels"][valid] if tgt["labels"].shape[0] == valid.shape[0] else tgt["labels"].new_empty((0,))
            if "area" in tgt:
                tgt["area"] = tgt["area"][valid] if tgt["area"].shape[0] == valid.shape[0] else tgt["area"].new_empty((0,))
            
            # mosaic_targets에 담기
            tgt["boxes"] = xyxy_t
            mosaic_targets.append(tgt)

        # (F) 최종 merged_target => cat
        merged_target = {}
        for key in mosaic_targets[0].keys():
            vals = [tt[key] for tt in mosaic_targets]
            if isinstance(vals[0], torch.Tensor):
                merged_target[key] = torch.cat(vals, dim=0)
            else:
                merged_target[key] = vals

        # (G) 최종 merged_image -> tensor(float)
        mosaic_tensor = pil_to_tensor(merged_image).float().div(255.0)

        return mosaic_tensor, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """
        유사하게 “dataset 버전 mosaic”도 clamp+reorder+filter
        """
        placement_offsets = [
            [0, 0],
            [max_width, 0],
            [0, max_height],
            [max_width, max_height],
        ]
        merged_image = Image.new(mode="L", size=(max_width * 2, max_height * 2), color=0)

        # paste
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                if img.shape[0] == 1:
                    pil_img = to_pil_image(img, mode="L")
                elif img.shape[0] == 3:
                    pil_img = to_pil_image(img, mode="RGB").convert("L")
                else:
                    raise ValueError(f"Unexpected shape: {img.shape}")
            else:
                pil_img = img
            merged_image.paste(pil_img, tuple(placement_offsets[i]))

        # offsets
        offsets_xyxy = torch.tensor([
            [0, 0, 0, 0],
            [max_width, 0, max_width, 0],
            [0, max_height, 0, max_height],
            [max_width, max_height, max_width, max_height],
        ], dtype=torch.float32)

        # cat
        mosaic_tgt = {}
        for key in targets[0].keys():
            vals = []
            if key == "boxes":
                for i, t in enumerate(targets):
                    # cxcywh->xyxy
                    cxcys = t["boxes"].as_subclass(torch.Tensor)
                    xyxy = box_convert(cxcys, in_fmt="cxcywh", out_fmt="xyxy")
                    # offset
                    xyxy += offsets_xyxy[i]
                    # clamp & reorder
                    x1, y1, x2, y2 = xyxy.unbind(-1)
                    x1 = x1.clamp(0, max_width*2)
                    x2 = x2.clamp(0, max_width*2)
                    y1 = y1.clamp(0, max_height*2)
                    y2 = y2.clamp(0, max_height*2)

                    new_x1 = torch.min(x1, x2)
                    new_x2 = torch.max(x1, x2)
                    new_y1 = torch.min(y1, y2)
                    new_y2 = torch.max(y1, y2)
                    xyxy = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)

                    w = new_x2 - new_x1
                    h = new_y2 - new_y1
                    valid = (w>1e-4) & (h>1e-4)
                    xyxy = xyxy[valid]

                    vals.append(xyxy)
            else:
                for i, t in enumerate(targets):
                    vals.append(t[key])
            # cat
            if isinstance(vals[0], torch.Tensor):
                mosaic_tgt[key] = torch.cat(vals, dim=0)
            else:
                mosaic_tgt[key] = vals

        merged_tensor = pil_to_tensor(merged_image).float().div(255.0)
        return merged_tensor, mosaic_tgt

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_imgs, resized_tgts, mh, mw = self.load_samples_from_dataset(image, target, dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_imgs, resized_tgts, mh, mw)
        

        # Clamp boxes and convert
        if 'boxes' in mosaic_target:
            # mosaic_image.shape => (C, H, W)
            h, w = mosaic_image.shape[-2], mosaic_image.shape[-1]
            mosaic_target['boxes'] = convert_to_tv_tensor(
                mosaic_target['boxes'],
                'boxes',
                box_format='xyxy',
                spatial_size=(h, w)
            )
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')

        # Apply affine
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        return mosaic_image, mosaic_target, dataset
