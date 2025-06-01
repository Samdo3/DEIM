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
        기존:
          merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, ...)
          merged_image.paste(img, placement_offsets[i])

        수정:
          1) merged_image = Image.new(mode="L", ...)
          2) img가 PIL이 아니면 to_pil_image(img, mode="L")
        """
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        # 흑백 => mode="L"
        merged_image = Image.new(mode="L", size=(max_width * 2, max_height * 2), color=0)

        offsets_tensor = torch.tensor([
            [0, 0, 0, 0],
            [max_width, 0, max_width, 0],
            [0, max_height, 0, max_height],
            [max_width, max_height, max_width, max_height],
        ], dtype=torch.float32)

        mosaic_target = []
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]  # tv_tensor?
            target = sample["labels"]

            # 1) 만약 CXCYWH -> XYXY 변환
            boxes_cxcywh = target['boxes']  # BoundingBoxes(CXCYWH)

            # (1) cxcywh -> xyxy 변환
            boxes_xyxy_tensor = box_convert(
                boxes_cxcywh.as_subclass(torch.Tensor),
                in_fmt="cxcywh",
                out_fmt="xyxy"
            )

            # 2) offset 더하기 (x1, y1, x2, y2 각각에)
            # offsets_tensor[i] = [ox1, oy1, ox2, oy2] 식
            boxes_xyxy_tensor += offsets_tensor[i]

            # 다시 boxes_xyxy에 덮어쓰기
            boxes_xyxy = boxes_xyxy_tensor

            # 3) target['boxes']를 XYXY로 교체
            #   -> 굳이 BoundingBoxes 객체로 다시 만들 수도 있고, 
            #      나중 convert_to_tv_tensor에서 'xyxy'로 처리하므로 일단 tensor만 둬도 됨.
            target['boxes'] = boxes_xyxy  # 이 상태가 xyxy tensor

            # 만약 img가 PIL Image가 아니라면, to_pil_image로 변환
            if not isinstance(img, Image.Image):
                # 흑백 => mode="L"
                if img.shape[0] == 1:
                    # OK: "L"
                    img = to_pil_image(img, mode="L")
                elif img.shape[0] == 3:
                    # color => "RGB"
                    img = to_pil_image(img, mode="RGB").convert("L")
                    # convert("L")로 흑백으로 바꿀 수도
                else:
                    raise ValueError(f"Unexpected channels: {img.shape[0]}")

            merged_image.paste(img, tuple(placement_offsets[i]))
            # target['boxes'] = target['boxes'] + offsets[i]
            mosaic_target.append(target)

        merged_target = {}

        # 예: 4개 타겟을 합친다고 가정
        for key in mosaic_target[0].keys():
            values = [t[key] for t in mosaic_target]

            if isinstance(values[0], torch.Tensor):
                # 텐서인 경우만 cat
                merged_target[key] = torch.cat(values, dim=0)
            else:
                # 문자열, int 등 텐서가 아닌 경우 => 단순 리스트 묶기 or 첫 번째 값만 유지
                # 예) merged_target[key] = [v for v in values]
                #    혹은 merged_target[key] = values[0]
                merged_target[key] = values
        
        mosaic_image_tensor = pil_to_tensor(merged_image)  # shape = (C,H,W)
        # === 수정: uint8 -> float ===
        mosaic_image_tensor = mosaic_image_tensor.to(torch.float32).div(255.0)

        return mosaic_image_tensor, merged_target

    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        """
        Similar fix for dataset-based mosaic
        """
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        # 흑백 => mode="L"
        merged_image = Image.new(mode="L", size=(max_width * 2, max_height * 2), color=0)

        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                # 흑백 => mode="L"
                if img.shape[0] == 1:
                    # OK: "L"
                    img = to_pil_image(img, mode="L")
                elif img.shape[0] == 3:
                    # color => "RGB"
                    img = to_pil_image(img, mode="RGB").convert("L")
                    # convert("L")로 흑백으로 바꿀 수도
                else:
                    raise ValueError(f"Unexpected channels: {img.shape[0]}")
                    
            merged_image.paste(img, tuple(placement_offsets[i]))

        # Merge targets
        offsets = torch.tensor([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]).repeat(1, 2)
        merged_target = {}
        for key in targets[0].keys():
            # 각 타겟에서 key 항목들을 모음
            values = []
            if key == "boxes":
                # boxes는 offsets를 더해야 하므로
                for i, t in enumerate(targets):
                    boxes_cxcywh = t["boxes"]  # 예: cxcywh로 저장된 BoundingBoxes 혹은 Tensor
                    boxes_xyxy_tensor = box_convert(
                        boxes_cxcywh.as_subclass(torch.Tensor),
                        in_fmt='cxcywh',
                        out_fmt='xyxy'
                    )    
                    # 2) offset
                    boxes_xyxy_tensor += offsets[i]
                    # 3) 최종 보관
                    values.append(boxes_xyxy_tensor)
            else:
                # 그외 필드는 offset이 필요 없을 수도 있음
                for i, t in enumerate(targets):
                    values.append(t[key])

            # 텐서이면 cat
            if isinstance(values[0], torch.Tensor):
                merged_target[key] = torch.cat(values, dim=0)
            else:
                # 예: tomo_id, slice_idx 같은 문자열이나 스칼라면
                # 전부 리스트로 묶거나, 하나만 쓰거나 원하는대로 처리
                merged_target[key] = values  # 혹은 merged_target[key] = values[0]

        mosaic_image_tensor = pil_to_tensor(merged_image)  # shape = (C,H,W)
        # === 수정: uint8 -> float ===
        mosaic_image_tensor = mosaic_image_tensor.to(torch.float32).div(255.0)

        return mosaic_image_tensor, merged_target

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
