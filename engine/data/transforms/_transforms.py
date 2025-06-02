"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask
from .._misc import BoundingBoxes as TVBoundingBoxes
from .._misc import SanitizeBoundingBoxes


from ...core import register
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        TVBoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


# @register()
# class ConvertBoxes(T.Transform):
#     _transformed_types = (
#         BoundingBoxes,
#     )
#     def __init__(self, fmt='', normalize=False) -> None:
#         super().__init__()
#         self.fmt = fmt
#         self.normalize = normalize

#     def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
#         spatial_size = getattr(inpt, _boxes_keys[1])
#         if self.fmt:
#             in_fmt = inpt.format.value.lower()
#             inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
#             inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

#         if self.normalize:
#             inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

#         return inpt

@register()
class ConvertBoxes(T.Transform):
    """
    예: 기존에 (cxcywh) → (xyxy)로 변환 후 plain Tensor로 바뀌어
    'format' 속성이 사라지는 문제가 있었음.
    
    수정안:
     1) box_convert()로 텐서 변환
     2) 다시 BoundingBoxes._wrap() 호출해 sub-class 유지
    """
    # (BoundingBoxes,) 로 제한
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt: str = "", normalize: bool = False) -> None:
        """
        Args:
            fmt: 바꿀 포맷 (예: 'xyxy' or 'cxcywh' 등)
            normalize: 추가로 0~1 normalize 할지 여부
        """
        super().__init__()
        self.fmt = fmt  # 바꿀 타겟 포맷(e.g. 'cxcywh', 'xyxy', 등)
        self.normalize = normalize

    def transform(self, inpt: BoundingBoxes, params: dict) -> BoundingBoxes:
        # 만약 inpt가 BoundingBoxes가 아닐 수도 있으니 방어
        if not isinstance(inpt, BoundingBoxes):
            return inpt
        
        # raw tensor
        coords = inpt.as_subclass(torch.Tensor)  # shape [N,4]
        in_fmt_enum = inpt.format  # e.g. BoundingBoxFormat.CXCYWH
        in_fmt_str = in_fmt_enum.value.lower()  # 'cxcywh', 'xyxy', ...
        
        # 바꿀 포맷이 지정되지 않았다면, 그냥 그대로 둠
        out_fmt_str = self.fmt.lower() if self.fmt else in_fmt_str
        
        # 1) box_convert
        converted = box_convert(coords, in_fmt=in_fmt_str, out_fmt=out_fmt_str)
        # => shape [N,4] plain Tensor

        # 2) 다시 BoundingBoxes로 wrap
        # format=BoundingBoxFormat[out_fmt_str.upper()] => 'xyxy'→BoundingBoxFormat.XYXY
        out_enum = BoundingBoxFormat[out_fmt_str.upper()]
        new_bboxes = BoundingBoxes._wrap(
            converted,
            format=out_enum,
            canvas_size=inpt.canvas_size,  # 기존 canvas_size 유지
            check_dims=False
        )

         # (추가 구현) normalize=True일 때 정규화 처리
        # → 실제로는 아래처럼 별도 로직을 넣어줘야 한다.
        if self.normalize:
            # 예시 구현
            new_bboxes = self._apply_normalize(new_bboxes)

        return new_bboxes

    def _apply_normalize(self, bboxes: BoundingBoxes) -> BoundingBoxes:
        """ bounding boxes를 [0..1] 범위로 정규화 """

        # bboxes는 이미 self.fmt 대로 변환된 상태.
        # bboxes.format 은 BoundingBoxFormat.XYXY / .XYWH / .CXCYWH 중 하나
        # bboxes.canvas_size = (H, W)  튜플
        H, W = bboxes.canvas_size
        data_tensor = bboxes.as_subclass(torch.Tensor)  # shape [N,4]

        if bboxes.format == BoundingBoxFormat.XYXY:
            # x1,y1,x2,y2
            x1, y1, x2, y2 = data_tensor.unbind(-1)
            # 폭,높이가 0이 되지 않도록 clamp
            # x1 = x1.clamp_(0, W)  # 필요하다면 clamp
            # ...
            # 정규화
            x1 = x1 / W
            x2 = x2 / W
            y1 = y1 / H
            y2 = y2 / H
            normalized_data = torch.stack([x1,y1,x2,y2], dim=-1)

            # 다시 BoundingBoxes로
            out_bboxes = BoundingBoxes._wrap(
                normalized_data,
                format=BoundingBoxFormat.XYXY,
                canvas_size=(1,1),  # 0~1 범위이므로 (1,1)로 설정 (혹은 (H,W) 그대로 두기도 함)
                check_dims=False
            )
            return out_bboxes

        elif bboxes.format == BoundingBoxFormat.XYWH:
            # x,y,w,h
            x, y, w, h = data_tensor.unbind(-1)
            x = x / W
            w = w / W
            y = y / H
            h = h / H
            normalized_data = torch.stack([x,y,w,h], dim=-1)
            out_bboxes = BoundingBoxes._wrap(
                normalized_data,
                format=BoundingBoxFormat.XYWH,
                canvas_size=(1,1),
                check_dims=False
            )
            return out_bboxes

        elif bboxes.format == BoundingBoxFormat.CXCYWH:
            # cx,cy,w,h
            cx, cy, bw, bh = data_tensor.unbind(-1)
            cx /= W
            bw /= W
            cy /= H
            bh /= H
            normalized_data = torch.stack([cx,cy,bw,bh], dim=-1)
            out_bboxes = BoundingBoxes._wrap(
                normalized_data,
                format=BoundingBoxFormat.CXCYWH,
                canvas_size=(1,1),
                check_dims=False
            )
            return out_bboxes
        else:
            # 혹시 모르는 포맷
            return bboxes

@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
