"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

# DEIM/engine/data/transforms/container.py

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T # T로 alias
from typing import Any, Dict, List, Optional, Callable, Tuple as PyTuple # Tuple -> PyTuple (Python 기본 타입)
import importlib
import random
import copy # 추가

# GLOBAL_CONFIG와 register는 engine.core.workspace에서 가져옴
from engine.core.workspace import GLOBAL_CONFIG, register
# EmptyTransform 등은 현재 디렉토리의 _transforms.py에서 가져옴
# _transforms.py에 정의된 클래스들을 명시적으로 import
from ._transforms import (
    EmptyTransform,
    # RandomPhotometricDistort, # T.RandomPhotometricDistort를 직접 사용하므로 주석 처리
    # RandomZoomOut,            # T.RandomZoomOut을 직접 사용하므로 주석 처리
    # RandomHorizontalFlip,     # T.RandomHorizontalFlip을 직접 사용하므로 주석 처리
    # Resize,                   # T.Resize를 직접 사용하므로 주석 처리
    PadToSize,
    SanitizeBoundingBoxes,      # engine.data._misc.SanitizeBoundingBoxes 임포트 (from .._misc import SanitizeBoundingBoxes)
                                # _transforms.py 에서는 torchvision.transforms.v2.SanitizeBoundingBoxes를 register(name='SanitizeBoundingBoxes')로 등록.
                                # 여기서는 _transforms.py에 정의된 SanitizeBoundingBoxes를 사용한다고 가정.
    # RandomCrop,               # T.RandomCrop을 직접 사용하므로 주석 처리
    Normalize,                  # _transforms.py의 Normalize
    ConvertBoxes,               # _transforms.py의 ConvertBoxes
    ConvertPILImage,            # _transforms.py의 ConvertPILImage (오류 해결 위해 제거 예정)
    RandomIoUCrop               # _transforms.py의 RandomIoUCrop
)
from .mosaic import Mosaic

# torchvision.tv_tensors에서 필요한 타입들을 가져옵니다.
from torchvision.tv_tensors import Image as TVImage, BoundingBoxes

torchvision.disable_beta_transforms_warning()


@register()
class Compose(T.Compose): # torchvision.transforms.v2.Compose 상속
    def __init__(self, ops: List[Dict], policy: Optional[Dict] = None, mosaic_prob: float = -0.1) -> None:
        transforms = []
        if ops is not None:
            for op_cfg_orig in ops:
                op_cfg = copy.deepcopy(op_cfg_orig)
                if isinstance(op_cfg, dict):
                    name = op_cfg.pop('type')
                    transform_instance = None

                    # 1. 알려진 로컬/커스텀 Transform 먼저 확인 (GLOBAL_CONFIG보다 우선)
                    # _transforms.py 와 mosaic.py 에서 직접 import 한 클래스들을 사용
                    known_local_transforms = {
                        'Mosaic': Mosaic,
                        'EmptyTransform': EmptyTransform,
                        'PadToSize': PadToSize,
                        'SanitizeBoundingBoxes': SanitizeBoundingBoxes, # _transforms.py 또는 _misc.py 에서 온 것
                        'ConvertBoxes': ConvertBoxes,
                        'Normalize': Normalize, # _transforms.py 의 Normalize
                        'RandomIoUCrop': RandomIoUCrop,
                        # 'ConvertPILImage': ConvertPILImage, # 오류 유발 Transform은 일단 제외
                        # torchvision.transforms.v2의 클래스들은 아래 hasattr(T, name)에서 처리
                    }
                    if name in known_local_transforms:
                        transform_class = known_local_transforms[name]
                        transform_instance = transform_class(**op_cfg)
                        print(f"     ### Transform (Local/Custom) @{type(transform_instance).__name__} ###")

                    # 2. GLOBAL_CONFIG에서 찾아보기
                    elif transform_instance is None and name in GLOBAL_CONFIG and \
                       '_pymodule' in GLOBAL_CONFIG[name] and '_name' in GLOBAL_CONFIG[name]:
                        try:
                            transform_module_obj = GLOBAL_CONFIG[name]['_pymodule']
                            transform_class_name = GLOBAL_CONFIG[name]['_name']
                            if hasattr(transform_module_obj, transform_class_name):
                                transform_class = getattr(transform_module_obj, transform_class_name)
                                transform_instance = transform_class(**op_cfg)
                                print(f"     ### Transform (from GLOBAL_CONFIG) @{type(transform_instance).__name__} ###")
                        except Exception as e:
                            print(f"Warning: Failed to load {name} from GLOBAL_CONFIG: {e}.")
                            pass # 실패 시 다음 단계로

                    # 3. torchvision.transforms.v2 (T)에서 찾아보기
                    elif transform_instance is None and hasattr(T, name):
                        transform_class = getattr(T, name)
                        transform_instance = transform_class(**op_cfg)
                        print(f"     ### Transform (from torchvision.transforms.v2) @{type(transform_instance).__name__} ###")
                    
                    if transform_instance is None:
                        raise ValueError(f"Transform type '{name}' not found or not correctly registered/imported.")

                    transforms.append(transform_instance)
                    if isinstance(op_cfg_orig, dict) and 'type' not in op_cfg_orig:
                         op_cfg_orig['type'] = name 
                    # print("     ### Transform @{} ###    ".format(type(transform_instance).__name__)) # 최종 cfg 로그와 중복
                elif isinstance(op_cfg, nn.Module):
                    transforms.append(op_cfg)
                else:
                    raise ValueError(f"Transform config must be a dict or nn.Module, got {type(op_cfg)}")
        else:
            transforms =[EmptyTransform(), ]
        
        super().__init__(transforms)
        
        self.policy = policy
        # YAML의 mosaic_prob 값 사용, 없으면 __init__ 기본값 사용
        if hasattr(self, 'mosaic_prob') and self.mosaic_prob == -0.1 and mosaic_prob != -0.1:
            self.mosaic_prob = mosaic_prob
        elif not hasattr(self, 'mosaic_prob'):
            self.mosaic_prob = mosaic_prob
        print(f"Compose initialized with mosaic_prob: {getattr(self, 'mosaic_prob', 'Not Set')}")


        self._forward_methods = {
            'default': self.default_forward,
            'stop_epoch': self.stop_epoch_forward,
            'stop_sample': self.stop_sample_forward,
        }

    def get_forward(self, name: Optional[str] = None) -> Callable[..., Any]: # Callable 타입 힌트 명확화
        name = name if name else 'default'
        if name not in self._forward_methods:
            raise ValueError(f"Unknown forward method name: {name}. Available: {list(self._forward_methods.keys())}")
        return self._forward_methods[name]

    def forward(self, *inputs: Any) -> Any:
        # inputs는 (img, target, dataset_instance) 튜플
        if self.policy and 'name' in self.policy:
            return self.get_forward(self.policy['name'])(*inputs)
        else:
            return self.get_forward('default')(*inputs)

    def _apply_single_transform(self, transform_obj: nn.Module, 
                                image: Any, 
                                target: Optional[Dict], 
                                dataset_instance: Optional[Any] # Mosaic에서만 사용
                               ) -> PyTuple[Any, Optional[Dict]]:
        """
        Helper to apply a single transform.
        It dispatches the call to the transform based on its type (Mosaic vs torchvision vs other custom).
        Returns (new_image, new_target). 
        """
        if isinstance(transform_obj, Mosaic):
            # Mosaic.forward는 (self, *inputs)를 받고, 내부에서 image, target, dataset = inputs[0] 로 처리.
            # 즉, (image, target, dataset_instance) 튜플을 '하나의 인자'로 전달.
            # print(f"    Applying Mosaic: type(img)={type(image)}, type(tgt)={type(target)}")
            output_tuple = transform_obj((image, target, dataset_instance)) 
            new_image, new_target, _ = output_tuple # dataset은 반환되지만 여기서 사용 안 함
            return new_image, new_target
        
        # torchvision.transforms.v2.Transform 또는 _transforms.py에 정의된 그것의 하위 클래스들
        # (예: ConvertBoxes, SanitizeBoundingBoxes, Normalize, Resize, RandomHorizontalFlip 등)
        # 이들은 (image, target)을 여러 인자로 받거나, image만 받음.
        # target이 None일 수도 있음.
        # transform_obj.forward(*inputs)가 (image, target)을 인자로 받아 tree_flatten 처리.
        # print(f"    Applying {type(transform_obj).__name__}: type(img)={type(image)}, type(tgt)={type(target)}")
        if target is not None:
            # 대부분의 torchvision.transforms.v2 변환은 (Any) 또는 (TVTensor, TVTensor, ...)를 *inputs로 받음
            # Image와 target 딕셔너리를 함께 전달하면, Transform.forward 내부의 tree_flatten이
            # Image는 변환 대상으로, target 딕셔너리 내의 BoundingBoxes 등도 변환 대상으로 인식.
            # target 딕셔너리 자체가 transform 메서드에 전달되지 않도록 함.
            new_image, new_target = transform_obj(image, target)
        else:
            new_image = transform_obj(image) # target이 없는 경우 (예: 일부 이미지 전용 변환)
            new_target = None 
        return new_image, new_target
        

    def default_forward(self, *inputs: Any) -> PyTuple[Any, Dict, Any]:
        current_img, current_target, current_dataset_instance = inputs
        
        for transform_obj in self.transforms:
            before_boxes = current_target['boxes'].clone() if 'boxes' in current_target else None
            current_img, current_target = self._apply_single_transform(
                transform_obj, current_img, current_target, current_dataset_instance
            )
            after_boxes = current_target['boxes'].clone() if 'boxes' in current_target else None
            # 디버그 출력
            print(f"{type(transform_obj).__name__} -> #boxes={len(after_boxes)}, any neg? {(after_boxes[...,2:] < after_boxes[...,:2]).any()}")
        return current_img, current_target, current_dataset_instance

    def stop_epoch_forward(self, *inputs: Any) -> PyTuple[Any, Dict, Any]:
        img_orig, target_orig, dataset_instance = inputs
        
        current_img = img_orig
        current_target = target_orig

        # dataset_instance가 epoch 속성을 가지고 있는지 확인
        if not hasattr(dataset_instance, 'epoch'):
            print(
                "Warning: stop_epoch_forward expects dataset_instance to have 'epoch' attribute. "
                "Applying default transforms for this sample."
            )
            return self.default_forward(img_orig, target_orig, dataset_instance)
        cur_epoch = dataset_instance.epoch
        
        # policy 설정 유효성 검사
        if not (self.policy and isinstance(self.policy, dict) and \
                'ops' in self.policy and 'epoch' in self.policy):
            print("Warning: 'policy' or its required keys ('ops', 'epoch') are not correctly defined in Compose. "
                  "Applying all transforms by default (stop_epoch_forward).")
            return self.default_forward(img_orig, target_orig, dataset_instance)

        policy_ops_names = self.policy['ops']
        policy_epoch_thresholds = self.policy['epoch'] # 예: [4, 34, 58]
        
        # Mosaic 적용 여부 결정 (Stage 2 에서만 확률적으로)
        with_mosaic_this_iteration = False
        if isinstance(policy_epoch_thresholds, list) and len(policy_epoch_thresholds) == 3:
            if policy_epoch_thresholds[0] <= cur_epoch < policy_epoch_thresholds[1]: 
                # self.mosaic_prob는 YAML에서 Compose 초기화 시 전달됨
                # (Compose.__init__에서 mosaic_prob이 제대로 설정되었는지 확인 필요)
                current_mosaic_prob = getattr(self, 'mosaic_prob', 0.0) # 안전하게 접근
                if current_mosaic_prob > 0 and random.random() <= current_mosaic_prob:
                    with_mosaic_this_iteration = True
        
        for transform_obj in self.transforms:
            transform_name = type(transform_obj).__name__
            apply_this_transform = True

            # 정책에 따라 특정 변환 건너뛰기 로직
            if transform_name in policy_ops_names:
                if isinstance(policy_epoch_thresholds, list) and len(policy_epoch_thresholds) == 3:
                    # Stage 1 (NoAug 기간) 또는 Stage 4 (NoAug 기간)인지 확인
                    is_in_no_aug_period = (cur_epoch < policy_epoch_thresholds[0]) or \
                                          (cur_epoch >= policy_epoch_thresholds[2]) # policy_epoch[2]는 마지막 NoAug 시작 에폭
                    if is_in_no_aug_period:
                        apply_this_transform = False
                    else: # Stage 2 또는 Stage 3 (증강 적용 기간)
                        if transform_name == 'Mosaic':
                            if not with_mosaic_this_iteration:
                                apply_this_transform = False
                        # DEIM 원본 코드에서는 Mosaic과 RandomZoomOut/RandomIoUCrop을 동시에 적용하지 않음
                        elif (transform_name == 'RandomZoomOut' or transform_name == 'RandomIoUCrop') and with_mosaic_this_iteration:
                            apply_this_transform = False 
                elif isinstance(policy_epoch_thresholds, int): # 단일 에폭 값 정책
                    if cur_epoch >= policy_epoch_thresholds:
                        apply_this_transform = False
            
            if apply_this_transform:
                # print(f"  Epoch {cur_epoch}: Applying {transform_name} ...")
                current_img, current_target = self._apply_single_transform(
                    transform_obj, current_img, current_target, dataset_instance
                )
        
        return current_img, current_target, dataset_instance


    def stop_sample_forward(self, *inputs: Any):
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        self.global_samples += 1

        return sample
