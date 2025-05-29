"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def stats(
    cfg,
    default_input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:
    input_shape_to_use = None

    # 1. YAML 설정의 'eval_spatial_size'를 최우선으로 사용 (H, W 순서로 가정)
    # cfg.yaml_cfg는 YAMLConfig 인스턴스일 때만 존재. cfg가 BaseConfig일 수도 있음.
    # cfg 객체가 YAMLConfig 인스턴스인지 확인하거나, 안전하게 getattr 사용.
    eval_size_from_yaml = None
    if hasattr(cfg, 'yaml_cfg') and isinstance(cfg.yaml_cfg, dict): # cfg.yaml_cfg가 dict인지 확인
        eval_size_from_yaml = cfg.yaml_cfg.get('eval_spatial_size', None)
    elif hasattr(cfg, 'eval_spatial_size'): # cfg 객체에 직접 속성이 있는 경우
        eval_size_from_yaml = cfg.eval_spatial_size

    if isinstance(eval_size_from_yaml, list) and len(eval_size_from_yaml) == 2:
        h, w = eval_size_from_yaml # YAML에서 [H, W] 순서로 정의되었다고 가정
        input_shape_to_use = (1, 3, h, w)
        print(f"FLOPs calculation: Using eval_spatial_size from YAML: {input_shape_to_use}")

    # 2. 'eval_spatial_size'가 없으면, train_dataloader.collate_fn.base_size 사용
    if input_shape_to_use is None:
        if hasattr(cfg, 'train_dataloader') and \
           hasattr(cfg.train_dataloader, 'collate_fn') and \
           hasattr(cfg.train_dataloader.collate_fn, 'base_size'): # base_size 속성 존재 확인
            base_size = cfg.train_dataloader.collate_fn.base_size
            if isinstance(base_size, int):
                input_shape_to_use = (1, 3, base_size, base_size)
                print(f"FLOPs calculation: Using collate_fn.base_size: {input_shape_to_use}")

    # 3. 위 두 가지 모두 없으면, 함수 정의 시의 default_input_shape 사용
    if input_shape_to_use is None:
        input_shape_to_use = default_input_shape
        print(f"FLOPs calculation: Using default_input_shape: {input_shape_to_use}")
    
    # 모델 복사 및 deploy 모드 설정
    # cfg.model은 YAMLConfig의 프로퍼티를 통해 모델 인스턴스를 반환
    if cfg.model is None:
        raise ValueError("cfg.model is None. Model has not been initialized in YAMLConfig.")
    model_for_info = copy.deepcopy(cfg.model)
    if hasattr(model_for_info, 'deploy'): # deploy 메서드가 있는지 확인
        model_for_info.deploy()
    else:
        model_for_info.eval() # deploy 없으면 eval()이라도 호출

    flops, macs, _ = calculate_flops(model=model_for_info,
                                     input_shape=input_shape_to_use, # 결정된 input_shape 사용
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=False,
                                     include_backPropagation=False) # 순방향 연산만 계산

    # 파라미터 수 계산 (학습 가능한 파라미터만)
    params = sum(p.numel() for p in model_for_info.parameters() if p.requires_grad)
    # 또는 전체 파라미터: params = sum(p.numel() for p in model_for_info.parameters())
    
    del model_for_info # 메모리 해제

    return params, {"Model FLOPs": flops, "MACs": macs, "Params (trainable)": params}
