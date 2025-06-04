"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable
import torch
from torch.utils.tensorboard import SummaryWriter # 현재 미사용으로 주석 처리 또는 필요시 활성화
# from torch.cuda.amp.grad_scaler import GradScaler # kwargs로 받으므로 직접 import 불필요
from tqdm import tqdm


# engine.optim 및 engine.misc 경로가 solver 모듈 기준으로 올바른지 확인
try:
    from ..optim import ModelEMA # Warmup은 FlatCosine에 내장될 수 있으므로 직접 사용 안 할 수 있음
    from ..misc import MetricLogger, SmoothedValue, dist_utils
    from ..data.dataset.byu_evaluator import BYU2DEvaluator
    from ..optim.lr_scheduler import FlatCosineLRScheduler # 타입 체크용
except ImportError: # IDE 등에서 상대 경로 인식 못할 경우 대비
    from engine.optim import ModelEMA
    from engine.misc import MetricLogger, SmoothedValue, dist_utils
    from engine.data.dataset.byu_evaluator import BYU2DEvaluator
    from engine.optim.lr_scheduler import FlatCosineLRScheduler


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    lr_scheduler_to_use=None,
    lr_warmup_scheduler_to_use=None,
    iter_per_epoch=0,
    print_freq=10,
    writer=None,
    scaler=None,
    ema=None
):
    model.train()
    criterion.train()

    # metric_logger는 계속 사용 가능 (통계 계산, 마지막에 평균 로그)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    # 1) TQDM으로 data_loader 감싸서 진행바 표시
    #    total=len(data_loader)가 batch 수를 의미
    pbar = tqdm(enumerate(data_loader), desc=header, total=len(data_loader), leave=False)

    for i, (samples, targets) in pbar:
        samples = samples.to(device)

        device_targets = []
        for t in targets:
            target_on_device = {}
            for k, v in t.items():
                if isinstance(v, torch.Tensor):
                    target_on_device[k] = v.to(device)
                else:
                    target_on_device[k] = v
            device_targets.append(target_on_device)

        # (1) 웜업 스케줄러
        if lr_warmup_scheduler_to_use is not None:
            if epoch == 0 and hasattr(lr_warmup_scheduler_to_use, 'finished') and not lr_warmup_scheduler_to_use.finished():
                lr_warmup_scheduler_to_use.step()
            elif epoch == 0 and not hasattr(lr_warmup_scheduler_to_use, 'finished'):
                lr_warmup_scheduler_to_use.step()

        # (2) autocast
        enabled_autocast = (scaler is not None)
        autocast_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=enabled_autocast):
            outputs = model(samples, device_targets)
            for t_idx, tgt in enumerate(device_targets):
                if "boxes" in tgt and tgt["boxes"].numel() > 0:
                    # 예: matcher는 cxcywh 기반이라고 가정 -> (cx,cy,w,h) => (x1,y1,x2,y2) 변환
                    # 만약 이미 xyxy라면 box_xyxy = tgt["boxes"] 사용
                    # 여기서는 cxcywh라고 가정
                    from ..deim.box_ops import box_cxcywh_to_xyxy

                    boxes_cxcywh = tgt["boxes"]
                    # 혹은 boxes가 BoundingBoxes 객체라면 .as_subclass(torch.Tensor) 등으로 tensor 얻기
                    # boxes_cxcywh_t = boxes_cxcywh.as_subclass(torch.Tensor)

                    # 변환
                    box_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)

                    x1, y1, x2, y2 = box_xyxy.unbind(-1)
                    broken_mask = (x2 < x1) | (y2 < y1)
                    if broken_mask.any():
                        print(f"[DEBUG][Epoch={epoch}][Iter={i}] Found broken box in device_targets[{t_idx}]:")
                        print("box_xyxy[broken_mask] =", box_xyxy[broken_mask])
                        # 필요하면 sys.exit(1)로 중단, 혹은 continue
            loss_dict = criterion(outputs, device_targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training.", loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # (3) 메인 LR 스케줄러
        if lr_scheduler_to_use is not None and isinstance(lr_scheduler_to_use, FlatCosineLRScheduler):
            current_total_iter = epoch * iter_per_epoch + i
            lr_scheduler_to_use.step(current_iter=current_total_iter, optimizer=optimizer)

        # LR 프린트 (param_groups가 여러 개면 리스트로 출력)
        # if i % print_freq == 0:
        #     lr_values = [pg["lr"] for pg in optimizer.param_groups]
        #     print(f"[Epoch={epoch}][Iter={i}] LR(s)={lr_values}")

        if ema is not None:
            ema.update(model)

        torch.cuda.synchronize()

        # metric_logger 업데이트
        metric_logger.update(loss=loss_value)
        for k_loss, v_loss in loss_dict.items():
            if k_loss in weight_dict:
                metric_logger.update(**{f'loss_{k_loss}': v_loss.item()})
        current_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=current_lr)

        # 2) TQDM 진행바에 현재 loss, lr 표시
        if i % print_freq == 0:
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr": f"{current_lr:.6f}"
            })

        # 3) TensorBoard 기록
        #    (iter_per_epoch가 0 이상이라면 global_step을 "epoch * iter_per_epoch + i"로 계산)
        global_step = epoch * iter_per_epoch + i
        if writer and dist_utils.is_main_process():
            # print_freq마다 기록해도 되고, 매 step 기록해도 됨
            if i % print_freq == 0:
                writer.add_scalar('train/iter_loss', loss_value, global_step)
                writer.add_scalar('train/iter_lr', current_lr, global_step)

    # Epoch가 끝날 때 평균 스탯 요약
    print_str = f"==> [Epoch {epoch}]  "
    for k, meter in metric_logger.meters.items():
        print_str += f"{k}: {meter.global_avg:.4f}  "
        # TensorBoard에 epoch 단위로도 기록할 수 있음
        if writer and dist_utils.is_main_process():
            writer.add_scalar(f'train_epoch/{k}', meter.global_avg, epoch)

    print(print_str)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor: torch.nn.Module,
    data_loader: Iterable,
    evaluator: BYU2DEvaluator,
    device: torch.device,
    writer=None,           # 추가: TensorBoard writer
    epoch: int = 0,        # 추가: Epoch 정보를 받아 로깅에 활용 가능
    print_freq=10,
    **kwargs
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    
   # TQDM으로 감싸서 진행바 표시
    pbar = tqdm(enumerate(data_loader), desc=header, total=len(data_loader), leave=False)

    for i, (samples, targets) in pbar:
        samples = samples.to(device)
        outputs = model(samples)  # 추론

        if criterion is not None:
            loss_dict = criterion(outputs, targets, epoch=-1)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_value = losses.item()
            metric_logger.update(loss=loss_value)

            # TQDM 진행바 업데이트
            if i % print_freq == 0:
                pbar.set_postfix({"val_loss": f"{loss_value:.4f}"})

        # PostProcessor 적용
        # PostProcessor는 (outputs, orig_target_sizes)를 받음
        # orig_target_sizes는 각 2D 슬라이스의 원본 크기 (여기서는 512x512)
        # BYUDataset2DSlices의 target에는 'spatial_size' (H,W)가 있음
        slice_spatial_sizes = torch.stack([t['spatial_size'] for t in targets]).to(device) # (B, 2) 형태, H,W 순서
        results = postprocessor(outputs, slice_spatial_sizes) # results: list of dicts, 각 dict는 슬라이스별 예측

        # Evaluator 업데이트용 데이터 준비
        eval_inputs_for_updater = {}
        for i, res_dict_per_slice in enumerate(results): # 배치 내 각 슬라이스 결과
            target_from_dataset = targets[i] # 해당 슬라이스의 GT 및 메타데이터
            # evaluator.update()가 받는 형식에 맞춰서 재구성
            # image_id는 dataset 내의 unique index (슬라이스 아이템의 인덱스)
            img_id_from_dataset = target_from_dataset['image_id'].item() # Dataset에서 제공한 고유 인덱스

            eval_inputs_for_updater[img_id_from_dataset] = {
                'scores': res_dict_per_slice['scores'].cpu(),
                'labels': res_dict_per_slice['labels'].cpu(),
                'boxes': res_dict_per_slice['boxes'].cpu(), # PostProcessor 출력 박스 좌표 (픽셀 단위, cxcywh 또는 xyxy)
                # Dataset에서 온 메타데이터
                'tomo_id': target_from_dataset['tomo_id'], # 문자열
                'slice_idx_in_tomo': target_from_dataset['slice_idx_in_tomo'].cpu(),
                'orig_tomo_dims_zxy': target_from_dataset['orig_tomo_dims_zxy'].cpu(),
                'voxel_spacing_angstroms': target_from_dataset['voxel_spacing_angstroms'].cpu()
            }
        
        if evaluator is not None:
            evaluator.update(eval_inputs_for_updater)

    metric_logger.synchronize_between_processes() # 분산 환경용
    print("Averaged stats (from MetricLogger):", metric_logger) # 학습 중 손실 등 로깅

    # TensorBoard에 검증 손실 등을 epoch 단위로 로그
    if writer and dist_utils.is_main_process():
        for k, meter in metric_logger.meters.items():
            writer.add_scalar(f'val_epoch/{k}', meter.global_avg, epoch)
    
    final_eval_results = {}
    if evaluator is not None:
        evaluator.synchronize_between_processes()
        # evaluator.accumulate() # CocoEvaluator에는 있지만, 우리 BYU2DEvaluator에는 필요 없을 수 있음
        
        # summarize 호출 시 필요한 GT 정보 파일 경로 및 원본 shape 맵 전달
        # 이 정보는 data_loader.dataset 에서 가져올 수 있음
        df_labels_original_path = data_loader.dataset.original_labels_csv_path
        # original_shapes_map은 Dataset의 self.original_shapes를 직접 사용
        original_shapes_map_for_gt = data_loader.dataset.original_shapes

        final_eval_results = evaluator.summarize(
            df_labels_original_path=df_labels_original_path,
            original_shapes_map_for_gt=original_shapes_map_for_gt
        )
        if final_eval_results:
             print(f"BYU Specific Evaluation Results: {final_eval_results}")
    
    # 반환값은 DetSolver에서 사용할 수 있도록 조정
    # CocoEvaluator는 coco_eval['bbox'].stats 등을 반환
    # 우리는 final_eval_results (dict) 와 evaluator 객체 자체를 반환
    return final_eval_results, evaluator