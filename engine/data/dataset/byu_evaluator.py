# DEIM/engine/data/dataset/byu_evaluator.py
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import os

from ...core import register # engine.core.workspace.register
# from ...misc import dist_utils # 만약 분산 환경에서 평가한다면 필요

# 평가 지표 이미지에서 F-beta 공식 확인
# F_beta = (1 + beta^2) * (TP) / ((1 + beta^2) * TP + beta^2 * FN + FP)
# beta = 2

@register() # DEIM 시스템에 등록
class BYU2DEvaluator:
    __inject__ = [] # YAML에서 주입받을 파라미터가 있다면 명시

    def __init__(self, distance_threshold_angstroms: float = 1000.0, beta: float = 2.0):
        self.distance_threshold_angstroms = distance_threshold_angstroms
        self.beta = beta
        self.beta_squared = beta ** 2

        # 토모그램별 예측 결과를 저장할 딕셔너리
        # key: tomo_id
        # value: list of tuples (score, pred_z, pred_y, pred_x, slice_idx, orig_tomo_dims_zxy, voxel_spacing_angstroms)
        self.predictions_per_tomo = defaultdict(list)

        # 토모그램별 실제 모터 위치를 저장할 딕셔너리
        # key: tomo_id
        # value: list of actual motor coords [z_orig, y_orig, x_orig] (이미 원본 좌표계)
        #        또는 모터가 없으면 빈 리스트
        self.ground_truths_per_tomo = {}

        # 이미 처리한 ground truth tomo_id를 저장하여 중복 로드 방지
        self.processed_gt_tomos = set()


    def update(self, predictions):
        """
        매 배치(또는 일부 데이터)의 예측 결과를 누적합니다.
        predictions (dict): 모델의 PostProcessor 출력을 DetEngine의 evaluate 함수에서 가공한 형태.
                           key: image_id (여기서는 Dataset의 슬라이스 아이템 인덱스)
                           value (dict): {
                               'scores': tensor (예측 신뢰도),
                               'labels': tensor (예측 클래스, 여기서는 모터=0),
                               'boxes': tensor (예측 2D 박스 [cx, cy, w, h] in normalized 0-1),
                               # BYUDataset2DSlices에서 추가한 정보들
                               'tomo_id': str,
                               'slice_idx_in_tomo': tensor (int),
                               'orig_tomo_dims_zxy': tensor ([Z0, H0, W0]),
                               'voxel_spacing_angstroms': tensor (float)
                           }
                           (주의: 'boxes'는 PostProcessor를 거치면서 0-1 정규화된 좌표일 수 있음.
                            Dataset에서 픽셀 단위로 줬다면, PostProcessor가 어떻게 처리하는지 확인 필요.
                            만약 정규화되었다면, 원래 512x512 슬라이스 크기로 다시 변환 필요)
        """
        for img_id, pred_dict in predictions.items():
            tomo_id = pred_dict['tomo_id'] # Dataset에서 전달된 문자열 tomo_id
            slice_idx = pred_dict['slice_idx_in_tomo'].item()
            orig_dims = pred_dict['orig_tomo_dims_zxy'] # (Z0, H0, W0) 텐서
            voxel_spacing = pred_dict['voxel_spacing_angstroms'].item() # Å

            # 현재 슬라이스에 대한 예측 정보 저장
            for i in range(len(pred_dict['scores'])):
                score = pred_dict['scores'][i].item()
                label = pred_dict['labels'][i].item() # 모터 클래스 ID (예: 0)

                # 우리 대회는 단일 클래스(모터)이므로, label이 모터 클래스와 일치하는지 확인
                if label == 0: # YAML에서 num_classes=1로 설정했으므로 모터 클래스는 0
                    x1, y1, x2, y2 = pred_dict['boxes'][i].tolist() # PostProcessor가 이미 [x1, y1, x2, y2] 픽셀 좌표를 반환

                    # 중심점(cx, cy)를 픽셀 단위로 계산
                    pred_x_pixel = (x1 + x2) / 2.0
                    pred_y_pixel = (y1 + y2) / 2.0


                    self.predictions_per_tomo[tomo_id].append(
                        (score, slice_idx, pred_y_pixel, pred_x_pixel, slice_idx, orig_dims, voxel_spacing)
                    )

            # Ground truth 로드 (한번만)
            if tomo_id not in self.processed_gt_tomos:
                # BYUDataset2DSlices에서 모든 GT 정보를 가져올 수 있도록 수정하거나,
                # 여기서 df_labels_all_tomos를 직접 참조 (Dataset과 동일한 파일 사용 가정)
                # 이 Evaluator가 dataset 객체를 직접 참조할 수 있다면 더 좋음.
                # 여기서는 dataset에 접근할 수 없다고 가정하고, ann_file을 다시 로드하거나 미리 로드.
                # (가장 좋은 방법은 evaluate 함수에서 gt를 같이 넘겨주거나, dataset 객체를 evaluator에 전달)
                # 임시로, Dataset과 동일한 df_labels_all_tomos를 사용한다고 가정
                # (실제로는 DetEngine의 evaluate 함수에서 ground truth를 준비해서 같이 넘겨줘야 함)

                # 이 부분은 DetEngine의 evaluate 함수에서 실제 라벨을 가져와서 한번만 저장하도록 수정 필요.
                # 여기서는 update 호출 시 ground truth가 있다고 가정하지 않음.
                # 대신, summarize() 할 때 ground truth를 로드하여 매칭.
                self.processed_gt_tomos.add(tomo_id)


    def _load_ground_truth_for_tomo(self, tomo_id, df_labels_gt, original_shapes_map):
        """특정 tomo_id에 대한 원본 3D ground truth 좌표를 로드 (이미 원본 좌표계)."""
        if tomo_id in self.ground_truths_per_tomo:
            return self.ground_truths_per_tomo[tomo_id]

        gt_coords_orig = []
        motor_locations = df_labels_gt[df_labels_gt['tomo_id'] == tomo_id]
        if not motor_locations.empty:
            for _, row in motor_locations.iterrows():
                # train_labels_original.csv의 Motor axis 0,1,2 사용
                if row['Motor axis 0'] != -1: # -1은 모터 없음
                    # Z, Y, X 순서로 저장
                    gt_coords_orig.append([
                        float(row['Motor axis 0']),
                        float(row['Motor axis 1']),
                        float(row['Motor axis 2'])
                    ])
        self.ground_truths_per_tomo[tomo_id] = gt_coords_orig
        return gt_coords_orig


    def summarize(self, df_labels_original_path: str = None, original_shapes_map_for_gt: dict = None):
        """
        모든 예측 결과를 종합하여 최종 평가 지표를 계산하고 요약.
        df_labels_original_path: train_labels_original.csv 파일 경로. 평가 시 GT 로드용.
        original_shapes_map_for_gt: tomo_id별 원본 shape 정보. (Dataset에서 사용한 것과 동일)
        """
        if df_labels_original_path is None or original_shapes_map_for_gt is None:
            # 실전에서는 DetSolver나 main_worker에서 dataset 객체를 통해 이 정보들을 전달받아야 함.
            # 여기서는 BYUDataset2DSlices와 유사하게 로드 (임시)
            print("Warning: Ground truth path or shapes map not provided to summarizer. Trying to load from default paths...")
            # 경로는 BYUDataset2DSlices의 경로 설정을 참고하여 맞추어야 함
            # 예: default_gt_path = os.path.join('./', 'train_labels_original.csv')
            #     default_shapes = ... (BYUDataset2DSlices에서 로드하는 방식과 동일하게)
            # 여기서는 외부에서 전달된다고 가정하고, 없으면 에러 대신 빈 결과 반환 시도.
            if not hasattr(self, 'df_labels_gt_all'): # 한번만 로드
                try:
                    # 이 부분은 실제로는 DetSolver에서 dataset.df_labels_original 등을 전달받아야 함
                    self.df_labels_gt_all = pd.read_csv(os.path.join('./', 'train_labels_original.csv')) # 경로 주의!
                    self.original_shapes_for_gt_eval = {}
                    for _, row in self.df_labels_gt_all.drop_duplicates(subset=['tomo_id']).iterrows():
                         self.original_shapes_for_gt_eval[row['tomo_id']] = {
                            'Z0_orig': row['Array shape (axis 0)'],
                            'H0_orig': row['Array shape (axis 1)'],
                            'W0_orig': row['Array shape (axis 2)']
                        }
                except FileNotFoundError:
                    print("Error: Could not load ground truth for summarization.")
                    return {"F_beta": 0.0, "Precision": 0.0, "Recall": 0.0, "TP": 0, "FP": 0, "FN": 0}
        else:
            # 여기서 df_labels_original_path, original_shapes_map_for_gt 둘 다 정상
            # 1) df_labels_original_path로 GT CSV를 로드
            self.df_labels_gt_all = pd.read_csv(df_labels_original_path)
            self.original_shapes_for_gt_eval = original_shapes_map_for_gt

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        # 모든 tomo_id (예측이 있거나 GT가 있는 모든 토모그램)
        all_eval_tomo_ids = set(self.predictions_per_tomo.keys()) | set(self.df_labels_gt_all['tomo_id'].unique())


        for tomo_id in all_eval_tomo_ids:
            # 1. Ground Truth 준비 (원본 좌표계)
            # self._load_ground_truth_for_tomo(tomo_id, self.df_labels_gt_all, self.original_shapes_for_gt_eval)
            # ground_truth_motors_orig = self.ground_truths_per_tomo.get(tomo_id, [])
            # 수정: 바로 gt coords 가져오기
            gt_motor_rows = self.df_labels_gt_all[self.df_labels_gt_all['tomo_id'] == tomo_id]
            ground_truth_motors_orig = []
            if not gt_motor_rows.empty:
                for _, row in gt_motor_rows.iterrows():
                    if row['Motor axis 0'] != -1:
                        ground_truth_motors_orig.append([
                            float(row['Motor axis 0']), float(row['Motor axis 1']), float(row['Motor axis 2'])
                        ])


            # 2. 해당 tomo_id에 대한 최종 3D 예측 결정
            final_pred_z_orig, final_pred_y_orig, final_pred_x_orig = -1, -1, -1
            best_score = -1.0

            if tomo_id in self.predictions_per_tomo:
                tomo_predictions = self.predictions_per_tomo[tomo_id]
                if tomo_predictions: # 해당 토모그램에 대한 예측이 있는 경우
                    # 가장 높은 신뢰도를 가진 예측 선택
                    # (score, slice_idx, pred_y_pixel, pred_x_pixel, _, orig_dims_tensor, voxel_spacing)
                    tomo_predictions.sort(key=lambda x: x[0], reverse=True)
                    best_pred_info = tomo_predictions[0]

                    best_score = best_pred_info[0]
                    pred_z_slice = best_pred_info[1] # 예측된 Z 슬라이스 인덱스 (0-127)
                    pred_y_pixel = best_pred_info[2] # 512x512 슬라이스 내 Y 픽셀 좌표
                    pred_x_pixel = best_pred_info[3] # 512x512 슬라이스 내 X 픽셀 좌표
                    orig_dims_zxy = best_pred_info[5].tolist() # [Z0, H0, W0]
                    # voxel_spacing = best_pred_info[6] # 이미 Å 단위

                    # 3. 예측 좌표를 원본 토모그램 좌표계로 변환 (Readme.md 참고)
                    # z = ẑ * Z0 / 128
                    # y = ŷ * H0 / 512
                    # x = x̂ * W0 / 512
                    # 여기서 ẑ는 pred_z_slice, ŷ는 pred_y_pixel, x̂는 pred_x_pixel
                    # Z0, H0, W0는 orig_dims_zxy
                    # 128, 512, 512는 리사이즈된 .npy 볼륨의 D, H, W
                    Z0, H0, W0 = orig_dims_zxy[0], orig_dims_zxy[1], orig_dims_zxy[2]
                    resized_D, resized_H, resized_W = 128, 512, 512

                    final_pred_z_orig = pred_z_slice * Z0 / resized_D
                    final_pred_y_orig = pred_y_pixel * H0 / resized_H
                    final_pred_x_orig = pred_x_pixel * W0 / resized_W

            # 4. 모터 유무 판별을 위한 신뢰도 임계값 (Phase 1에서 결정된 값 사용)
            # 여기서는 예시로 0.5 사용. 실제로는 검증 세트에서 튜닝한 값 사용.
            # 이 임계값은 YAML에서 evaluator 파라미터로 받거나, solver에서 설정해줄 수 있음.
            # evaluator:
            #   type: BYU2DEvaluator
            #   confidence_threshold: 0.5 # 예시
            confidence_threshold = getattr(self, 'confidence_threshold', 0.5) # YAML에서 설정 가능하도록

            motor_predicted = (best_score >= confidence_threshold)

            # 5. TP, FP, FN 계산
            has_gt_motor = len(ground_truth_motors_orig) > 0

            if motor_predicted:
                if has_gt_motor:
                    # 예측했고, 실제 모터도 있음 -> 거리 계산
                    # 이 대회는 테스트셋에 모터 0 또는 1개. 훈련셋은 여러개 가능.
                    # 여기서는 GT가 여러개여도 첫번째 GT와 비교 (테스트셋 기준).
                    # 또는 가장 가까운 GT와 비교. 여기서는 첫번째 GT와 비교 가정.
                    gt_z, gt_y, gt_x = ground_truth_motors_orig[0]
                    
                    # 물리적 거리 계산 (이미 모든 좌표는 원본 복셀 좌표계)
                    # voxel_spacing은 GT쪽이 아니라 예측쪽 볼륨의 것을 사용해야 함 (평가를 위해선 일관된 기준 필요)
                    # 하지만 각 토모그램은 고유의 voxel_spacing을 가짐.
                    # 거리는 ( (pred_z - gt_z)*spacing_z )^2 + ... 로 계산되어야 하나,
                    # 대회 규칙은 유클리드 거리 (픽셀) * 복셀 간격 = 물리적 거리.
                    # 즉, (pred_z - gt_z), (pred_y - gt_y), (pred_x - gt_x)는 복셀 단위 차이.
                    # 이 복셀 단위 차이에 복셀 간격(Å/voxel)을 곱해야 물리적 차이(Å).
                    # voxel_spacing_map에서 해당 tomo_id의 복셀 간격을 가져와야 함.
                    # BYUDataset2DSlices에서 이미 voxel_spacing_angstroms를 넘겨주므로,
                    # best_pred_info[6]에 저장되어 있음.
                    current_voxel_spacing = best_pred_info[6] if 'best_pred_info' in locals() and best_pred_info[6] is not None else self.voxel_spacing_map.get(tomo_id, 10.0)


                    dist_voxels = np.sqrt(
                        (final_pred_z_orig - gt_z)**2 +
                        (final_pred_y_orig - gt_y)**2 +
                        (final_pred_x_orig - gt_x)**2
                    )
                    dist_angstroms = dist_voxels * current_voxel_spacing

                    if dist_angstroms <= self.distance_threshold_angstroms:
                        true_positives += 1
                    else:
                        # 예측했지만, 실제 모터와 너무 멈 (또는 다른 모터로 잘못 예측 - 여기선 모터 1개 가정)
                        false_positives += 1 # 예측은 했으므로 FP
                        false_negatives += 1 # 실제 GT 모터를 못 찾았으므로 FN도 카운트 (대회 평가 방식 확인 필요)
                                             # 평가 이미지(image_724779.png)에서는 FN = distance > threshold if y exists
                                             # 이므로, GT가 있는데 예측이 멀면 FN.
                else:
                    # 예측했지만, 실제 모터 없음
                    false_positives += 1
            else: # motor_predicted == False
                if has_gt_motor:
                    # 예측 안 했지만, 실제 모터 있음
                    false_negatives += 1
                else:
                    # 예측 안 했고, 실제 모터도 없음 -> True Negative (F-beta 계산에는 사용 안됨)
                    pass
        
        # 6. 최종 F-beta 점수 계산
        # F_beta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # From image_724779.png: F_beta = (1 + beta^2) * TP / ((1 + beta^2) * TP + beta^2 * FN + FP)

        if ( (1 + self.beta_squared) * true_positives + self.beta_squared * false_negatives + false_positives ) == 0:
            f_beta_score = 0.0
        else:
            f_beta_score = (1 + self.beta_squared) * true_positives / \
                           ( (1 + self.beta_squared) * true_positives + self.beta_squared * false_negatives + false_positives )

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        results = {
            "F_beta": f_beta_score,
            "Precision": precision,
            "Recall": recall,
            "TP": true_positives,
            "FP": false_positives,
            "FN": false_negatives
        }
        print(f"Evaluation Results: {results}")
        return results

    def synchronize_between_processes(self):
        """분산 학습 환경에서 모든 프로세스의 결과를 동기화 (필요시 구현)"""
        # if not dist_utils.is_dist_available_and_initialized():
        #     return
        # # self.predictions_per_tomo 등을 all_gather로 모으고, main_process에서만 summarize
        pass

    def accumulate(self):
        """모든 예측 누적 (이미 update에서 수행)"""
        pass