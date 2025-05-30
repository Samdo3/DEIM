# DEIM/engine/data/dataset/byu_dataset_2d_slices.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import copy
from sklearn.model_selection import train_test_split # 추가

from ._dataset import DetDataset # DEIM 프로젝트 내의 올바른 상대 경로로 수정
from ...core import register      # DEIM 프로젝트 내의 올바른 상대 경로로 수정
from .._misc import convert_to_tv_tensor, _boxes_keys, Image as TVImage_from_misc, BoundingBoxes as TVBoundingBoxes_from_misc

# 위 import 문들이 실제 프로젝트 구조에 맞게 되어 있다고 가정하고,
# 필요한 torchvision.tv_tensors 관련 클래스를 직접 import 합니다.
from torchvision.tv_tensors import Image as TVImage, BoundingBoxes, BoundingBoxFormat
from engine.data._misc import _boxes_keys # engine.core.register 와 같은 레벨의 engine.data._misc
from engine.core import register

@register()
class BYUDataset2DSlices(Dataset): # DetDataset 대신 torch.utils.data.Dataset을 직접 상속받도록 수정 (DetDataset의 __getitem__ 로직과 충돌 방지)
    __inject__ = ['transforms']

    def __init__(self,
                 data_dir: str,                 # YAML에서: ./ (DEIM 루트)
                 ann_file_name: str,            # YAML에서: train_labels.csv (리사이징된 좌표)
                 voxel_spacing_file_name: str,  # YAML에서: train_voxel_spacing.csv
                 original_labels_file_name: str,# YAML에서: train_labels_original.csv
                 transforms=None,
                 is_train: bool = True,
                 train_val_split_ratio: float = 0.9,
                 random_seed: int = 42,
                 virtual_box_wh: tuple = (10, 10),
                 # 추가: 최종 이미지 크기 (YAML에서 설정하거나, 고정값 사용)
                 output_size_h: int = 512,
                 output_size_w: int = 512
                ):
        # super().__init__() # DetDataset 상속 안 하므로 제거 또는 Dataset의 init 호출
        self.data_dir = data_dir 
        self.transforms = transforms
        self.is_train = is_train
        self.virtual_box_w, self.virtual_box_h = virtual_box_wh
        self.random_seed = random_seed
        self.train_val_split_ratio = train_val_split_ratio
        self.output_size_h = output_size_h
        self.output_size_w = output_size_w

        # 파일 경로 조합
        self.ann_file_path = os.path.join(self.data_dir, ann_file_name)
        self.voxel_spacing_file_path = os.path.join(self.data_dir, voxel_spacing_file_name)
        self.original_labels_csv_path = os.path.join(self.data_dir, original_labels_file_name)

        # CSV 파일 로드
        try:
            self.df_labels_src = pd.read_csv(self.ann_file_path) # 모든 라벨 (분할 전)
            df_voxel_spacing = pd.read_csv(self.voxel_spacing_file_path)
            self.voxel_spacing_map = pd.Series(df_voxel_spacing.voxel_spacing_angstroms.values,
                                               index=df_voxel_spacing.tomo_id).to_dict()

            df_original_labels = pd.read_csv(self.original_labels_csv_path)
            self.original_shapes = {} 
            for _, row in df_original_labels.drop_duplicates(subset=['tomo_id']).iterrows():
                # 원본 CSV의 컬럼명이 'Array shape axis 0' 등으로 되어 있을 수 있으므로 확인 필요
                self.original_shapes[row['tomo_id']] = {
                    'Z0_orig': row.get('Array shape axis 0', row.get('Array shape (axis 0)', 128)), # 컬럼명 유연하게 처리
                    'H0_orig': row.get('Array shape axis 1', row.get('Array shape (axis 1)', 512)),
                    'W0_orig': row.get('Array shape axis 2', row.get('Array shape (axis 2)', 512))
                }
        except FileNotFoundError as e:
            print(f"오류: CSV 파일 경로를 확인하세요. {e}")
            print(f"  ann_file_path: {os.path.abspath(self.ann_file_path)}")
            print(f"  voxel_spacing_file_path: {os.path.abspath(self.voxel_spacing_file_path)}")
            print(f"  original_labels_csv_path: {os.path.abspath(self.original_labels_csv_path)}")
            raise

        # 토모그램 파일 목록 및 ID 수집
        tomo_file_paths_all = []
        tomo_ids_all = []
        for folder_name in ['motor_0', 'motor_1']:
            current_dir = os.path.join(self.data_dir, folder_name)
            if os.path.exists(current_dir):
                for f_name in sorted(os.listdir(current_dir)):
                    if f_name.endswith('.npy'):
                        tomo_file_paths_all.append(os.path.join(current_dir, f_name))
                        tomo_ids_all.append(os.path.splitext(f_name)[0])
            else:
                print(f"Warning: Directory not found - {os.path.abspath(current_dir)}")
        
        if not tomo_file_paths_all:
            raise FileNotFoundError(f"No .npy files found. Check data_dir and folder structure.")

        # 훈련/검증 분할 (토모그램 ID 기준)
        unique_tomo_ids_for_split = sorted(list(set(tomo_ids_all))) # df_labels_src가 아닌 실제 파일 기반 ID 사용
        
        active_tomo_ids_set = set()
        if len(unique_tomo_ids_for_split) > 1 :
            # 계층적 샘플링을 위한 레이블 생성 (tomo_id별 모터 유무)
            # self.df_labels_src는 리사이징된 좌표 파일(train_labels.csv)이므로, z=-1로 판단
            motor_status_per_tomo = self.df_labels_src.groupby('tomo_id').apply(
                lambda g: 1 if (g['z'] != -1).any() else 0
            )
            # unique_tomo_ids_for_split에 없는 tomo_id가 motor_status_per_tomo에 있을 수 있으므로 reindex
            stratify_labels = motor_status_per_tomo.reindex(unique_tomo_ids_for_split, fill_value=0).values

            if len(np.unique(stratify_labels)) < 2: # 계층화할 클래스가 1개뿐일 경우
                train_tomo_ids, val_tomo_ids = train_test_split(
                    unique_tomo_ids_for_split,
                    train_size=self.train_val_split_ratio,
                    random_state=self.random_seed
                )
            else:
                train_tomo_ids, val_tomo_ids = train_test_split(
                    unique_tomo_ids_for_split,
                    train_size=self.train_val_split_ratio,
                    random_state=self.random_seed,
                    stratify=stratify_labels
                )
            
            if self.is_train:
                active_tomo_ids_set = set(train_tomo_ids)
            else:
                active_tomo_ids_set = set(val_tomo_ids)
        elif len(unique_tomo_ids_for_split) == 1: # 데이터가 1개뿐인 경우
            active_tomo_ids_set = set(unique_tomo_ids_for_split)
            if self.is_train : print("Warning: Only 1 tomo_id for training set after split attempt.")
            else: print("Warning: Only 1 tomo_id for validation set after split attempt.")
        else: # 데이터가 아예 없는 경우
            print("Warning: No tomo_ids available for splitting.")


        # 최종 slice_items 구성
        self.slice_items = []
        temp_tomo_id_to_path = {tid: path for tid, path in zip(tomo_ids_all, tomo_file_paths_all)}

        for tomo_id in active_tomo_ids_set:
            if tomo_id not in temp_tomo_id_to_path:
                # print(f"Warning: Tomo_id {tomo_id} from split not found in discovered .npy files. Skipping.")
                continue
            tomo_file_path = temp_tomo_id_to_path[tomo_id]
            num_slices_in_tomo = 128 # 고정값
            for slice_idx in range(num_slices_in_tomo):
                self.slice_items.append((tomo_file_path, slice_idx, tomo_id))

        if not self.slice_items:
            print(f"Warning: No slice items generated for {'train' if self.is_train else 'val'} set. Active Tomo IDs: {len(active_tomo_ids_set)}")
        else:
            print(f"{'훈련' if self.is_train else '검증'} 데이터셋: {len(active_tomo_ids_set)} 토모그램, {len(self.slice_items)} 개별 슬라이스 아이템")
            print(f"  첫 번째 slice_item 예시: {self.slice_items[0] if self.slice_items else '없음'}")

        # # ----- 샘플링 로직 추가 -----
        # if self.is_train:
        #     import random
        #     from collections import defaultdict

        #     # 원하는 슬라이스 수 K
        #     k_slices_per_tomo = 10  
        #     # 원하는 양성:음성 비율 (예: 1:1)
        #     # ratio = 양성 / (양성 + 음성) 이라고 가정
        #     pos_ratio = 0.3

        #     # 1) self.slice_items에 "양성 여부"를 추가로 표시
        #     #    (이미 load_item에서 라벨을 확인하지만, 아래에서 편하게 확인하기 위해 미리 체크)
        #     def is_positive_slice(tomo_id, slice_idx):
        #         # df_labels_src에서 (z == slice_idx) 레코드가 하나라도 존재하면 양성
        #         # z가 float형으로 들어있을 수도 있으므로 round/int 변환
        #         subset = self.df_labels_src[
        #             (self.df_labels_src['tomo_id'] == tomo_id) &
        #             (self.df_labels_src['z'].round().astype(int) == slice_idx)
        #         ]
        #         return not subset.empty  # 비어있지 않으면 양성

        #     # (tomo_file_path, slice_idx, tomo_id) -> (tomo_file_path, slice_idx, tomo_id, is_positive)
        #     slice_items_with_label = []
        #     for (tomo_file_path, slice_idx, tomo_id) in self.slice_items:
        #         slice_items_with_label.append(
        #             (tomo_file_path, slice_idx, tomo_id, is_positive_slice(tomo_id, slice_idx))
        #         )

        #     # 2) 토모그램별로 양성/음성 리스트를 분리
        #     slices_by_tomo = defaultdict(lambda: {"pos": [], "neg": []})
        #     for (tomo_file_path, slice_idx, tomo_id, is_pos) in slice_items_with_label:
        #         if is_pos:
        #             slices_by_tomo[tomo_id]["pos"].append((tomo_file_path, slice_idx, tomo_id))
        #         else:
        #             slices_by_tomo[tomo_id]["neg"].append((tomo_file_path, slice_idx, tomo_id))

        #     new_slice_items = []
        #     # 3) 각 토모그램에서 비율에 맞춰 샘플링
        #     for tid, pos_neg_dict in slices_by_tomo.items():
        #         all_pos = pos_neg_dict["pos"]
        #         all_neg = pos_neg_dict["neg"]
                
        #         # 토모그램별 양성/음성 실제 개수
        #         num_pos = len(all_pos)
        #         num_neg = len(all_neg)
                
        #         if (num_pos + num_neg) == 0:
        #             continue  # 해당 토모그램이 슬라이스가 전혀 없는 경우(이상 케이스)
                
        #         # 여기서는 pos_ratio = 0.5 (즉, 1:1) 예시
        #         # 일반화하려면 pos_ratio와 1-pos_ratio로 나누어 계산
        #         target_pos = int(k_slices_per_tomo * pos_ratio)
        #         target_neg = k_slices_per_tomo - target_pos
                
        #         # 실제로는 토모그램별 양성 개수가 target_pos보다 부족할 수 있으므로 min() 처리
        #         sampled_pos = random.sample(all_pos, min(target_pos, num_pos))
        #         # 양성을 먼저 뽑고, 남은 것은 음성에서 뽑음
        #         # (양성이 부족하면 남은 만큼 음성을 더 뽑을 수도 있지만, 코드 단순화 위해 일단은 target_neg 그대로 뽑음)
        #         sampled_neg = random.sample(all_neg, min(target_neg, num_neg))

        #         # 혹시 "양성이 부족해서" target_pos를 다 못 채운 경우 -> 남은만큼 음성으로 대체
        #         # (사용 여부는 상황에 맞게 결정)
        #         lack_pos = target_pos - len(sampled_pos)
        #         if lack_pos > 0:
        #             # 음성 여유분이 있다면 추가로 뽑기
        #             neg_rest = list(set(all_neg) - set(sampled_neg))  # 아직 안 뽑힌 음성
        #             additional_neg_needed = min(lack_pos, len(neg_rest))
        #             if additional_neg_needed > 0:
        #                 sampled_neg_extra = random.sample(neg_rest, additional_neg_needed)
        #                 sampled_neg.extend(sampled_neg_extra)
                
        #         # 반대로 "음성이 부족해서" target_neg를 다 못 채운 경우 -> 양성으로 대체해도 되는지?
        #         # 필요하다면 비슷한 로직을 추가
                
        #         final_slices = sampled_pos + sampled_neg
                
        #         # K개보다 적을 수도 있으므로, 다시 한 번 K개로 자르기
        #         if len(final_slices) > k_slices_per_tomo:
        #             final_slices = random.sample(final_slices, k_slices_per_tomo)
                
        #         new_slice_items.extend(final_slices)

        #     old_count = len(self.slice_items)
        #     self.slice_items = new_slice_items
        #     new_count = len(self.slice_items)
        #     print(f"[토모그램별 샘플링+비율] 전체 {old_count}개 슬라이스 -> {new_count}개 "
        #           f"(각 토모그램당 최대 {k_slices_per_tomo}개, 양성:음성 비율 {pos_ratio}:"
        #           f"{round(1.0 - pos_ratio,2)})")
        # # --------------------------------


    def __len__(self):
        return len(self.slice_items)

    def load_item(self, idx):
        tomo_file_path, slice_idx, tomo_id = self.slice_items[idx]
        
        try:
            # .npy 파일은 이미 (128, 512, 512) 형태이고, 각 슬라이스는 (512, 512)
            # mmap_mode='r'은 원본 Readme.md 코드에 없었으므로, 일반적인 로드로 변경
            vol_data = np.load(tomo_file_path) # (128, 512, 512)
            slice_data_np = vol_data[slice_idx].astype(np.float32) # (512, 512)
        except Exception as e:
            print(f"Error loading or slicing .npy file {tomo_file_path} at index {idx}, slice {slice_idx}: {e}")
            # 임시로 빈 이미지 반환 또는 오류 발생
            slice_data_np = np.zeros((self.output_size_h, self.output_size_w), dtype=np.float32)


        # img_tensor: (C, H, W), float32. 모델 입력에 맞게 0-1 또는 정규화된 값이어야 함.
        # 여기서는 0-255 범위로 가정하고, Normalize에서 처리.
        img_tensor = torch.from_numpy(slice_data_np).unsqueeze(0).repeat(3, 1, 1) # (3, H, W)

        # 라벨 정보 가져오기 (리사이징된 좌표 사용 - train_labels.csv)
        motor_on_this_slice = self.df_labels_src[
            (self.df_labels_src['tomo_id'] == tomo_id) &
            (self.df_labels_src['z'].round().astype(int) == slice_idx) # z는 리사이징된 슬라이스 인덱스
        ]

        boxes_cxcywh_list = []
        labels_list = []
        if not motor_on_this_slice.empty:
            for _, motor_row in motor_on_this_slice.iterrows():
                # train_labels.csv의 y, x는 이미 512x512 기준 좌표
                if motor_row['y'] != -1 and motor_row['x'] != -1: 
                    cx = float(motor_row['x']) # x가 가로, y가 세로
                    cy = float(motor_row['y'])
                    w = float(self.virtual_box_w)
                    h = float(self.virtual_box_h)
                    boxes_cxcywh_list.append([cx, cy, w, h])
                    labels_list.append(0) # 모터 클래스 ID = 0

        target = {}
        # target['spatial_size']는 BoundingBoxes 객체 생성 시 사용될 이미지의 실제 크기 (H,W)
        # 여기서는 최종 입력 슬라이스 크기
        target['spatial_size'] = torch.tensor([self.output_size_h, self.output_size_w], dtype=torch.long)
        
        boxes_tensor = torch.tensor(boxes_cxcywh_list, dtype=torch.float32) # (N, 4) 또는 (0, 4)

        # BoundingBoxes 객체로 변환
        # _boxes_keys[1]은 'spatial_size' 또는 'canvas_size' (torchvision 버전에 따라 다름)
        if boxes_tensor.numel() > 0:
            box_kwargs = {'format': BoundingBoxFormat.CXCYWH, _boxes_keys[1]: tuple(target['spatial_size'].tolist())}
            target['boxes'] = BoundingBoxes(boxes_tensor, **box_kwargs)
        else: # 빈 박스 처리
            box_kwargs = {'format': BoundingBoxFormat.CXCYWH, _boxes_keys[1]: tuple(target['spatial_size'].tolist())}
            target['boxes'] = BoundingBoxes(torch.empty((0, 4), device=img_tensor.device, dtype=torch.float32), **box_kwargs)
        
        if labels_list:
            target['labels'] = torch.tensor(labels_list, dtype=torch.long) # (N,)
        else:
            target['labels'] = torch.empty((0,), dtype=torch.long) # (0,)

        target['image_id'] = torch.tensor([idx]) # 배치 내 고유 ID (DataLoader에서 사용)
        target['tomo_id'] = tomo_id # 평가 시 사용
        target['slice_idx_in_tomo'] = torch.tensor([slice_idx], dtype=torch.long) # 평가 시 사용
        
        orig_shape_info = self.original_shapes.get(tomo_id, {'Z0_orig': self.output_size_h, 'H0_orig': self.output_size_h, 'W0_orig': self.output_size_w}) # 기본값은 리사이즈된 크기로
        target['orig_tomo_dims_zxy'] = torch.tensor([orig_shape_info['Z0_orig'], orig_shape_info['H0_orig'], orig_shape_info['W0_orig']], dtype=torch.long)
        target['voxel_spacing_angstroms'] = torch.tensor([self.voxel_spacing_map.get(tomo_id, 10.0)], dtype=torch.float32) # 기본값 10.0A

        img_tv_tensor = TVImage(img_tensor) # Image 객체로 변환

        return img_tv_tensor, target

    def __getitem__(self, idx):
        # DetDataset의 __getitem__ 로직을 따름
        img, target = self.load_item(idx)
        
        # self.transforms는 YAML에서 Compose 객체로 주입됨
        # Compose 객체는 tv_tensor를 입력으로 받도록 설계되어 있음
        if self.transforms is not None:
            # self.transforms의 입력은 (image, target, dataset_instance) 형태를 기대할 수 있음
            # 또는 (image, target)만 받을 수도 있음. DEIM의 Compose는 (image, target, dataset)을 받음.
            img, target, _ = self.transforms(img, target, self) 
        return img, target

    # set_epoch 메소드는 transforms.Compose.policy에서 사용될 수 있으므로 유지
    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else 0 # 기본값 0으로 변경