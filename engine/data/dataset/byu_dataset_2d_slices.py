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
import random
from collections import defaultdict


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
                 virtual_box_wh: tuple = (20, 20),
                 # 추가: 최종 이미지 크기 (YAML에서 설정하거나, 고정값 사용)
                 output_size_h: int = 512,
                 output_size_w: int = 512,
                 sample_nearby: int = 1,      # 중심 슬라이스 주변 ±1
                 neg_per_pos_tomo: int = 3,   # 양성 토모그램당 배경 슬라이스 개수
                 neg_per_neg_tomo: int = 3    # 완전 음성 토모그램(모터 없는)에 대해 샘플링할 슬라이스 개수
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


        # (A) tomo_id -> path dict
        tid2path = {tid: p for tid, p in zip(tomo_ids_all, tomo_file_paths_all)}

        # (B) 라벨 있는 슬라이스 z 수집
        pos_slices_per_tomo = defaultdict(set)
        for idx, row in self.df_labels_src.iterrows():
            tid = row['tomo_id']
            z_rounded = int(round(row['z']))
            if z_rounded >= 0 and z_rounded < 128:
                pos_slices_per_tomo[tid].add(z_rounded)

        # 양성 tomo / 음성 tomo 구분
        pos_tomos = []
        neg_tomos = []
        for tid in active_tomo_ids_set:
            if len(pos_slices_per_tomo[tid]) > 0:
                pos_tomos.append(tid)
            else:
                neg_tomos.append(tid)

        # ★★ 학습 세트이고, 샘플링을 적용하려면:
        if self.is_train:
            final_items = []

            def clamp_z(z):
                return max(0, min(127, z))

            # 1) 양성 tomo: 라벨 ± sample_nearby + 음성 N개
            for tid in pos_tomos:
                if tid not in tid2path:
                    continue
                path = tid2path[tid]
                pos_zlist = sorted(list(pos_slices_per_tomo[tid]))

                # (a) “±sample_nearby” 부분
                pos_nearby_z = set()
                for zc in pos_zlist:
                    for dz in range(-sample_nearby, sample_nearby + 1):
                        pos_nearby_z.add(clamp_z(zc + dz))

                # (b) 해당 tomo에서 남은 z(=배경) 중 일부
                all_z = set(range(128))
                background_z = list(all_z - pos_nearby_z)
                if len(background_z) > neg_per_pos_tomo:
                    background_z = random.sample(background_z, neg_per_pos_tomo)

                # 모으기
                used_zs = sorted(list(pos_nearby_z)) + sorted(background_z)
                for zval in used_zs:
                    final_items.append((path, zval, tid))

            # 2) 완전 음성 tomo
            for tid in neg_tomos:
                if tid not in tid2path:
                    continue
                path = tid2path[tid]
                # 128개 중 neg_per_neg_tomo개
                all_z = list(range(128))
                if len(all_z) > neg_per_neg_tomo:
                    sampled_z = random.sample(all_z, neg_per_neg_tomo)
                else:
                    sampled_z = all_z
                for zval in sampled_z:
                    final_items.append((path, zval, tid))

            self.slice_items = final_items
            print(f"[Train sampling] from ~{len(active_tomo_ids_set)*128} slices => {len(self.slice_items)} slices.")
        else:
            # 검증 세트는 전부 사용 (원하면 동일 샘플링 적용 가능)
            final_items = []
            for tid in active_tomo_ids_set:
                if tid not in tid2path:
                    continue
                path = tid2path[tid]
                for z in range(128):
                    final_items.append((path, z, tid))
            self.slice_items = final_items
            print(f"[Val set - use ALL] {len(self.slice_items)} slices from {len(active_tomo_ids_set)} tomos")

    
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
        areas_list = [] # <<< 면적을 저장할 리스트 추가
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
                    areas_list.append(w * h) # <<< 박스 면적 추가

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

        # --- 'area' 키 추가 ---
        if areas_list:
            target['area'] = torch.tensor(areas_list, dtype=torch.float32)
        else:
            target['area'] = torch.empty((0,), dtype=torch.float32) # 박스가 없으면 빈 area 텐서
        # --- 'area' 키 추가 끝 ---

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

        # if target['boxes'].shape[0]>0:
        #     print(f"[DEBUG] AFTER transforms => #boxes={target['boxes'].shape[0]}")


        return img, target

    # set_epoch 메소드는 transforms.Compose.policy에서 사용될 수 있으므로 유지
    def set_epoch(self, epoch) -> None:
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else 0 # 기본값 0으로 변경