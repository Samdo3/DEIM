"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re
import copy

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict

class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self, ):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self, ) -> torch.nn.Module:
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model

    @property
    def postprocessor(self, ) -> torch.nn.Module:
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self, ) -> torch.nn.Module:
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self, ) -> optim.Optimizer:
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            self._optimizer = create('optimizer', self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
            print(f'Initial lr: {self._lr_scheduler.get_last_lr()}')
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg :
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader

    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._scaler = create('scaler', self.global_cfg)
        return super().scaler

    @property
    def evaluator(self, ):
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            evaluator_type = self.yaml_cfg['evaluator'].get('type', None) # get()으로 안전하게 접근
            if evaluator_type == 'CocoEvaluator':
                # CocoEvaluator에 대한 기존 특별 처리 로직
                from ..data import get_coco_api_from_dataset # data 모듈에서 함수 import
                # val_dataloader가 먼저 빌드되도록 보장
                if self._val_dataloader is None:
                    _ = self.val_dataloader # self.val_dataloader 프로퍼티 호출하여 빌드
                
                if self._val_dataloader is not None and hasattr(self._val_dataloader, 'dataset'):
                    base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                    # create 함수는 global_cfg['evaluator'] 딕셔너리 내용을 사용.
                    # 여기에 coco_gt 인자를 추가하여 전달.
                    evaluator_creation_cfg = copy.deepcopy(self.global_cfg['evaluator'])
                    evaluator_creation_cfg['coco_gt'] = base_ds
                    self._evaluator = create('evaluator', evaluator_creation_cfg) # 수정된 cfg로 생성
                else:
                    raise ValueError("Validation Dataloader or its dataset is not available for CocoEvaluator.")
            
            elif evaluator_type == 'BYU2DEvaluator': # 우리가 추가한 커스텀 Evaluator 타입
                # BYU2DEvaluator는 특별한 인자(예: coco_gt) 없이 YAML에 정의된 파라미터로 생성
                # create 함수는 global_cfg['evaluator'] 딕셔너리 내용을 사용
                print(f"     ## Creating Evaluator of type: {evaluator_type} ##")
                self._evaluator = create('evaluator', self.global_cfg) # global_cfg['evaluator'] 사용
            
            elif evaluator_type is not None: # 다른 타입의 Evaluator (일반적인 생성 방식)
                print(f"     ## Creating Evaluator of type: {evaluator_type} (using default create) ##")
                self._evaluator = create('evaluator', self.global_cfg)

            else: # evaluator type이 지정되지 않은 경우
                raise ValueError("Evaluator 'type' is not specified in YAML config.")
                


        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert 'type' in cfg, ''
        cfg = copy.deepcopy(cfg)

        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), ''

        param_groups = []
        visited = []
        for pg in cfg['params']:
            pattern = pg['params']
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ''

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided.
        """
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
            and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
                '`batch_size` or `total_batch_size` should be choosed one'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            from ..misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size should be divisible by world size'
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if 'total_batch_size' in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop('total_batch_size')
        print(f'building {name} with batch_size={bs}...')
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)
        return loader
