"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import math
from functools import partial
import torch # torch import 추가


def flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):
    """
    Computes the learning rate using a warm-up, flat, and cosine decay schedule.

    Args:
        total_iter (int): Total number of iterations.
        warmup_iter (int): Number of iterations for warm-up phase.
        flat_iter (int): Number of iterations for flat phase.
        no_aug_iter (int): Number of iterations for no-augmentation phase.
        current_iter (int): Current iteration.
        init_lr (float): Initial learning rate.
        min_lr (float): Minimum learning rate.

    Returns:
        float: Calculated learning rate.
    """
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / float(warmup_iter)) ** 2
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= total_iter - no_aug_iter:
        return min_lr
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (current_iter - flat_iter) /
                                           (total_iter - flat_iter - no_aug_iter)))
        return min_lr + (init_lr - min_lr) * cosine_decay


# @register() # 만약 DEIM 시스템에 이 스케줄러를 타입으로 직접 등록하려면 필요
class FlatCosineLRScheduler(object): # PyTorch의 LRScheduler를 상속하지 않음
    """
    Implements a learning rate schedule with warm-up, flat, and cosine decay phases.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        lr_gamma (float): Factor to compute minimum learning rate (min_lr = base_lr * lr_gamma).
        iter_per_epoch (int): Number of iterations per epoch.
        total_epochs (int): Total number of training epochs.
        warmup_iter (int): Number of iterations for warm-up phase.
        flat_epochs (int): Number of epochs for flat phase.
        no_aug_epochs (int): Number of epochs for no-augmentation phase (learning rate is min_lr).
    """
    def __init__(self, optimizer, lr_gamma, iter_per_epoch, total_epochs,
                 warmup_iter, flat_epochs, no_aug_epochs, scheduler_type="flatcosine"): # scheduler_type은 현재 미사용
        
        # optimizer.param_groups[0]['initial_lr']이 설정되어 있다고 가정.
        # 또는 base_lrs를 optimizer.param_groups에서 직접 가져옴.
        # DEIM의 BaseSolver는 optimizer 생성 시 param_groups에 'initial_lr'을 설정해줄 수 있음.
        # 만약 없다면, 현재 lr을 initial_lr로 사용.
        self.base_lrs = []
        for group in optimizer.param_groups:
            if 'initial_lr' not in group: # initial_lr이 없다면 현재 lr을 base로 사용
                group['initial_lr'] = group['lr']
            self.base_lrs.append(group['initial_lr'])

        self.min_lrs = [base_lr * lr_gamma for base_lr in self.base_lrs]
        self.optimizer = optimizer # optimizer 저장

        total_iter = int(iter_per_epoch * total_epochs)
        # warmup_iter는 이미 절대 반복 횟수
        flat_iter_abs = int(iter_per_epoch * flat_epochs) # flat_epochs를 반복 횟수로 변환
        no_aug_iter_abs = int(iter_per_epoch * no_aug_epochs) # no_aug_epochs를 반복 횟수로 변환

        # flat_cosine_schedule 함수에 전달될 flat_iter는 "웜업 이후부터 플랫 구간 끝까지의 총 반복 횟수"를 의미할 수 있음.
        # DEIM의 rt_deim.yml 주석: flat_epoch: 29 # 4 + epoch // 2, e.g., 40 = 4 + 72 / 2
        # 위 주석은 flat_epoch이 "웜업 에폭 + (전체 에폭 - 웜업 에폭 - no_aug 에폭) / 2" 를 의미할 수 있음을 시사.
        # 여기서는 YAML의 flat_epochs가 플랫 구간의 "길이" (에폭 단위)라고 가정.
        # flat_cosine_schedule 함수의 flat_iter는 "플랫 구간이 끝나는 시점의 반복 횟수"를 의미.
        # 따라서, flat_iter_for_schedule = warmup_iter + flat_iter_abs
        
        # no_aug_iter_for_schedule는 "no_aug_epochs 만큼의 반복 횟수"
        # total_iter는 전체 반복 횟수

        # print(self.base_lrs, self.min_lrs, total_iter, warmup_iter, flat_iter_for_schedule, no_aug_iter_abs)
        self.lr_func = partial(flat_cosine_schedule, total_iter, warmup_iter, warmup_iter + flat_iter_abs, no_aug_iter_abs)
        
        # get_last_lr()를 위해 현재 lr 저장할 변수 추가
        self.last_epoch = -1 # PyTorch LRScheduler와 유사하게
        self._last_lr = [group['lr'] for group in optimizer.param_groups]


    def step(self, current_iter, optimizer=None): # optimizer 인자 추가 (호출 시 전달됨)
        """
        Updates the learning rate of the optimizer at the current iteration.
        """
        if optimizer is None: # 호환성을 위해
            optimizer = self.optimizer

        for i, (param_group, base_lr, min_lr) in enumerate(zip(optimizer.param_groups, self.base_lrs, self.min_lrs)):
            new_lr = self.lr_func(current_iter=current_iter, init_lr=base_lr, min_lr=min_lr)
            param_group["lr"] = new_lr
        
        self._last_lr = [group['lr'] for group in optimizer.param_groups] # 현재 lr 업데이트

    def get_last_lr(self):
        """ Returns the last computed learning rate by this scheduler. """
        # self._last_lr이 __init__에서 현재 lr로 초기화되고, step()에서 업데이트되도록 보장
        return self._last_lr

    # PyTorch LRScheduler와의 호환성을 위한 state_dict 및 load_state_dict (필요시)
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'lr_func'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        # lr_func는 다시 빌드해야 할 수 있음 (또는 저장/로드 가능한 형태로 변경)
        # 여기서는 간단히 유지
