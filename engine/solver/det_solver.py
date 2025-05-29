# DEIM/engine/solver/det_solver.py
import time
import json
import datetime
from pathlib import Path
import torch

from tqdm import tqdm

# DEIM 프로젝트 내의 올바른 상대 경로로 수정
from ..misc import dist_utils, stats
from ..optim.lr_scheduler import FlatCosineLRScheduler # FlatCosineLRScheduler 직접 import
from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_metric = 0.0 # 최고 성능 추적용
        # self.optimizer는 self.cfg.optimizer를 통해 YAMLConfig가 생성
        # self.lr_scheduler는 self.cfg.lr_scheduler를 통해 YAMLConfig가 생성 (예: MultiStepLR)
        # self.lr_warmup_scheduler는 self.cfg.lr_warmup_scheduler를 통해 YAMLConfig가 생성

    def fit(self, ):
        self.train() # 모델을 train 모드로 설정
        args = self.cfg # YAMLConfig 객체

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("----" * 10 + " Start training " + "----" * 11)

        # --- 학습률 스케줄러 설정 ---
        iter_per_epoch = len(self.train_dataloader)
        if iter_per_epoch == 0:
            raise ValueError("Train Dataloader is empty or not properly initialized.")

        # YAMLConfig가 생성한 기본 lr_scheduler와 lr_warmup_scheduler를 일단 가져옴
        current_lr_scheduler = self.lr_scheduler
        current_lr_warmup_scheduler = self.lr_warmup_scheduler

        # rt_deim.yml의 'lrsheduler: flatcosine' 설정을 확인하여 FlatCosineLRScheduler 사용 여부 결정
        if hasattr(args, 'lrsheduler') and args.lrsheduler == 'flatcosine':
            print(f"     ## Configuring LR Scheduler as FlatCosineLRScheduler (from 'lrsheduler: flatcosine' in YAML) ## ")
            print(f"        iter_per_epoch: {iter_per_epoch}")

            required_params = ['lr_gamma', 'warmup_iter', 'flat_epoch', 'no_aug_epoch', 'epoches']
            missing_params = [p for p in required_params if not hasattr(args, p)]
            if missing_params:
                raise AttributeError(f"Missing FlatCosineLRScheduler parameters in YAML config: {', '.join(missing_params)}")

            print(f"        Using FlatCosine params from YAML: lr_gamma={args.lr_gamma}, warmup_iter={args.warmup_iter}, "
                  f"flat_epochs={args.flat_epoch}, no_aug_epochs={args.no_aug_epoch}, total_epochs={args.epoches}")

            current_lr_scheduler = FlatCosineLRScheduler(
                optimizer=self.optimizer,
                lr_gamma=args.lr_gamma,
                iter_per_epoch=iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter, # FlatCosineLRScheduler 자체 웜업 사용
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch
            )
            print(f"        Successfully initialized FlatCosineLRScheduler. Initial LR: {current_lr_scheduler.get_last_lr()}")
            
            # FlatCosineLRScheduler가 자체 웜업을 하므로, 외부 LinearWarmup은 비활성화
            if args.warmup_iter > 0 and current_lr_warmup_scheduler is not None:
                print(f"        Note: FlatCosineLRScheduler (warmup_iter={args.warmup_iter}) will handle warmup. "
                      f"Disabling external lr_warmup_scheduler ({type(current_lr_warmup_scheduler).__name__}).")
                current_lr_warmup_scheduler = None # 외부 웜업 스케줄러 사용 안 함
        
        elif current_lr_scheduler is not None:
            print(f"     ## Using LR Scheduler from YAMLConfig (e.g., rt_optimizer.yml): {type(current_lr_scheduler).__name__} ##")
        else:
            print("     ## No LR Scheduler configured. ##")


        start_time = time.time()
        start_epoch = self.last_epoch + 1
        print(f"Starting training from epoch {start_epoch} to {args.epoches}")

        # [추가] epoch 루프에 TQDM을 적용하면 "Epoch 1/60..." 형태의 진행 바를 표시 가능
        # leave=False 로 하면 완전히 끝난 후 진행바가 사라짐
        for epoch in tqdm(range(start_epoch, args.epoches), desc="Training Epochs", leave=True):
            # 분산 학습 시 epoch 설정
            if dist_utils.is_dist_available_and_initialized():
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

            # Dataset, CollateFn 등에서도 epoch 사용
            if hasattr(self.train_dataloader.dataset, 'set_epoch'):
                self.train_dataloader.dataset.set_epoch(epoch)
            if hasattr(self.train_dataloader.collate_fn, 'set_epoch'):
                self.train_dataloader.collate_fn.set_epoch(epoch)

            # ------------ 실제 한 에폭 학습 ------------
            train_stats = train_one_epoch(
                model=self.model,
                criterion=self.criterion,
                data_loader=self.train_dataloader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                max_norm=getattr(args, 'clip_max_norm', 0.1),
                lr_scheduler_to_use=current_lr_scheduler,
                lr_warmup_scheduler_to_use=current_lr_warmup_scheduler,
                iter_per_epoch=iter_per_epoch,
                print_freq=args.print_freq,
                writer=self.writer,  # TensorBoard writer
                scaler=self.scaler,
                ema=self.ema
            )
            # ------------------------------------------
            
            # (Optional) 에폭 단위의 스케줄러 스텝
            if current_lr_scheduler is not None and not isinstance(current_lr_scheduler, FlatCosineLRScheduler):
                # 외부 웜업이 끝났는지 확인
                if current_lr_warmup_scheduler is None or \
                   (hasattr(current_lr_warmup_scheduler, 'finished') and current_lr_warmup_scheduler.finished()):
                    current_lr_scheduler.step()
                else:
                    print(f"Epoch {epoch}: External warmup in progress or finished() not available, skipping main LR scheduler step.")

            # (Optional) 주기적으로 검증 실행
            if (epoch + 1) % args.val_freq == 0 or epoch == args.epoches - 1:
                module_to_eval = (self.ema.module
                                  if self.ema and self.ema.device == self.device and epoch >= getattr(args.ema, 'start', 0)
                                  else self.model)
                test_stats_dict, _ = evaluate(
                    module_to_eval,
                    self.criterion if getattr(args, 'val_loss', False) else None,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                    print_freq=args.print_freq
                )
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'val_{k}': v for k, v in test_stats_dict.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }

                # Best metric (예: F_beta) 업데이트
                current_f_beta = test_stats_dict.get("F_beta", 0.0)
                if current_f_beta > self.best_metric:
                    self.best_metric = current_f_beta
                    checkpoint_best_path = Path(self.output_dir) / 'model_best.pth'
                    dist_utils.save_on_master(self.state_dict(epoch, metric=current_f_beta), str(checkpoint_best_path))
                    print(f"Saved best model with F_beta: {current_f_beta:.4f} at epoch {epoch}")

            else:
                # 검증 스킵하는 경우
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }

            # 체크포인트 저장
            if self.output_dir and (epoch + 1) % args.checkpoint_freq == 0:
                checkpoint_path = Path(self.output_dir) / f'checkpoint{epoch:04}.pth'
                dist_utils.save_on_master(self.state_dict(epoch), str(checkpoint_path))

            # 로그 파일 저장
            if self.output_dir and dist_utils.is_main_process():
                with (Path(self.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            # [추가] TensorBoard에 epoch 단위 기록
            if self.writer and dist_utils.is_main_process():
                # train stats
                for k_log, v_log in train_stats.items():
                    self.writer.add_scalar(f'train/{k_log}', v_log, epoch)
                # val stats
                if 'test_stats_dict' in locals() and ((epoch + 1) % args.val_freq == 0 or epoch == args.epoches - 1):
                    for k_log, v_log in test_stats_dict.items():
                        self.writer.add_scalar(f'val/{k_log}', v_log, epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()
        args = self.cfg
        module_to_eval = self.ema.module if self.ema and self.ema.device == self.device else self.model
        test_stats, _ = evaluate(
            module_to_eval,
            self.criterion if getattr(args, 'val_loss', False) else None,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            print_freq=args.print_freq
        )
        print("Validation results:", test_stats)

    def state_dict(self, epoch=-1, metric=None):
        state = super().state_dict(epoch)
        if metric is not None:
            state['best_metric'] = metric
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if 'best_metric' in state:
            self.best_metric = state['best_metric']