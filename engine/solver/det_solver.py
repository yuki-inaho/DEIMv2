"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 D-FINE authors. All Rights Reserved.
"""

import time
import json
import datetime

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from beartype import beartype

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


class DetSolver(BaseSolver):

    # Persistent sidecar file that records the current top-K best checkpoints.
    # Lets `resume` keep pruning/ordering correct without re-evaluating history.
    _TOPK_STATE_FILE = 'topk_checkpoints.json'

    def __init__(self, cfg):
        super().__init__(cfg)
        yaml_cfg: Dict[str, object] = getattr(cfg, 'yaml_cfg', {}) or {}

        # K-best checkpoint saving (0 = disabled, keeps the historical single-best
        # best_stg1/best_stg2 behaviour).
        self.topk_checkpoints = int(yaml_cfg.get('topk_checkpoints', 0))
        # Validate the monitor metric once per run, not per epoch.
        self.topk_monitor_key: str = str(
            yaml_cfg.get('topk_monitor_key', 'coco_eval_bbox')
        )

        # Validation cadence: evaluate every N epochs (1 = current per-epoch
        # behaviour). Stage-transition epoch (collate stop_epoch) and the final
        # epoch always run validation for correctness.
        self.eval_every_n_epoch = max(1, int(yaml_cfg.get('eval_every_n_epoch', 1)))

        # (metric, epoch) pairs ordered best-first; restored on resume from sidecar
        # once output_dir is known (in fit(), after _setup()).
        self._topk: List[Tuple[float, int]] = []

    def _set_optimizer_mode(self, mode: str) -> None:
        optimizer = getattr(self, 'optimizer', None)
        mode_fn = getattr(optimizer, mode, None)
        if callable(mode_fn):
            mode_fn()

    def _save_state(self, checkpoint_path) -> None:
        self._set_optimizer_mode('eval')
        dist_utils.save_on_master(self.state_dict(), checkpoint_path)
        self._set_optimizer_mode('train')

    # ------------------------------------------------------------------
    # Top-K best checkpoint management (config-driven, opt-in)
    # ------------------------------------------------------------------
    @beartype
    def _topk_sidecar_path(self) -> Optional[Path]:
        if not getattr(self, 'output_dir', None):
            return None
        return self.output_dir / self._TOPK_STATE_FILE

    @beartype
    def _load_topk_state(self) -> None:
        if not dist_utils.is_main_process():
            return
        sidecar = self._topk_sidecar_path()
        if sidecar is not None and sidecar.exists():
            try:
                data = json.loads(sidecar.read_text())
                self._topk = [
                    (float(entry['metric']), int(entry['epoch']))
                    for entry in data.get('topk', [])
                ]
                self._topk.sort(key=lambda x: x[0], reverse=True)
                self._topk = self._topk[: self.topk_checkpoints]
                # Prune files that are no longer in the (restored) top-K.
                kept_epochs = {e for _, e in self._topk}
                for f in self.output_dir.glob('best_ep*.pth'):
                    if int(f.stem.split('_ep')[-1]) not in kept_epochs:
                        f.unlink(missing_ok=True)
                print(
                    f'[TopK] restored {len(self._topk)} best checkpoint(s) '
                    f'from {sidecar}'
                )
            except Exception as exc:  # noqa: BLE001 - sidecar is best-effort
                print(f'[TopK] WARNING: could not restore sidecar {sidecar}: {exc}')

    @beartype
    def _save_topk_state(self) -> None:
        if not dist_utils.is_main_process():
            return
        sidecar = self._topk_sidecar_path()
        if sidecar is None:
            return
        sidecar.write_text(
            json.dumps(
                {'topk': [{'metric': m, 'epoch': e} for m, e in self._topk]},
                indent=2,
            )
        )

    @beartype
    def _maybe_save_topk(self, metric: float, epoch: int) -> None:
        """Keep the K best checkpoints by `metric`, pruning worse ones."""
        if self.topk_checkpoints <= 0 or not self.output_dir:
            return
        if not dist_utils.is_main_process():
            return

        self._topk.append((metric, epoch))
        self._topk.sort(key=lambda x: x[0], reverse=True)
        self._topk = self._topk[: self.topk_checkpoints]

        kept_epochs = {e for _, e in self._topk}
        for f in self.output_dir.glob('best_ep*.pth'):
            ep = int(f.stem.split('_ep')[-1])
            if ep not in kept_epochs:
                f.unlink(missing_ok=True)

        if epoch in kept_epochs:
            self._save_state(self.output_dir / f'best_ep{epoch:04}.pth')

        self._save_topk_state()
        print(
            '[TopK] best checkpoints: '
            + ', '.join(f'ep{e}: {m:.4f}' for m, e in self._topk)
        )

    @beartype
    def _topk_primary_metric(self, test_stats: Dict[str, List[float]]) -> float:
        """Return the scalar used for top-K ranking."""
        if self.topk_monitor_key in test_stats:
            return float(test_stats[self.topk_monitor_key][0])
        for v in test_stats.values():
            return float(v[0])
        return float('-inf')

    def fit(self, ):
        self.train()
        # Restore the persisted top-K snapshot now that output_dir is known.
        if self.topk_checkpoints > 0:
            self._load_topk_state()
        self._set_optimizer_mode('train')
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i in [194, 195]:
                print(f"Index {i}: {name} - requires_grad: {param.requires_grad}")

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches, 
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if not p.requires_grad])
        print(f'number of non-trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            self._set_optimizer_mode('eval')
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            self._set_optimizer_mode('train')
            for k in test_stats:
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f'best_stat: {best_stat}')

        best_stat_print = best_stat.copy()
        # Config-driven early stopping; monitor validation mAP after the
        # configured stage transition rather than during warm-up/augmentation.
        early_stop_cfg = getattr(self.cfg, 'yaml_cfg', {}) or {}
        early_stop_enabled = bool(early_stop_cfg.get('early_stop', False))
        early_stop_patience = int(early_stop_cfg.get('early_stop_patience', 10))
        early_stop_min_delta = float(early_stop_cfg.get('early_stop_min_delta', 0.0))
        early_stop_start_epoch = int(early_stop_cfg.get('early_stop_start_epoch', 0))
        early_stop_wait = 0
        early_stop_best = top1
        if early_stop_enabled:
            print(
                f'## EarlyStopping ON: patience={early_stop_patience}, '
                f'min_delta={early_stop_min_delta}, start_epoch={early_stop_start_epoch} '
                f'(metric = val mAP) ##'
            )
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):
            self._set_optimizer_mode('train')

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self._set_optimizer_mode('train')
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_scheduler is not None and (self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished()):
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    self._save_state(checkpoint_path)

            # Validation cadence (config-driven via eval_every_n_epoch).
            # Stage-transition epoch and the final epoch always validate so the
            # stg1/stg2 semantics and a final metric are preserved.
            do_eval = (
                (epoch - start_epoch) % self.eval_every_n_epoch == 0
                or epoch == self.train_dataloader.collate_fn.stop_epoch
                or epoch == args.epoches - 1
            )
            test_stats = {}
            coco_evaluator = None
            if do_eval:
                module = self.ema.module if self.ema else self.model
                self._set_optimizer_mode('eval')
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device
                )
                self._set_optimizer_mode('train')

                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]

                    if best_stat[k] > top1:
                        best_stat_print['epoch'] = epoch
                        top1 = best_stat[k]
                        if self.output_dir:
                            if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                                self._save_state(self.output_dir / 'best_stg2.pth')
                            else:
                                self._save_state(self.output_dir / 'best_stg1.pth')

                    best_stat_print[k] = max(best_stat[k], top1)
                    print(f'best_stat: {best_stat_print}')  # global best

                    if best_stat['epoch'] == epoch and self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            if test_stats[k][0] > top1:
                                top1 = test_stats[k][0]
                                self._save_state(self.output_dir / 'best_stg2.pth')
                        else:
                            top1 = max(test_stats[k][0], top1)
                            self._save_state(self.output_dir / 'best_stg1.pth')

                    elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        best_stat = {'epoch': -1, }
                        self.ema.decay -= 0.0001
                        self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                        self._set_optimizer_mode('train')
                        print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

                # K-best checkpoint saving (config-driven, opt-in)
                self._maybe_save_topk(
                    self._topk_primary_metric(test_stats), epoch
                )


            early_stop_should_stop = False
            if early_stop_enabled and epoch >= early_stop_start_epoch:
                if top1 > early_stop_best + early_stop_min_delta:
                    early_stop_best = top1
                    early_stop_wait = 0
                else:
                    early_stop_wait += 1
                    print(
                        f'[EarlyStop] no-improve {early_stop_wait}/{early_stop_patience} '
                        f'(best mAP={early_stop_best:.4f}, cur best={top1:.4f})'
                    )
                    early_stop_should_stop = early_stop_wait >= early_stop_patience

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

            if early_stop_should_stop:
                print(
                    f'[EarlyStop] stopping at epoch {epoch}: no val(mAP) improvement '
                    f'for {early_stop_patience} epochs (best mAP={early_stop_best:.4f})'
                )
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        self._set_optimizer_mode('eval')
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
