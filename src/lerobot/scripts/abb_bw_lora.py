#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from peft import get_peft_model, LoraConfig

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_eval import eval_policy_all
from pathlib import Path
import copy
import warnings
warnings.filterwarnings("ignore")
import datetime
from tqdm import tqdm 


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def get_trainable_parameters(model):
    trainable_params = []
    for name, module in model.named_modules():
        if any(p.requires_grad for p in module.parameters()):
            trainable_params.append(name.split(".")[-1])
    return list(set(trainable_params))

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9855'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

@parser.wrap()
def train(cfg: TrainPipelineConfig, rank=0, world_size=1):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(f"[Rank {rank}] Start of training")
    setup_ddp(rank, world_size)
    print(f"[Rank {rank}] DDP setup complete.")
    cfg.eval_freq = 100
    cfg.log_freq = 25
    # cfg.batch_size = 16
    # cfg.steps=4000
    # 设置数据集路径
    # cfg.dataset.root = '/home/bwli/storage/pi_0/datasets/' + cfg.dataset.repo_id
    # cfg.dataset.root = '/home/szxie/bw_pi0/make_dataset/baked/' + cfg.dataset.repo_id
    cfg.validate()
    if rank == 0:
        logging.info(pformat(cfg.to_dict()))

    # 创建实验文件夹
    if rank == 0:
        experiment_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path("save") / experiment_time
        experiment_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Experiment directory created: {experiment_dir}")
        
        # 创建CSV文件路径
        train_metrics_path = experiment_dir / "train_loss.csv"
        eval_metrics_path = experiment_dir / "eval_loss.csv"
        checkpoint_path = experiment_dir / "best_checkpoint.pth"
        
        # 初始化CSV文件
        with open(train_metrics_path, "w") as f:
            f.write("step,epoch,loss\n")
        with open(eval_metrics_path, "w") as f:
            f.write("step,epoch,train_loss,eval_loss\n")
        
        # 初始化最佳评估损失
        best_eval_loss = float('inf')
        logging.info(f"Initial best eval loss: {best_eval_loss}")
    else:
        experiment_dir = None
        train_metrics_path = None
        eval_metrics_path = None
        checkpoint_path = None
        best_eval_loss = None

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # 检测是否需要创建评估数据集
    logging.info("Checking for evaluation dataset")
    eval_dataset = None
    if cfg.dataset.repo_id.endswith("_train"):
        # 创建评估数据集配置
        eval_repo_id = cfg.dataset.repo_id.replace("_train", "_eval")
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.dataset.repo_id = eval_repo_id
        eval_cfg.dataset.root = '/home/bwli/storage/pi_0/datasets/' + eval_repo_id
        
        try:
            eval_dataset = make_dataset(eval_cfg)
            logging.info(f"Evaluation dataset created with {eval_dataset.num_frames} frames from {eval_dataset.num_episodes} episodes")
        except Exception as e:
            logging.warning(f"Failed to create evaluation dataset for {eval_repo_id}: {e}")
            eval_dataset = None
    else:
        logging.info("Training dataset does not end with '_train', skipping evaluation dataset creation")

    logging.info("Creating policy")
    cfg.policy.device = f"cuda:{rank}"
    print(f"[Rank {rank}] Creating policy.", 'device:', cfg.policy.device)
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )
    
    target_modules = []
    for name, _ in policy.model.named_parameters():
        # 检查参数名称是否不包含paligemma且包含指定的子串
        if ("paligemma_with_expert.paligemma" not in name and
            any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head", "linear"])):
            # 去掉参数名称最后的".weight"或".bias"后缀
            module_name = name.rsplit(".", 1)[0]
            target_modules.append(module_name)
    
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )
    policy.model = get_peft_model(policy.model, lora_config)

    for name, param in policy.model.named_parameters():
        if (("paligemma_with_expert.paligemma" not in name) and
            not any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj", "lm_head", "linear"])):
            param.requires_grad = True
    '''
    if rank == 0:
        print("=" * 80)
        total_size = 0
        print(f"{'参数名称':<40} {'形状':<15} {'大小(MB)':<10} {'可训练':<8}")
        print("=" * 80)
        
        for name, param in policy.model.named_parameters():
            param_size = param.nelement() * param.element_size() / (1024 * 1024)  # 计算MB
            total_size += param_size
            trainable = param.requires_grad
            print(f"{name:<40} {str(tuple(param.shape)):<15} {param_size:.4f} MB    {str(trainable):<8}")
        
        print("=" * 80)
        print(f"总参数大小: {total_size:.4f} MB")
    '''

    policy.model.to(device)
    policy.model = DDP(policy.model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    print(f"[Rank {rank}] Policy model wrapped with DDP.")

    print("Creating optimizer and scheduler, rank:", rank)

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats
        
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )
    
    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    # 检查是否有现有的最佳checkpoint
    if rank == 0 and checkpoint_path.exists():
        logging.info(f"Rank {rank}: Loading best checkpoint from {checkpoint_path}")
        print(f"[Rank {rank}] Best checkpoint file found at {checkpoint_path}. Loading...")
        # 加载整个检查点文件
        checkpoint = torch.load(checkpoint_path, map_location={f'cuda:0': f'cuda:{rank}'}, weights_only=False)
        # 加载模型的可训练参数，使用 strict=False 允许不完全匹配，忽略检查点中不存在的参数
        policy.model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # 加载优化器和学习率调度器的状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        step = checkpoint.get('step', 0)
        best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        print(f"[Rank {rank}] Successfully loaded best checkpoint. Starting step: {step}, Best eval loss: {best_eval_loss}")
    else:
        # 如果指定了 resume 但文件不存在，警告并从头开始
        if rank == 0:
            logging.warning(f"Rank {rank}: Best checkpoint not found at {checkpoint_path}. Starting from scratch.")
            print(f"[Rank {rank}] Best checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        dist.barrier() # 等待所有进程加载完毕
    

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    # 使用当前目录作为输出目录的基础，尽管checkpoint直接保存在文件，output_dir可能用于其他输出
    output_base_dir = Path(".")

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_base_dir.resolve()}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    # 分布式采样器
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # 创建评估数据集的dataloader
    eval_dataloader = None
    if eval_dataset is not None:
        eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=eval_sampler,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        logging.info(f"Evaluation dataloader created with {len(eval_dataloader)} batches")

    dl_iter = cycle(train_dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in tqdm(range(step, cfg.steps)):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step and rank == 0:
            # logging.info(train_tracker)
            print(f"[Rank {rank}] Logging metrics at step {step}.")
            # 打印 train_tracker 的详细指标信息
            metrics_dict = train_tracker.to_dict()
            logging.info(f"[Rank {rank}] Metrics: {metrics_dict}")

            # 保存训练指标到CSV文件
            try:
                with open(train_metrics_path, "a") as f:
                    f.write(f"{step},{metrics_dict['epochs']},{metrics_dict['loss']}\n")
            except Exception as e:
                logging.error(f"Error writing to train_loss.csv: {e}")

            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        dist.barrier() # 确保所有进程在checkpoint操作前后同步
        if is_eval_step and rank == 0 and eval_dataloader is not None:
            policy.eval()
            
            # 获取当前epoch信息
            current_metrics_dict = train_tracker.to_dict()
            
            # 修复：使用train_tracker中的当前训练损失，而不是重新计算
            # 这样可以确保训练损失和评估损失基于相同的模型状态
            current_train_loss = current_metrics_dict.get('loss', 0.0)
            
            # 计算评估集损失
            eval_loss = 0
            eval_count = 0
            with torch.no_grad():
                for batch in eval_dataloader:
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device, non_blocking=True)
                    loss, _ = policy.forward(batch)
                    eval_loss += loss.item()
                    eval_count += 1
            avg_eval_loss = eval_loss / eval_count if eval_count > 0 else 0

            logging.info(f"Step {step} - Train Loss: {current_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

            # 保存评估指标到CSV
            try:
                with open(eval_metrics_path, "a") as f:
                    f.write(f"{step},{current_metrics_dict['epochs']},{current_train_loss},{avg_eval_loss}\n")
            except Exception as e:
                logging.error(f"Error writing to eval_loss.csv: {e}")

            # 检查是否需要保存最佳checkpoint
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                logging.info(f"New best eval loss: {best_eval_loss:.4f}. Saving checkpoint...")
                print(f"[Rank {rank}] Saving best checkpoint at step {step} to {checkpoint_path}.")
                
                # 保存最佳checkpoint
                trainable_state_dict = {name: param for name, param in policy.model.module.named_parameters() if param.requires_grad}
                torch.save({
                    'step': step,
                    'model_state_dict': trainable_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'config': cfg.to_dict(),
                    'best_eval_loss': best_eval_loss
                }, checkpoint_path)
                
                logging.info(f"Best checkpoint saved with eval loss: {best_eval_loss:.4f}")
            else:
                logging.info(f"Eval loss {avg_eval_loss:.4f} not better than best {best_eval_loss:.4f}. Skipping checkpoint save.")
            
            policy.train()
        elif is_eval_step and rank == 0 and eval_dataloader is None:
            # 如果没有评估数据集，每个eval_freq都保存checkpoint（覆盖之前的）
            logging.info(f"Step {step} - No evaluation dataset, saving checkpoint...")
            print(f"[Rank {rank}] Saving checkpoint at step {step} to {checkpoint_path}.")
            
            # 保存checkpoint
            trainable_state_dict = {name: param for name, param in policy.model.module.named_parameters() if param.requires_grad}
            torch.save({
                'step': step,
                'model_state_dict': trainable_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'config': cfg.to_dict(),
                'best_eval_loss': best_eval_loss
            }, checkpoint_path)
            
            logging.info(f"Checkpoint saved at step {step}")

    # 训练完成后，保存最终checkpoint到指定路径
    if rank == 0:
        logging.info("Training completed. Saving final checkpoint with all trainable parameters.")
        print(f"[Rank {rank}] Training completed. Saving final checkpoint...")
        
        # 保存最终checkpoint到实验文件夹
        final_checkpoint_path = checkpoint_path
        
        # 获取所有可训练参数的状态字典
        trainable_state_dict = {name: param for name, param in policy.model.module.named_parameters() if param.requires_grad}
        
        # 保存最终checkpoint
        torch.save({
            'step': step,
            'model_state_dict': trainable_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'config': cfg.to_dict(),
            'final_training_metrics': train_tracker.to_dict(),
            'num_learnable_params': num_learnable_params,
            'num_total_params': num_total_params,
            'best_eval_loss': best_eval_loss
        }, final_checkpoint_path)
        
        logging.info(f"Final checkpoint saved to: {final_checkpoint_path}")
        print(f"[Rank {rank}] Final checkpoint saved to: {final_checkpoint_path}")
        
        # 等待checkpoint保存完成
        dist.barrier()
    
    cleanup_ddp()
    print(f"[Rank {rank}] DDP cleanup complete.")
    if rank == 0:
        logging.info("End of training")
        print(f"[Rank {rank}] End of training process.")

if __name__ == "__main__":
    init_logging()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    world_size = torch.cuda.device_count()
    print(f"[Main Process] Detected {world_size} visible GPUs.")
    # 直接用mp.spawn分布式并行运行train函数
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    print(f"[Main Process] 实验全部完成.")
