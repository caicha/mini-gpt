"""
训练器模块：负责训练循环、验证、模型保存
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from config.config import ModelConfig, TrainingConfig, PathConfig
from src.model import MiniGPT
from src.dataset import CharDataset, get_batch


class Trainer:
    """训练器"""

    def __init__(
            self,
            model: MiniGPT,
            dataset: CharDataset,
            model_config: ModelConfig,
            training_config: TrainingConfig,
            path_config: PathConfig,
            device: torch.device
    ):
        self.model = model
        self.dataset = dataset
        self.model_config = model_config
        self.training_config = training_config
        self.path_config = path_config
        self.device = device

        # 优化器
        self.optimizer = self._create_optimizer()

        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=path_config.log_dir)
        self.step = 0

        # 最佳验证损失
        self.best_val_loss = float('inf')

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器（AdamW + 权重衰减）"""
        cfg = self.training_config

        # 分离需要权重衰减的参数
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'ln' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': cfg.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        return torch.optim.AdamW(param_groups, lr=cfg.learning_rate)

    def get_lr(self) -> float:
        """获取当前学习率（带预热）"""
        cfg = self.training_config
        if self.step < cfg.warmup_iters:
            # 线性预热
            return cfg.learning_rate * (self.step + 1) / cfg.warmup_iters
        else:
            # 余弦退火
            progress = (self.step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
            return cfg.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))

    def train_step(self) -> float:
        """执行一步训练"""
        cfg = self.training_config

        # 更新学习率
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # 采样batch
        x, y = get_batch(
            self.dataset,
            cfg.batch_size,
            self.model_config.block_size,
            self.device
        )

        # 前向传播
        logits, loss = self.model(x, y)

        # 反向传播
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """在验证集上评估模型"""
        cfg = self.training_config
        self.model.eval()

        total_loss = 0.0
        for _ in range(cfg.eval_iters):
            x, y = get_batch(
                self.dataset,
                cfg.batch_size,
                self.model_config.block_size,
                self.device
            )
            _, loss = self.model(x, y)
            total_loss += loss.item()

        self.model.train()
        return {'val_loss': total_loss / cfg.eval_iters}

    def save_checkpoint(self, is_best: bool = False):
        """保存模型检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'model_config': self.model_config,
            'vocab_size': self.dataset.vocab_size,
            'stoi': self.dataset.stoi,
            'itos': self.dataset.itos,
        }

        # 保存最新检查点
        latest_path = Path(self.path_config.checkpoint_dir) / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = Path(self.path_config.checkpoint_dir) / 'best.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载模型检查点"""
        if not Path(checkpoint_path).exists():
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        return True

    def train(self):
        """完整训练循环"""
        cfg = self.training_config

        print(f"开始训练...")
        print(f"设备: {self.device}")
        print(f"模型参数量: {self.model.get_num_params():,}")
        print(f"词典大小: {self.dataset.vocab_size}")
        print(f"训练步数: {cfg.max_iters}")

        # 训练循环
        pbar = tqdm(range(self.step, cfg.max_iters))
        for step in pbar:
            self.step = step

            # 训练一步
            loss = self.train_step()

            # 记录训练损失
            self.writer.add_scalar('train/loss', loss, step)
            self.writer.add_scalar('train/lr', self.get_lr(), step)

            # 更新进度条
            pbar.set_description(f"Step {step}, Loss: {loss:.4f}")

            # 定期评估
            if step % cfg.eval_interval == 0:
                metrics = self.evaluate()
                self.writer.add_scalar('val/loss', metrics['val_loss'], step)

                # 保存最佳模型
                if metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['val_loss']
                    self.save_checkpoint(is_best=True)
                    print(f"✓ 最佳模型已保存 (val_loss: {metrics['val_loss']:.4f})")

            # 定期保存检查点
            if step % cfg.save_interval == 0 and step > 0:
                self.save_checkpoint()
                print(f"✓ 检查点已保存 (step: {step})")

        # 最终保存
        self.save_checkpoint()
        self.writer.close()
        print(f"训练完成！最佳验证损失: {self.best_val_loss:.4f}")