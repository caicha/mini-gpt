#!/usr/bin/env python
"""
训练入口脚本
"""

import torch
import argparse
from pathlib import Path

from config.config import (
    ModelConfig, TrainingConfig, PathConfig,
    model_config, training_config, path_config
)
from src.model import MiniGPT
from src.dataset import CharDataset
from src.trainer import Trainer
from src.utils import set_seed, get_device


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MiniGPT训练脚本')
    parser.add_argument('--batch_size', type=int, default=training_config.batch_size)
    parser.add_argument('--max_iters', type=int, default=training_config.max_iters)
    parser.add_argument('--learning_rate', type=float, default=training_config.learning_rate)
    parser.add_argument('--n_embd', type=int, default=model_config.n_embd)
    parser.add_argument('--n_head', type=int, default=model_config.n_head)
    parser.add_argument('--n_layer', type=int, default=model_config.n_layer)
    parser.add_argument('--block_size', type=int, default=model_config.block_size)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    args = parser.parse_args()

    # 更新配置
    model_config.n_embd = args.n_embd
    model_config.n_head = args.n_head
    model_config.n_layer = args.n_layer
    model_config.block_size = args.block_size
    training_config.batch_size = args.batch_size
    training_config.max_iters = args.max_iters
    training_config.learning_rate = args.learning_rate

    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")

    # 加载数据
    print(f"加载数据: {path_config.data_path}")
    with open(path_config.data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    dataset = CharDataset(text, model_config.block_size)
    model_config.vocab_size = dataset.vocab_size
    print(f"数据集大小: {len(dataset)} 个样本")
    print(f"词典大小: {dataset.vocab_size}")

    # 创建模型
    model = MiniGPT(model_config).to(device)
    print(f"模型参数量: {model.get_num_params():,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        dataset=dataset,
        model_config=model_config,
        training_config=training_config,
        path_config=path_config,
        device=device
    )

    # 恢复训练
    if args.resume:
        if trainer.load_checkpoint(args.resume):
            print(f"✓ 从检查点恢复: {args.resume}")
        else:
            print(f"⚠ 检查点不存在: {args.resume}")

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()