"""
数据集处理模块
"""

import torch
import numpy as np
from typing import Tuple, Optional


class CharDataset:
    """字符级数据集"""

    def __init__(self, text: str, block_size: int):
        """
        Args:
            text: 原始文本
            block_size: 上下文长度
        """
        self.text = text
        self.block_size = block_size

        # 建立字符映射表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # 编码为数字序列
        self.data = [self.stoi[ch] for ch in text]

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个训练样本

        Args:
            idx: 起始位置
        Returns:
            x: (block_size,) 输入序列
            y: (block_size,) 目标序列（x向右移动一位）
        """
        x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y

    def encode(self, text: str) -> torch.Tensor:
        """将文本编码为token索引"""
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """将token索引解码为文本"""
        return ''.join([self.itos[int(t)] for t in tokens])


def get_batch(
        dataset: CharDataset,
        batch_size: int,
        block_size: int,
        device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    随机采样一个batch的训练数据

    Args:
        dataset: 字符数据集
        batch_size: 批量大小
        block_size: 上下文长度
        device: 设备（cpu/cuda）
    Returns:
        x: (batch_size, block_size)
        y: (batch_size, block_size)
    """
    # 随机选择batch_size个起始位置
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i][0] for i in ix])
    y = torch.stack([dataset[i][1] for i in ix])
    return x.to(device), y.to(device)