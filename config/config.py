"""
配置文件：所有超参数和路径统一管理
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """模型架构配置"""
    vocab_size: int = -1  # 动态计算，由数据决定
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1
    bias: bool = False  # 是否使用偏置


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 32
    max_iters: int = 3000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eval_interval: int = 100
    eval_iters: int = 200
    save_interval: int = 500
    log_interval: int = 100
    gradient_clip: float = 1.0
    warmup_iters: int = 100
    warmup_ratio: float = 0.1


@dataclass
class PathConfig:
    """路径配置"""
    data_path: str = "data/shakespeare.txt"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    def __post_init__(self):
        # 自动创建必要的目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class GenerationConfig:
    """文本生成配置"""
    max_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    seed: Optional[int] = None


# 全局配置实例
model_config = ModelConfig()
training_config = TrainingConfig()
path_config = PathConfig()
generation_config = GenerationConfig()