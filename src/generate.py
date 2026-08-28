"""
文本生成模块
"""

import torch
from typing import Optional

from config.config import GenerationConfig
from src.model import MiniGPT


class Generator:
    """文本生成器"""

    def __init__(
            self,
            model: MiniGPT,
            stoi: dict,
            itos: dict,
            device: torch.device,
            config: GenerationConfig
    ):
        self.model = model
        self.stoi = stoi
        self.itos = itos
        self.device = device
        self.config = config

        self.model.to(device)
        self.model.eval()

    def encode(self, text: str) -> torch.Tensor:
        """将文本编码为token索引"""
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        """将token索引解码为文本"""
        return ''.join([self.itos[int(t)] for t in tokens])

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        根据提示词生成文本

        Args:
            prompt: 提示词（起始文本）
        Returns:
            生成的完整文本
        """
        # 编码提示词
        idx = self.encode(prompt).unsqueeze(0).to(self.device)

        # 生成
        cfg = self.config
        generated_idx = self.model.generate(
            idx,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k
        )

        # 解码并返回
        full_text = self.decode(generated_idx[0])
        return full_text