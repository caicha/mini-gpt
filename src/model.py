"""
MiniGPT模型定义
包含：Self-Attention, Multi-Head Attention, FeedForward, Transformer Block
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config.config import ModelConfig


class Head(nn.Module):
    """单头自注意力机制"""

    def __init__(self, head_size: int, block_size: int, n_embd: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 因果掩码：确保只能看到当前位置之前的token
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) 输入张量
        Returns:
            out: (B, T, head_size) 注意力输出
        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # 计算注意力分数（点积）
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)

        # 因果掩码：遮挡未来的位置
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 加权求和
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd必须能被n_head整除"

        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([
            Head(head_size, config.block_size, config.n_embd, config.dropout)
            for _ in range(config.n_head)
        ])
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """前馈网络（MLP）"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Transformer Block: Attention + FeedForward + 残差连接"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差连接：x + 子层(x)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Mini-GPT主模型"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 词嵌入 + 位置嵌入
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer层堆叠
        self.blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer)
        ])

        # 最终LayerNorm + 输出层
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: (B, T) 输入token索引
            targets: (B, T) 目标token索引（训练时使用）
        Returns:
            logits: (B, T, vocab_size)
            loss: 交叉熵损失（如果targets不为None）
        """
        B, T = idx.shape
        assert T <= self.config.block_size, f"序列长度{T}超过block_size{self.config.block_size}"

        # token嵌入 + 位置嵌入
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        x = tok_emb + pos_emb

        # 经过Transformer层
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失
        loss = None
        if targets is not None:
            # 重塑为二维张量以便计算交叉熵
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
            self,
            idx: torch.Tensor,
            max_tokens: int = 200,
            temperature: float = 0.8,
            top_k: int = 40
    ) -> torch.Tensor:
        """
        生成文本

        Args:
            idx: (B, T) 起始token索引
            max_tokens: 最大生成token数
            temperature: 温度参数（控制多样性）
            top_k: Top-K采样参数
        Returns:
            idx: (B, T + max_tokens) 生成的token索引
        """
        self.eval()
        for _ in range(max_tokens):
            # 如果序列超过block_size，截断
            idx_cond = idx[:, -self.config.block_size:]
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只取最后一个位置
            # 温度缩放
            logits = logits / temperature
            # Top-K采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Softmax转概率并采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def get_num_params(self) -> int:
        """获取模型参数量"""
        return sum(p.numel() for p in self.parameters())