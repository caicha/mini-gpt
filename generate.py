#!/usr/bin/env python
"""
生成入口脚本
"""

import torch
import argparse

from config.config import generation_config, path_config
from src.model import MiniGPT
from src.generate import Generator
from src.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser(description='MiniGPT文本生成')
    parser.add_argument('--prompt', type=str, default='To be', help='提示词')
    parser.add_argument('--max_tokens', type=int, default=generation_config.max_tokens)
    parser.add_argument('--temperature', type=float, default=generation_config.temperature)
    parser.add_argument('--top_k', type=int, default=generation_config.top_k)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 更新配置
    generation_config.max_tokens = args.max_tokens
    generation_config.temperature = args.temperature
    generation_config.top_k = args.top_k

    # 设置随机种子
    set_seed(args.seed)

    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")

    # import numpy
    # 在加载前添加安全全局类型
    # torch.serialization.add_safe_globals([numpy._core.multiarray.scalar, numpy.dtype])
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 恢复模型配置
    from config.config import ModelConfig
    model_config = checkpoint['model_config']
    model_config.vocab_size = checkpoint['vocab_size']

    # 创建模型
    model = MiniGPT(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 加载模型: {args.checkpoint}")
    print(f"模型参数量: {model.get_num_params():,}")

    # 创建生成器
    generator = Generator(
        model=model,
        stoi=checkpoint['stoi'],
        itos=checkpoint['itos'],
        device=device,
        config=generation_config
    )

    # 生成文本
    print(f"\n提示词: {args.prompt}")
    print("-" * 50)
    generated_text = generator.generate(args.prompt)
    print(generated_text)
    print("-" * 50)


if __name__ == "__main__":
    main()