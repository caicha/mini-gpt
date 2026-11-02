# Mini-GPT：从零构建迷你GPT

## 项目简介
这是一个从零实现的迷你GPT训练项目，用于理解Transformer Decoder-Only架构的核心原理。模型在莎士比亚作品集上训练，能够生成类似风格的文本。

## 这个项目能给你什么？

- **从零理解Transformer**：每个核心概念都有具体数值手算示例
- **完整可运行的训练流程**：数据加载 → 模型定义 → 训练 → 生成
- **中文注释 + 学习手册**：不是只给代码，而是讲清楚"为什么"
- **极简代码**：核心模型不到200行，逐行可读

## 核心特性
- ✅ 完整实现Self-Attention、Multi-Head Attention
- ✅ 支持训练/验证/生成全流程
- ✅ 模型检查点自动保存与恢复
- ✅ 训练损失实时监控
- ✅ 支持CPU和GPU训练

## 安装依赖
```bash
uv pip install -e .

## 开始训练

### 使用默认配置训练
```bash
python train.py
```

### 自定义配置
```bash
python train.py --n_embd 256 --n_head 8 --n_layer 6 --max_iters 5000
```

## 生成文本

### 使用最佳模型生成
```bash
python generate.py --prompt "To be" --max_tokens 200
```

### 使用最新模型生成
```bash
python generate.py --checkpoint checkpoints/latest.pt --prompt "The"
```

## 常见问题
### Q: 训练时显存不足怎么办？

```bash
# 减小batch_size
python train.py --batch_size 16
# 减小模型尺寸
python train.py --n_embd 64 --n_head 2 --n_layer 2
```

### Q: 如何使用GPU训练？

```python
# 代码会自动检测GPU，无需额外配置
# 强制使用CPU：
device = torch.device('cpu')
```

### Q: 训练中断如何恢复？

```bash
python train.py --resume checkpoints/latest.pt
```

## 扩展建议
完成基础训练后，你可以尝试以下扩展：

1. 更大的数据集：替换为完整的维基百科或书籍语料

2. 更深的模型：增加 n_layer 到 12 或更多

3. 更长的上下文：增加 block_size（需要更多显存）

4. 混合精度训练：添加 torch.cuda.amp 加速

5. 对话微调：把数据集换成对话格式，微调成聊天模型
