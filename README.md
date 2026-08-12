# GRPO Playground

这个仓库用于从零理解并实现 GRPO。当前先从一个容易阅读的 decoder-only
Transformer 开始，后续再逐步加入生成、奖励计算和 GRPO 训练。

## 最小示例

```python
import torch

from transformer import Transformer, TransformerConfig


config = TransformerConfig(
    vocab_size=1000,
    max_seq_len=128,
    d_model=256,
    n_heads=8,
    n_layers=4,
)
model = Transformer(config)

input_ids = torch.randint(0, config.vocab_size, (2, 16))
logits = model(input_ids)
print(logits.shape)  # torch.Size([2, 16, 1000])
```

核心实现见 `transformer.py`，代码结构为：

1. `CausalSelfAttention`：带因果掩码的多头自注意力。
2. `FeedForward`：逐 token 的两层 MLP。
3. `TransformerBlock`：Pre-LN、注意力、MLP 和两个残差连接。
4. `Transformer`：token/位置嵌入、多个 block、语言模型输出层。

## 在 Tiny Shakespeare 上预训练

安装 PyTorch 后运行：

```bash
python3 train.py
```

`train.py` 会完成字符级 token 化、90%/10% 数据划分、随机采样 batch、
next-token 训练和验证。训练结束后，模型保存在
`checkpoints/tiny_shakespeare.pt`，并在终端输出一段生成文本。
