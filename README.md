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

`train.py` 会先建立预训练、SFT 和 GRPO 共用的字符词表，再完成 90%/10%
数据划分、随机采样 batch、next-token 训练和验证。预训练数据仍然只有 Tiny
Shakespeare；Countdown 字符此时只是在统一词表中预留位置。训练结束后，模型
保存在 `checkpoints/tiny_shakespeare.pt`。

## Countdown 任务

`countdown.py` 包含 Countdown Parquet 数据读取、递归求解器和奖励函数。
运行下面的命令可以读取数据，并查看前五道题的求解结果：

```bash
uv run countdown.py
```

奖励由合法表达式、正确使用全部数字和算式结果正确三个部分组成。完整正确的
回答获得 1 分，直接输出目标数字无法获得正确性奖励。

## Countdown 监督热身

正式进行 GRPO 前，先使用求解器生成的 10,000 条答案做一段监督训练：

```bash
uv run sft_countdown.py
```

脚本会从 `checkpoints/tiny_shakespeare.pt` 恢复相同的 tokenizer、模型结构和
全部预训练权重，再进行 SFT。监督 loss 只计算 `Equation:` 后面的答案 token，
不计算 prompt 和 padding。训练后的策略保存在
`checkpoints/countdown_sft.pt`，作为后续 GRPO 的初始策略。

完整顺序为：

```bash
uv run train.py
uv run sft_countdown.py
```
