"""使用 Tiny Shakespeare 数据集预训练一个小型 Transformer 语言模型。

这个脚本刻意把训练过程写得比较直接，方便观察语言模型预训练的完整流程：

1. 读取原始文本，并将每个字符转换成一个整数 token；
2. 从连续的 token 序列中随机截取训练样本；
3. 让 Transformer 根据前面的字符预测下一个字符；
4. 使用交叉熵计算预测误差，通过反向传播更新模型参数；
5. 定期在训练集和验证集上估算 loss；
6. 保存训练好的参数，并让模型生成一段文本。
"""

from dataclasses import asdict
from pathlib import Path

import torch

from transformer import Transformer, TransformerConfig


# 原始文本的位置，以及训练结束后模型参数的保存位置。
DATA_PATH = Path("data/tiny_shakespeare.txt")
CHECKPOINT_PATH = Path("checkpoints/tiny_shakespeare.pt")

# 下面这些常量是最主要的训练超参数。
#
# BATCH_SIZE 表示每一步同时训练多少段文本。
# SEQ_LEN 表示每段文本包含多少个字符，也就是 Transformer 的上下文长度。
# MAX_STEPS 表示一共更新多少次模型参数。
# EVAL_INTERVAL 表示每隔多少步评估一次模型。
# EVAL_STEPS 表示评估 loss 时随机取多少个 batch 求平均值。
# LEARNING_RATE 控制每次参数更新的幅度。
BATCH_SIZE = 32
SEQ_LEN = 128
MAX_STEPS = 2_000
EVAL_INTERVAL = 200
EVAL_STEPS = 20
LEARNING_RATE = 3e-4


class CharacterTokenizer:
    """最简单的字符级 tokenizer。

    它会收集数据集中出现过的所有字符，并为每个字符分配一个整数 ID。
    例如，假设字符表是 ["\n", "a", "b"]，那么 "ab" 会被编码成
    [1, 2]。这种方法不涉及 BPE 等复杂算法，很适合用来学习语言模型训练流程。
    """

    def __init__(self, text: str):
        # set(text) 去除重复字符，sorted(...) 保证每次运行时字符 ID 都相同。
        self.characters = sorted(set(text))

        # 编码时使用 char_to_id，解码时使用方向相反的 id_to_char。
        self.char_to_id = {char: index for index, char in enumerate(self.characters)}
        self.id_to_char = {index: char for index, char in enumerate(self.characters)}

    @property
    def vocab_size(self) -> int:
        """返回词表大小，也就是数据集中不同字符的数量。"""
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        """把字符串转换成 token ID 列表。"""
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        """把 token ID 列表还原成字符串。"""
        return "".join(self.id_to_char[token_id] for token_id in token_ids)


def choose_device() -> str:
    """选择当前机器上可用的计算设备。

    CUDA 对应 NVIDIA GPU，MPS 对应 Apple Silicon GPU。如果两者都不可用，
    就使用 CPU。对于这个教学用小模型，CPU 也可以运行，只是速度会慢一些。
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_batch(
    tokens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从完整 token 序列中随机构造一个训练 batch。

    inputs 和 targets 的形状都是 [batch_size, seq_len]，但 targets 相比
    inputs 向后移动了一个字符。例如原始 token 是：

        [10, 20, 30, 40]

    那么一组长度为 3 的训练样本就是：

        inputs  = [10, 20, 30]
        targets = [20, 30, 40]

    因此模型在每个位置上学习的任务都是“根据截至当前位置的内容，预测下一个
    字符”。这就是自回归语言模型的 next-token prediction 目标。
    """
    # 为 batch 中的每个样本随机选择一个起点。最后需要额外保留一个 target
    # 字符，因此起点最大只能到 len(tokens) - seq_len - 1。
    start_positions = torch.randint(len(tokens) - seq_len, (batch_size,))

    # inputs 从 start 开始截取 seq_len 个 token。
    inputs = torch.stack(
        [tokens[start : start + seq_len] for start in start_positions]
    )

    # targets 从 start + 1 开始，因此它恰好是 inputs 向后错开一个字符的结果。
    targets = torch.stack(
        [tokens[start + 1 : start + seq_len + 1] for start in start_positions]
    )

    # 数据默认保存在 CPU；这里只把当前 batch 搬到模型所在的计算设备。
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: Transformer,
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    device: str,
) -> dict[str, float]:
    """分别估算模型在训练集和验证集上的平均 loss。

    单个随机 batch 的 loss 波动比较大，所以这里取 EVAL_STEPS 个随机 batch
    的平均值。验证集没有参与参数更新，它的 loss 更能反映模型面对未见文本时
    的预测能力。
    """
    # eval() 会关闭 Dropout，使每次验证使用完整且确定的网络。
    model.eval()
    losses = {}

    # 对训练集和验证集执行完全相同的 loss 计算，方便比较是否出现过拟合。
    for split, tokens in (
        ("train", train_tokens),
        ("validation", validation_tokens),
    ):
        split_losses = []
        for _ in range(EVAL_STEPS):
            inputs, targets = make_batch(tokens, BATCH_SIZE, SEQ_LEN, device)
            split_losses.append(model.loss(inputs, targets).item())
        losses[split] = sum(split_losses) / len(split_losses)

    # 验证结束后恢复训练模式，重新启用 Dropout。
    model.train()
    return losses


def main() -> None:
    # 固定随机种子，使模型初始化和 batch 采样尽可能可以复现。
    torch.manual_seed(42)
    device = choose_device()

    # 读取完整文本，建立字符表，再将整个数据集编码成一条连续的 token 序列。
    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer(text)
    all_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 前 90% 用于更新模型参数，后 10% 只用于验证。这里按文本位置切分，
    # 而不是先随机打乱字符，因为语言模型必须保留字符原本的先后关系。
    split_position = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_position]
    validation_tokens = all_tokens[split_position:]

    # 这是一个约 82 万参数的小模型。词表大小由数据决定，最大序列长度必须与
    # make_batch 使用的 SEQ_LEN 一致。d_model、n_heads 和 n_layers 分别控制
    # 隐藏向量宽度、注意力头数和 Transformer Block 数量。
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=SEQ_LEN,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1,
    )
    model = Transformer(model_config).to(device)

    # AdamW 是 Transformer 训练中常用的优化器。它会根据每个参数的梯度，
    # 按照 LEARNING_RATE 指定的学习率更新参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # numel() 返回一个张量中的标量数量，把所有参数相加即可得到模型参数量。
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"vocabulary size: {tokenizer.vocab_size}")
    print(f"model parameters: {parameter_count:,}")

    model.train()
    for step in range(MAX_STEPS):
        # 定期计算训练集和验证集 loss。estimate_loss 被 @torch.no_grad() 修饰，
        # 所以评估过程不会建立反向传播所需的计算图，也不会更新任何参数。
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model, train_tokens, validation_tokens, device)
            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"validation loss {losses['validation']:.4f}"
            )

        # 每一步都重新随机截取一个 batch，让模型逐渐看到训练文本的不同片段。
        inputs, targets = make_batch(train_tokens, BATCH_SIZE, SEQ_LEN, device)

        # Transformer 输出每个位置对所有字符的 logits，model.loss 使用交叉熵
        # 比较 logits 和正确的下一个字符 targets。
        loss = model.loss(inputs, targets)

        # PyTorch 默认会累积梯度，所以反向传播前必须清空上一步的梯度。
        optimizer.zero_grad(set_to_none=True)

        # backward() 根据 loss 沿计算图反向计算每个可训练参数的梯度。
        loss.backward()

        # optimizer.step() 使用刚刚得到的梯度真正修改模型参数。
        optimizer.step()

    # checkpoint 不仅保存模型权重，还保存模型结构配置和 tokenizer 字符表。
    # 这样以后加载模型时，能够恢复与训练时完全一致的网络和 token 映射。
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model_config),
            "model_state_dict": model.state_dict(),
            "characters": tokenizer.characters,
        },
        CHECKPOINT_PATH,
    )
    print(f"checkpoint saved to {CHECKPOINT_PATH}")

    # 生成时关闭 Dropout，并用一个换行符作为初始提示。Tiny Shakespeare 的
    # 字符经过排序后，ID 0 对应换行符。模型每次预测一个新字符，再把这个字符
    # 拼回输入中继续预测，重复 400 次便得到一段自回归生成文本。
    model.eval()
    prompt = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(prompt, max_new_tokens=400)[0].tolist()
    print("\n--- generated sample ---")
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
