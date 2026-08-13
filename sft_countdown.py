"""使用求解器生成的标准答案，对 Countdown 策略进行监督热身。

这个阶段的目标不是让模型记住全部题目，而是先让它学会：

1. Countdown prompt 的文本结构；
2. 回答应该由数字、括号和四则运算符组成；
3. 回答结束时应该生成换行符。

有了这些基础行为，后续 GRPO 采样时才更容易得到不同奖励的回答，形成有效的
组内 advantage。
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from countdown import (
    CountdownTask,
    countdown_reward,
    load_countdown_tasks,
    solve_countdown,
)
from tokenizer import CharacterTokenizer
from transformer import Transformer, TransformerConfig


PRETRAIN_CHECKPOINT_PATH = Path("checkpoints/tiny_shakespeare.pt")
CHECKPOINT_PATH = Path("checkpoints/countdown_sft.pt")

# 前 10,000 道题用于训练，紧随其后的 1,000 道题只用于验证。
TRAIN_TASKS = 10_000
VALIDATION_TASKS = 1_000

BATCH_SIZE = 64
MAX_STEPS = 2_000
EVAL_INTERVAL = 200
EVAL_STEPS = 20
LEARNING_RATE = 3e-4


@dataclass(frozen=True)
class SupervisedExample:
    """一条由 prompt 和求解器标准答案组成的监督样本。"""

    prompt: str
    answer: str

    @property
    def full_text(self) -> str:
        # 换行符既代表答案结束，也是之后生成阶段使用的停止 token。
        return self.prompt + self.answer + "\n"


def choose_device() -> str:
    """优先选择 NVIDIA GPU 或 Apple GPU，否则使用 CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_supervised_examples(
    tasks: list[CountdownTask],
) -> list[SupervisedExample]:
    """使用递归求解器为 Countdown 题目生成标准答案。"""
    examples = []
    for task in tasks:
        solution = solve_countdown(task)
        if solution is not None:
            examples.append(SupervisedExample(prompt=task.prompt, answer=solution))
    return examples


def make_batch(
    examples: list[SupervisedExample],
    tokenizer: CharacterTokenizer,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采样监督样本，并构造带 loss mask 的 input 和 target。

    每条完整序列为：

        prompt + answer + 换行符

    和普通 next-token prediction 一样，target 是 input 向后错开一个 token。
    不同样本长度不同，所以 batch 右侧使用换行符补齐；补齐位置以及 prompt
    对应的 target 都设置成 -100。PyTorch 的 cross_entropy 会忽略 -100，
    因此模型只在答案和答案末尾的换行符上计算监督 loss。
    """
    example_indices = torch.randint(len(examples), (batch_size,)).tolist()
    selected_examples = [examples[index] for index in example_indices]
    encoded_texts = [
        tokenizer.encode(example.full_text) for example in selected_examples
    ]

    sequence_length = max(len(token_ids) - 1 for token_ids in encoded_texts)
    newline_id = tokenizer.char_to_id["\n"]

    inputs = torch.full(
        (batch_size, sequence_length),
        fill_value=newline_id,
        dtype=torch.long,
    )
    targets = torch.full(
        (batch_size, sequence_length),
        fill_value=-100,
        dtype=torch.long,
    )

    for row, (example, token_ids) in enumerate(zip(selected_examples, encoded_texts)):
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        prompt_length = len(tokenizer.encode(example.prompt))

        inputs[row, : len(input_ids)] = torch.tensor(input_ids)

        # target 的第 prompt_length - 1 个位置，预测的正好是答案第一个字符。
        # 从这个位置开始保留标签，前面的 prompt 标签继续维持 -100。
        answer_start = prompt_length - 1
        targets[row, answer_start : len(target_ids)] = torch.tensor(
            target_ids[answer_start:]
        )

    return inputs.to(device), targets.to(device)


def supervised_loss(
    model: Transformer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """只在 targets 不为 -100 的答案位置计算平均交叉熵。"""
    logits = model(inputs)
    return F.cross_entropy(
        logits.reshape(-1, model.config.vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def estimate_loss(
    model: Transformer,
    examples: list[SupervisedExample],
    tokenizer: CharacterTokenizer,
    device: str,
) -> float:
    """在若干随机验证 batch 上估算平均监督 loss。"""
    model.eval()
    losses = []
    for _ in range(EVAL_STEPS):
        inputs, targets = make_batch(examples, tokenizer, BATCH_SIZE, device)
        losses.append(supervised_loss(model, inputs, targets).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_completion(
    model: Transformer,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 32,
) -> str:
    """使用贪心解码生成一个算式，遇到换行符时停止。"""
    token_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    generated_ids = []
    newline_id = tokenizer.char_to_id["\n"]

    model.eval()
    for _ in range(max_new_tokens):
        model_input = token_ids[:, -model.config.max_seq_len :]
        next_token_logits = model(model_input)[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        next_token_id = next_token.item()

        if next_token_id == newline_id:
            break

        generated_ids.append(next_token_id)
        token_ids = torch.cat((token_ids, next_token), dim=1)

    return tokenizer.decode(generated_ids)


def main() -> None:
    torch.manual_seed(42)
    device = choose_device()

    tasks = load_countdown_tasks()
    supervised_tasks = tasks[: TRAIN_TASKS + VALIDATION_TASKS]
    examples = build_supervised_examples(supervised_tasks)

    train_examples = examples[:TRAIN_TASKS]
    validation_examples = examples[TRAIN_TASKS:]

    # SFT 不再创建新词表或随机初始化新模型，而是完整恢复预训练阶段的
    # tokenizer、模型结构和全部权重。这样两个训练阶段属于同一条模型血缘。
    checkpoint = torch.load(PRETRAIN_CHECKPOINT_PATH, map_location=device)
    tokenizer = CharacterTokenizer(checkpoint["characters"])
    config = TransformerConfig(**checkpoint["model_config"])
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 切换训练阶段时重新创建优化器，但被优化的仍然是刚刚加载的预训练参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device: {device}")
    print(f"training examples: {len(train_examples):,}")
    print(f"validation examples: {len(validation_examples):,}")
    print(f"vocabulary: {''.join(tokenizer.characters)!r}")
    print(f"model parameters: {parameter_count:,}")
    print(f"loaded pretrained model from {PRETRAIN_CHECKPOINT_PATH}")

    model.train()
    for step in range(MAX_STEPS):
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            validation_loss = estimate_loss(
                model, validation_examples, tokenizer, device
            )
            print(f"step {step:4d} | validation loss {validation_loss:.4f}")

        inputs, targets = make_batch(
            train_examples, tokenizer, BATCH_SIZE, device
        )
        loss = supervised_loss(model, inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "stage": "sft",
            "parent_checkpoint": str(PRETRAIN_CHECKPOINT_PATH),
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "characters": tokenizer.characters,
        },
        CHECKPOINT_PATH,
    )
    print(f"checkpoint saved to {CHECKPOINT_PATH}")

    print("\n--- validation samples ---")
    for task in tasks[TRAIN_TASKS : TRAIN_TASKS + 5]:
        completion = generate_completion(model, tokenizer, task.prompt)
        reward = countdown_reward(task, completion)
        print(task.prompt, end="")
        print(f"{completion!r}  reward={reward.reward:.1f}\n")


if __name__ == "__main__":
    main()
