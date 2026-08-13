"""GRPO 的组采样（group rollout）阶段。

对每道 Countdown 题，当前策略会采样 GROUP_SIZE 个不同回答。采样结果不仅
包含回答文本，还需要保留后续计算 GRPO loss 所需的信息：

* completion_ids：模型实际生成的 token；
* old_log_probs：生成这些 token 时，旧策略给出的 log probability；
* completion_mask：哪些位置属于有效回答，哪些只是 padding；
* rewards：每个完整回答得到的标量奖励；
* advantages：奖励在同一题内部标准化后的相对优势。

这一阶段使用 ``torch.no_grad()``，只负责采样训练数据，不更新模型参数。
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from countdown import CountdownTask, countdown_reward, load_countdown_tasks
from sft_countdown import TRAIN_TASKS, VALIDATION_TASKS, choose_device
from tokenizer import CharacterTokenizer
from transformer import Transformer, TransformerConfig


SFT_CHECKPOINT_PATH = Path("checkpoints/countdown_sft.pt")

GROUP_SIZE = 8
MAX_NEW_TOKENS = 32
TEMPERATURE = 1.0
ADVANTAGE_EPSILON = 1e-8


@dataclass
class RolloutBatch:
    """一批按题目分组的 GRPO 采样结果。

    前两个维度始终是 ``[题目数量, 每题回答数量]``，第三个维度是固定的最大
    生成长度。prompt_ids 使用普通列表保存，因为不同题目的 prompt 长度可能
    不同；其他数据可以整齐地堆叠成 Tensor。
    """

    tasks: list[CountdownTask]
    prompt_ids: list[list[int]]
    completions: list[list[str]]
    completion_ids: torch.Tensor
    old_log_probs: torch.Tensor
    completion_mask: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    temperature: float


def load_sft_policy(
    checkpoint_path: Path = SFT_CHECKPOINT_PATH,
    device: str = "cpu",
) -> tuple[Transformer, CharacterTokenizer]:
    """从 SFT checkpoint 恢复即将进行 GRPO 的策略模型。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint["stage"] != "sft":
        raise ValueError("GRPO rollout 必须从 SFT checkpoint 开始")

    tokenizer = CharacterTokenizer(checkpoint["characters"])
    config = TransformerConfig(**checkpoint["model_config"])
    model = Transformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def compute_group_advantages(
    rewards: torch.Tensor,
    epsilon: float = ADVANTAGE_EPSILON,
) -> torch.Tensor:
    """在每道题内部对 reward 做标准化。

    输入形状是 ``[num_tasks, group_size]``。每一行独立计算：

        advantage = (reward - group_mean) / (group_std + epsilon)

    高于同组平均 reward 的回答获得正 advantage，低于平均值的回答获得负
    advantage。如果同组回答奖励完全相同，分子全为 0，advantage 也全为 0，
    表示这一组没有提供“哪个回答更好”的学习信号。
    """
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True, correction=0)
    return (rewards - group_mean) / (group_std + epsilon)


@torch.no_grad()
def sample_completion_group(
    model: Transformer,
    tokenizer: CharacterTokenizer,
    task: CountdownTask,
    group_size: int = GROUP_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """为一道题并行采样一组回答。

    返回值依次是：回答文本、生成 token、旧策略 log-prob 和有效 token mask。
    三个 Tensor 的形状都是 ``[group_size, max_new_tokens]``。

    换行符承担 EOS（回答结束符）的作用。EOS token 本身属于有效生成，因此它
    的 mask 为 True；EOS 后面为了对齐而填充的位置，mask 才是 False。
    """
    device = next(model.parameters()).device
    prompt_ids = torch.tensor(
        tokenizer.encode(task.prompt),
        dtype=torch.long,
        device=device,
    )
    if len(prompt_ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt 和 completion 的总长度超过模型上下文长度")

    # 同一道题复制 group_size 份。每一行的 prompt 相同，但 multinomial 会让
    # 各行独立采样，因此能够得到不同候选回答。
    sequence_ids = prompt_ids.unsqueeze(0).repeat(group_size, 1)
    newline_id = tokenizer.char_to_id["\n"]

    completion_ids = torch.full(
        (group_size, max_new_tokens),
        fill_value=newline_id,
        dtype=torch.long,
        device=device,
    )
    old_log_probs = torch.zeros(
        (group_size, max_new_tokens),
        dtype=torch.float32,
        device=device,
    )
    completion_mask = torch.zeros(
        (group_size, max_new_tokens),
        dtype=torch.bool,
        device=device,
    )

    # active 表示哪些回答还没有生成换行符。已经结束的行仍然参与批量 forward，
    # 但它们后续位置的 mask 为 False，不会参与 GRPO loss。
    active = torch.ones(group_size, dtype=torch.bool, device=device)
    model.eval()

    for token_index in range(max_new_tokens):
        next_token_logits = model(sequence_ids)[:, -1, :] / temperature
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
        next_token_probs = next_token_log_probs.exp()
        sampled_token = torch.multinomial(next_token_probs, num_samples=1)

        # 已经结束的回答固定填充换行符，避免继续产生无意义 token。
        sampled_token = torch.where(
            active.unsqueeze(1),
            sampled_token,
            torch.full_like(sampled_token, newline_id),
        )
        sampled_log_prob = next_token_log_probs.gather(1, sampled_token).squeeze(1)

        completion_ids[:, token_index] = sampled_token.squeeze(1)
        old_log_probs[:, token_index] = torch.where(
            active,
            sampled_log_prob,
            torch.zeros_like(sampled_log_prob),
        )
        completion_mask[:, token_index] = active

        sequence_ids = torch.cat((sequence_ids, sampled_token), dim=1)
        active = active & (sampled_token.squeeze(1) != newline_id)

        # 这一组全部生成 EOS 后即可提前结束，不必继续执行模型 forward。
        if not active.any():
            break

    completions = []
    for token_ids, mask in zip(completion_ids, completion_mask):
        # mask 包含最后的换行符；用于 reward 的文本不需要 EOS，所以解码前去掉。
        valid_token_ids = token_ids[mask].tolist()
        if valid_token_ids and valid_token_ids[-1] == newline_id:
            valid_token_ids = valid_token_ids[:-1]
        completions.append(tokenizer.decode(valid_token_ids))

    return completions, completion_ids, old_log_probs, completion_mask


@torch.no_grad()
def collect_rollouts(
    model: Transformer,
    tokenizer: CharacterTokenizer,
    tasks: list[CountdownTask],
    group_size: int = GROUP_SIZE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> RolloutBatch:
    """为一批题目收集完整的分组 rollout，并计算 reward 与 advantage。"""
    all_completions = []
    all_completion_ids = []
    all_old_log_probs = []
    all_completion_masks = []
    all_rewards = []

    for task in tasks:
        completions, completion_ids, old_log_probs, completion_mask = (
            sample_completion_group(
                model=model,
                tokenizer=tokenizer,
                task=task,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        )
        rewards = [
            countdown_reward(task, completion).reward
            for completion in completions
        ]

        all_completions.append(completions)
        all_completion_ids.append(completion_ids)
        all_old_log_probs.append(old_log_probs)
        all_completion_masks.append(completion_mask)
        all_rewards.append(rewards)

    rewards = torch.tensor(
        all_rewards,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )

    return RolloutBatch(
        tasks=tasks,
        prompt_ids=[tokenizer.encode(task.prompt) for task in tasks],
        completions=all_completions,
        completion_ids=torch.stack(all_completion_ids),
        old_log_probs=torch.stack(all_old_log_probs),
        completion_mask=torch.stack(all_completion_masks),
        rewards=rewards,
        advantages=compute_group_advantages(rewards),
        temperature=temperature,
    )


def main() -> None:
    torch.manual_seed(42)
    device = choose_device()
    model, tokenizer = load_sft_policy(device=device)

    # GRPO 使用 SFT 训练集和验证集之后的新题目，避免重复使用前两个阶段的数据。
    first_grpo_task = TRAIN_TASKS + VALIDATION_TASKS
    tasks = load_countdown_tasks()[first_grpo_task : first_grpo_task + 2]
    rollout = collect_rollouts(model, tokenizer, tasks)

    print(f"device: {device}")
    print(f"completion_ids shape: {tuple(rollout.completion_ids.shape)}")
    print(f"old_log_probs shape: {tuple(rollout.old_log_probs.shape)}")
    print(f"completion_mask shape: {tuple(rollout.completion_mask.shape)}")
    print(f"rewards shape: {tuple(rollout.rewards.shape)}")
    print(f"advantages shape: {tuple(rollout.advantages.shape)}")

    for task_index, task in enumerate(tasks):
        print(f"\n{task.prompt}", end="")
        for group_index, completion in enumerate(
            rollout.completions[task_index]
        ):
            reward = rollout.rewards[task_index, group_index].item()
            advantage = rollout.advantages[task_index, group_index].item()
            token_count = rollout.completion_mask[task_index, group_index].sum().item()
            print(
                f"[{group_index}] {completion!r} | "
                f"tokens={token_count:2d} | "
                f"reward={reward:.1f} | advantage={advantage:+.3f}"
            )


if __name__ == "__main__":
    main()
