"""计算 GRPO 的 clipped policy loss 和 KL 惩罚。

前面的组采样已经为每个回答准备了 reward、advantage 和 old log-prob，
``grpo_log_probs.py`` 又计算了 current/reference log-prob。本文件把这些量组合成
一个可以执行 ``backward()`` 的标量 loss：

    ratio_t = pi_current(a_t | s_t) / pi_old(a_t | s_t)

    policy_objective_t = min(
        ratio_t * advantage,
        clip(ratio_t, 1-epsilon, 1+epsilon) * advantage,
    )

    loss = -policy_objective + beta * KL(current || reference)

其中 ``a_t`` 是 completion 在位置 t 实际生成的 token。padding 位置通过
``completion_mask`` 排除，不参与 loss。
"""

from dataclasses import dataclass

import torch

from countdown import load_countdown_tasks
from grpo_log_probs import (
    LogProbabilityComparison,
    compare_policy_log_probs,
    load_policy_and_reference,
)
from grpo_rollout import RolloutBatch, collect_rollouts
from sft_countdown import TRAIN_TASKS, VALIDATION_TASKS, choose_device


CLIP_EPSILON = 0.2
KL_BETA = 0.01


@dataclass
class GRPOLossOutput:
    """GRPO loss 及便于观察训练状态的几个分量。"""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    mean_kl: torch.Tensor
    clip_fraction: torch.Tensor


def masked_completion_mean(
    values: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """先对每个回答的有效 token 求平均，返回每个回答各自的均值。

    ``values`` 和 ``completion_mask`` 的形状都是：

        [num_tasks, group_size, max_new_tokens]

    返回形状是：

        [num_tasks, group_size]

    每个回答先独立除以自己的有效 token 数量，可以避免长回答因为 token 更多而
    在整体 loss 中占据更大权重。
    """
    mask = completion_mask.to(values.dtype)
    valid_token_counts = mask.sum(dim=-1)
    return (values * mask).sum(dim=-1) / valid_token_counts


def compute_grpo_loss(
    comparison: LogProbabilityComparison,
    rollout: RolloutBatch,
    clip_epsilon: float = CLIP_EPSILON,
    kl_beta: float = KL_BETA,
) -> GRPOLossOutput:
    """把概率比、组内 advantage 和 KL 组合成 GRPO loss。

    advantage 的形状是 ``[num_tasks, group_size]``，一个完整回答共享同一个
    advantage；概率比的形状多一个 token 维度，所以先用 ``unsqueeze(-1)``
    把 advantage 变成 ``[num_tasks, group_size, 1]``，再广播到回答的每个 token。
    """
    ratios = comparison.probability_ratios   #probability_ratios = torch.exp(current_log_probs - old_log_probs)
    advantages = rollout.advantages.to(ratios.device).unsqueeze(-1)
    completion_mask = rollout.completion_mask.to(ratios.device)

    # 未裁剪目标：奖励高的回答 advantage 为正，会提高这些 token 的概率；
    # 奖励低的回答 advantage 为负，会降低这些 token 的概率。
    unclipped_objective = ratios * advantages

    # 限制 current policy 相对 old policy 的单次变化幅度。这里不是直接把 ratio
    # 永远裁掉，而是与未裁剪目标取较小值，选择更保守的改进幅度。
    clipped_ratios = ratios.clamp(
        min=1.0 - clip_epsilon,
        max=1.0 + clip_epsilon,
    )
    clipped_objective = clipped_ratios * advantages
    per_token_policy_loss = -torch.minimum(
        unclipped_objective,
        clipped_objective,
    )

    # 先在每个 completion 内对有效 token 求平均，再对题目和组统一求平均。
    # 这样每个回答对最终 loss 的权重相同。
    policy_loss = masked_completion_mean(
        per_token_policy_loss,
        completion_mask,
    ).mean()
    mean_kl = masked_completion_mean(
        comparison.approximate_kl,
        completion_mask,
    ).mean()

    # policy_loss 推动模型偏向高奖励回答，KL 项则限制它不要偏离固定的 SFT
    # reference 太远。
    total_loss = policy_loss + kl_beta * mean_kl

    # clip_fraction 只用于观察有多少有效 token 的 ratio 超出了裁剪区间。
    clipped = (ratios - 1.0).abs() > clip_epsilon
    clip_fraction = clipped[completion_mask].float().mean()

    return GRPOLossOutput(
        total_loss=total_loss,
        policy_loss=policy_loss,
        mean_kl=mean_kl,
        clip_fraction=clip_fraction,
    )


def main() -> None:
    """在一小批真实 Countdown rollout 上演示 loss 和反向传播。"""
    torch.manual_seed(42)
    device = choose_device()
    policy, reference, tokenizer = load_policy_and_reference(device)

    first_grpo_task = TRAIN_TASKS + VALIDATION_TASKS
    tasks = load_countdown_tasks()[first_grpo_task : first_grpo_task + 2]
    rollout = collect_rollouts(policy, tokenizer, tasks)
    comparison = compare_policy_log_probs(policy, reference, rollout)
    loss_output = compute_grpo_loss(comparison, rollout)

    # 初次更新前 ratio=1、KL=0，而且每组 advantage 均值为 0，因此 policy loss
    # 的数值可能非常接近 0。但不同回答的梯度方向不同，仍然可以产生非零梯度。
    loss_output.total_loss.backward()
    gradient_norm = torch.sqrt(
        sum(
            parameter.grad.square().sum()
            for parameter in policy.parameters()
            if parameter.grad is not None
        )
    )

    print(f"device: {device}")
    print(f"total loss: {loss_output.total_loss.item():.6f}")
    print(f"policy loss: {loss_output.policy_loss.item():.6f}")
    print(f"mean KL: {loss_output.mean_kl.item():.8f}")
    print(f"clip fraction: {loss_output.clip_fraction.item():.4f}")
    print(f"policy gradient norm: {gradient_norm.item():.6f}")


if __name__ == "__main__":
    main()
