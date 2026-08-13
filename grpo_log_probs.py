"""重新计算 GRPO rollout 中每个 completion token 的概率。

组采样阶段已经保存了 ``old_log_probs``。在优化阶段，我们还需要：

* current_log_probs：当前可训练策略对同一批 token 给出的 log probability；
* reference_log_probs：固定 SFT 参考策略给出的 log probability；
* probability_ratios：当前策略概率与旧策略概率的比值；
* approximate_kl：当前策略相对参考策略的逐 token KL 估计。

刚从 SFT checkpoint 创建策略且尚未更新时，policy、old policy 和 reference
policy 是同一个模型，因此有效 token 上应该满足：

    current_log_probs ≈ old_log_probs ≈ reference_log_probs
    probability_ratios ≈ 1
    approximate_kl ≈ 0

先验证这些关系，再实现 GRPO loss，可以把 token 对齐错误提前暴露出来。
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from countdown import load_countdown_tasks
from grpo_rollout import RolloutBatch, collect_rollouts, load_sft_policy
from sft_countdown import TRAIN_TASKS, VALIDATION_TASKS, choose_device
from tokenizer import CharacterTokenizer
from transformer import Transformer


@dataclass
class LogProbabilityComparison:
    """当前、旧策略和参考策略在同一批 completion token 上的比较结果。"""

    current_log_probs: torch.Tensor
    reference_log_probs: torch.Tensor
    probability_ratios: torch.Tensor
    approximate_kl: torch.Tensor


def load_policy_and_reference(
    device: str,
) -> tuple[Transformer, Transformer, CharacterTokenizer]:
    """从同一个 SFT checkpoint 创建可训练 policy 和固定 reference。

    两个模型初始参数完全相同。policy 后续由优化器更新；reference 的参数关闭
    梯度并始终保持不变，用来约束 policy 不要偏离 SFT 模型太远。
    """
    policy, tokenizer = load_sft_policy(device=device)
    reference, _ = load_sft_policy(device=device)

    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    policy.eval()
    reference.eval()
    return policy, reference, tokenizer


def compute_completion_log_probs(
    model: Transformer,
    rollout: RolloutBatch,
) -> torch.Tensor:
    """重新计算 rollout 中每个 completion token 的 log probability。

    返回形状与 completion_ids 相同：

        [num_tasks, group_size, max_new_tokens]

    对一道题而言，先把同一个 prompt 复制 group_size 份，再分别拼上该组的
    completion。假设 prompt 长度为 P，那么：

    * logits[:, P-1] 预测 completion 的第 0 个 token；
    * logits[:, P]   预测 completion 的第 1 个 token；
    * 以此类推。

    padding 位置也会得到一个数值，但后续必须通过 completion_mask 排除。
    这里不使用 ``torch.no_grad()``，因为 current policy 的 log-prob 需要参与
    反向传播；调用者可在计算 reference policy 时单独关闭梯度。
    """
    device = next(model.parameters()).device
    all_log_probs = []

    for task_index, prompt_token_ids in enumerate(rollout.prompt_ids):
        completion_ids = rollout.completion_ids[task_index].to(device)
        group_size, max_new_tokens = completion_ids.shape

        prompt_ids = torch.tensor(
            prompt_token_ids,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).repeat(group_size, 1)
        full_sequence_ids = torch.cat((prompt_ids, completion_ids), dim=1)

        if full_sequence_ids.shape[1] > model.config.max_seq_len:
            raise ValueError("prompt 和 completion 的总长度超过模型上下文长度")

        logits = model(full_sequence_ids)
        prompt_length = prompt_ids.shape[1]

        # 因果语言模型在位置 t 的 logits 预测位置 t+1 的 token，所以预测第一
        # 个 completion token 的 logits 位于 prompt 最后一个 token 的位置。
        completion_logits = logits[
            :, prompt_length - 1 : prompt_length - 1 + max_new_tokens, :
        ]

        # rollout 是从 logits / temperature 对应的分布中采样的。这里沿用同一个
        # temperature，才能与采样时记录的 old_log_probs 做公平比较。
        completion_log_probs = F.log_softmax(
            completion_logits / rollout.temperature,
            dim=-1,
        )
        selected_log_probs = completion_log_probs.gather(
            dim=-1,
            index=completion_ids.unsqueeze(-1),
        ).squeeze(-1)
        all_log_probs.append(selected_log_probs)

    return torch.stack(all_log_probs)


def compare_policy_log_probs(
    policy: Transformer,
    reference: Transformer,
    rollout: RolloutBatch,
) -> LogProbabilityComparison:
    """计算 current/old/reference 概率关系和逐 token KL 估计。"""
    current_log_probs = compute_completion_log_probs(policy, rollout)

    # reference 只提供一个固定比较基准，不需要构建反向传播计算图。
    with torch.no_grad():
        reference_log_probs = compute_completion_log_probs(reference, rollout)

    old_log_probs = rollout.old_log_probs.to(current_log_probs.device)
    probability_ratios = torch.exp(current_log_probs - old_log_probs)

    # 这是 GRPO/PPO 实现中常用的非负 KL 估计：
    # exp(log(pi_ref) - log(pi)) - (log(pi_ref) - log(pi)) - 1
    reference_log_ratio = reference_log_probs - current_log_probs
    approximate_kl = (
        torch.exp(reference_log_ratio) - reference_log_ratio - 1
    )

    return LogProbabilityComparison(
        current_log_probs=current_log_probs,
        reference_log_probs=reference_log_probs,
        probability_ratios=probability_ratios,
        approximate_kl=approximate_kl,
    )


def main() -> None:
    torch.manual_seed(42)
    device = choose_device()
    policy, reference, tokenizer = load_policy_and_reference(device)

    first_grpo_task = TRAIN_TASKS + VALIDATION_TASKS
    tasks = load_countdown_tasks()[first_grpo_task : first_grpo_task + 2]
    rollout = collect_rollouts(policy, tokenizer, tasks)
    comparison = compare_policy_log_probs(policy, reference, rollout)

    mask = rollout.completion_mask
    current_old_difference = (
        comparison.current_log_probs[mask] - rollout.old_log_probs[mask]
    ).abs()
    reference_old_difference = (
        comparison.reference_log_probs[mask] - rollout.old_log_probs[mask]
    ).abs()
    valid_ratios = comparison.probability_ratios[mask]
    valid_kl = comparison.approximate_kl[mask]

    print(f"device: {device}")
    print(f"log-prob shape: {tuple(comparison.current_log_probs.shape)}")
    print(f"valid completion tokens: {mask.sum().item()}")
    print(
        "max |current - old|: "
        f"{current_old_difference.max().item():.8f}"
    )
    print(
        "max |reference - old|: "
        f"{reference_old_difference.max().item():.8f}"
    )
    print(
        "probability ratio min/mean/max: "
        f"{valid_ratios.min().item():.6f} / "
        f"{valid_ratios.mean().item():.6f} / "
        f"{valid_ratios.max().item():.6f}"
    )
    print(
        "approximate KL mean/max: "
        f"{valid_kl.mean().item():.8f} / {valid_kl.max().item():.8f}"
    )


if __name__ == "__main__":
    main()
