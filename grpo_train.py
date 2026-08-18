"""在 Countdown 任务上执行完整的 GRPO 训练。

这个脚本把前面实现的模块串成一条完整训练链路：

1. 从 SFT checkpoint 创建可训练 policy 和固定 reference；
2. 当前 policy 为一批题目采样多组回答；
3. 奖励函数为回答打分，并在每道题内部计算 advantage；
4. 在同一批 rollout 上进行多次 GRPO 更新；
5. 定期在未参与训练的 Countdown 题目上评估；
6. 保存最终的 GRPO checkpoint。

同一批 rollout 会重复训练几次。第一次更新前 current policy 与 old policy 相同，
ratio 等于 1；第一次 ``optimizer.step()`` 后 current policy 才发生变化，后续更新
中的 ratio 和 clipping 才会真正发挥作用。
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from countdown import CountdownTask, load_countdown_tasks
from grpo_log_probs import compare_policy_log_probs, load_policy_and_reference
from grpo_loss import CLIP_EPSILON, KL_BETA, compute_grpo_loss
from grpo_rollout import (
    GROUP_SIZE,
    MAX_NEW_TOKENS,
    SFT_CHECKPOINT_PATH,
    RolloutBatch,
    collect_rollouts,
)
from sft_countdown import TRAIN_TASKS, VALIDATION_TASKS, choose_device
from tokenizer import CharacterTokenizer
from transformer import Transformer


CHECKPOINT_PATH = Path("checkpoints/countdown_grpo.pt")

# SFT 使用前 11,000 道题。GRPO 从它们之后继续取数据，并另外留出一部分题目
# 进行独立验证。
GRPO_TRAIN_TASKS = 10_000
GRPO_VALIDATION_TASKS = 100

# 一次 rollout 包含 ROLLOUT_BATCH_SIZE 道题，每道题生成 GROUP_SIZE 个回答。
# 同一批回答复用 UPDATES_PER_ROLLOUT 次，第二次开始 ratio 才会偏离 1。
ROLLOUT_BATCH_SIZE = 4
NUM_ROLLOUTS = 100
UPDATES_PER_ROLLOUT = 4

LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 1.0
EVAL_INTERVAL = 10
EVAL_TASKS = 32


@dataclass(frozen=True)
class RolloutMetrics:
    """一批完整回答的奖励指标。"""

    mean_reward: float
    correct_answer_fraction: float
    solved_task_fraction: float
    mean_completion_length: float


@dataclass(frozen=True)
class UpdateMetrics:
    """一次 optimizer 更新时记录的训练指标。"""

    total_loss: float
    policy_loss: float
    mean_kl: float
    clip_fraction: float
    gradient_norm: float


def summarize_rollout(rollout: RolloutBatch) -> RolloutMetrics:
    """汇总一批 rollout 的奖励、正确率和回答长度。"""
    rewards = rollout.rewards
    correct_answers = rewards == 1.0

    return RolloutMetrics(
        mean_reward=rewards.mean().item(),
        # 所有采样回答中，完整正确回答所占的比例。
        correct_answer_fraction=correct_answers.float().mean().item(),
        # 一道题只要组内至少有一个正确回答，就认为本次采样解决了这道题。
        solved_task_fraction=correct_answers.any(dim=1).float().mean().item(),
        mean_completion_length=(
            rollout.completion_mask.sum(dim=-1).float().mean().item()
        ),
    )


def sample_task_batch(
    tasks: list[CountdownTask],
    batch_size: int,
) -> list[CountdownTask]:
    """从 GRPO 训练题目中随机选择一批不重复的任务。"""
    indices = torch.randperm(len(tasks))[:batch_size].tolist()
    return [tasks[index] for index in indices]


def optimize_rollout(
    policy: Transformer,
    reference: Transformer,
    rollout: RolloutBatch,
    optimizer: torch.optim.Optimizer,
    updates: int = UPDATES_PER_ROLLOUT,
    clip_epsilon: float = CLIP_EPSILON,
    kl_beta: float = KL_BETA,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> list[UpdateMetrics]:
    """保持 rollout 和 old log-prob 不变，对 current policy 更新多次。

    policy 使用 ``eval()`` 关闭 dropout，保证第一次重算的 current log-prob 与
    采样时保存的 old log-prob 一致。``eval()`` 只改变 dropout 等层的行为，
    不会关闭 autograd；current log-prob 仍然拥有完整计算图。
    """
    policy.eval()
    reference.eval()
    update_metrics = []

    for _ in range(updates):
        # 每一轮都要用更新后的 policy 重新计算 current log-prob。old log-prob
        # 继续保持为生成这批 rollout 时的旧策略概率。
        optimizer.zero_grad(set_to_none=True)
        comparison = compare_policy_log_probs(policy, reference, rollout)
        loss_output = compute_grpo_loss(
            comparison,
            rollout,
            clip_epsilon=clip_epsilon,
            kl_beta=kl_beta,
        )
        loss_output.total_loss.backward()

        # 梯度裁剪限制所有参数梯度的整体范数，避免单次更新过大。返回值是裁剪前
        # 的梯度范数，适合用来观察训练是否出现剧烈波动。
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=max_grad_norm,
        )
        optimizer.step()

        update_metrics.append(
            UpdateMetrics(
                total_loss=loss_output.total_loss.item(),
                policy_loss=loss_output.policy_loss.item(),
                mean_kl=loss_output.mean_kl.item(),
                clip_fraction=loss_output.clip_fraction.item(),
                gradient_norm=gradient_norm.item(),
            )
        )

    return update_metrics


@torch.no_grad()
def evaluate_policy(
    policy: Transformer,
    tokenizer: CharacterTokenizer,
    tasks: list[CountdownTask],
) -> RolloutMetrics:
    """在固定的未见题目上采样回答并汇总奖励。"""
    policy.eval()
    rollout = collect_rollouts(
        policy,
        tokenizer,
        tasks,
        group_size=GROUP_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    return summarize_rollout(rollout)


def print_evaluation(label: str, metrics: RolloutMetrics) -> None:
    """用一行文本打印验证指标。"""
    print(
        f"{label} | "
        f"reward {metrics.mean_reward:.3f} | "
        f"correct answers {metrics.correct_answer_fraction:.1%} | "
        f"solved tasks {metrics.solved_task_fraction:.1%} | "
        f"length {metrics.mean_completion_length:.1f}"
    )


def main() -> None:
    torch.manual_seed(42)
    device = choose_device()
    policy, reference, tokenizer = load_policy_and_reference(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE)

    all_tasks = load_countdown_tasks()
    grpo_start = TRAIN_TASKS + VALIDATION_TASKS
    grpo_train_tasks = all_tasks[
        grpo_start : grpo_start + GRPO_TRAIN_TASKS
    ]
    grpo_validation_start = grpo_start + GRPO_TRAIN_TASKS
    grpo_validation_tasks = all_tasks[
        grpo_validation_start : grpo_validation_start
        + GRPO_VALIDATION_TASKS
    ]
    evaluation_tasks = grpo_validation_tasks[:EVAL_TASKS]

    print(f"device: {device}")
    print(f"GRPO training tasks: {len(grpo_train_tasks):,}")
    print(f"GRPO validation tasks: {len(grpo_validation_tasks):,}")
    print(f"rollout batch size: {ROLLOUT_BATCH_SIZE}")
    print(f"group size: {GROUP_SIZE}")
    print(f"updates per rollout: {UPDATES_PER_ROLLOUT}")
    print(f"loaded SFT policy from {SFT_CHECKPOINT_PATH}")

    baseline_metrics = evaluate_policy(policy, tokenizer, evaluation_tasks)
    print_evaluation("baseline", baseline_metrics)

    for rollout_step in range(NUM_ROLLOUTS):
        task_batch = sample_task_batch(
            grpo_train_tasks,
            ROLLOUT_BATCH_SIZE,
        )

        # collect_rollouts 被 no_grad 修饰，只负责生成训练数据。这里保存下来的
        # old log-prob 在随后多次 optimize_rollout 更新中始终保持不变。
        rollout = collect_rollouts(
            policy,
            tokenizer,
            task_batch,
            group_size=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        rollout_metrics = summarize_rollout(rollout)
        updates = optimize_rollout(
            policy,
            reference,
            rollout,
            optimizer,
        )
        last_update = updates[-1]

        print(
            f"rollout {rollout_step + 1:3d}/{NUM_ROLLOUTS} | "
            f"reward {rollout_metrics.mean_reward:.3f} | "
            f"correct {rollout_metrics.correct_answer_fraction:.1%} | "
            f"loss {last_update.total_loss:+.4f} | "
            f"KL {last_update.mean_kl:.5f} | "
            f"clipped {last_update.clip_fraction:.1%} | "
            f"grad {last_update.gradient_norm:.3f}"
        )

        if (rollout_step + 1) % EVAL_INTERVAL == 0:
            validation_metrics = evaluate_policy(
                policy,
                tokenizer,
                evaluation_tasks,
            )
            print_evaluation(
                f"validation after rollout {rollout_step + 1}",
                validation_metrics,
            )

    final_metrics = evaluate_policy(policy, tokenizer, evaluation_tasks)
    print_evaluation("final", final_metrics)

    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "stage": "grpo",
            "parent_checkpoint": str(SFT_CHECKPOINT_PATH),
            "model_config": asdict(policy.config),
            "model_state_dict": policy.state_dict(),
            # AdamW 的动量和方差状态也一并保存，之后若要实现断点续训，可以
            # 同时恢复模型与优化器，而不是重新创建一个没有历史状态的 AdamW。
            "optimizer_state_dict": optimizer.state_dict(),
            "characters": tokenizer.characters,
            "training_config": {
                "num_rollouts": NUM_ROLLOUTS,
                "rollout_batch_size": ROLLOUT_BATCH_SIZE,
                "group_size": GROUP_SIZE,
                "updates_per_rollout": UPDATES_PER_ROLLOUT,
                "learning_rate": LEARNING_RATE,
                "clip_epsilon": CLIP_EPSILON,
                "kl_beta": KL_BETA,
                "max_grad_norm": MAX_GRAD_NORM,
            },
            "baseline_metrics": asdict(baseline_metrics),
            "final_metrics": asdict(final_metrics),
        },
        CHECKPOINT_PATH,
    )
    print(f"checkpoint saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
