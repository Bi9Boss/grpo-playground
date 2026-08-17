import unittest

import torch

from grpo_log_probs import LogProbabilityComparison
from grpo_loss import compute_grpo_loss, masked_completion_mean
from grpo_rollout import RolloutBatch


def make_rollout(
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
) -> RolloutBatch:
    """创建只包含 loss 测试所需字段的最小 rollout。"""
    shape = completion_mask.shape
    return RolloutBatch(
        tasks=[],
        prompt_ids=[],
        completions=[],
        completion_ids=torch.zeros(shape, dtype=torch.long),
        old_log_probs=torch.zeros(shape),
        completion_mask=completion_mask,
        rewards=torch.zeros_like(advantages),
        advantages=advantages,
        temperature=1.0,
    )


def make_comparison(
    ratios: torch.Tensor,
    approximate_kl: torch.Tensor | None = None,
) -> LogProbabilityComparison:
    """创建已知 ratio 和 KL 的概率比较结果。"""
    if approximate_kl is None:
        approximate_kl = torch.zeros_like(ratios)
    return LogProbabilityComparison(
        current_log_probs=torch.zeros_like(ratios),
        reference_log_probs=torch.zeros_like(ratios),
        probability_ratios=ratios,
        approximate_kl=approximate_kl,
    )


class GRPOLossTest(unittest.TestCase):
    def test_masked_mean_gives_each_completion_equal_weight(self):
        values = torch.tensor([[[1.0, 3.0, 100.0], [5.0, 100.0, 100.0]]])
        mask = torch.tensor([[[True, True, False], [True, False, False]]])

        completion_means = masked_completion_mean(values, mask)

        self.assertTrue(
            torch.allclose(completion_means, torch.tensor([[2.0, 5.0]]))
        )

    def test_clipping_uses_the_more_conservative_objective(self):
        # 正 advantage 的 ratio=1.5 被限制到 1.2；负 advantage 的 ratio=0.5
        # 被限制到 0.8。两个回答的目标分别为 1.2 和 -0.8。
        ratios = torch.tensor([[[1.5], [0.5]]])
        advantages = torch.tensor([[1.0, -1.0]])
        mask = torch.ones_like(ratios, dtype=torch.bool)
        rollout = make_rollout(advantages, mask)

        output = compute_grpo_loss(
            make_comparison(ratios),
            rollout,
            clip_epsilon=0.2,
            kl_beta=0.0,
        )

        self.assertAlmostEqual(output.policy_loss.item(), -0.2, places=6)
        self.assertAlmostEqual(output.clip_fraction.item(), 1.0, places=6)

    def test_total_loss_keeps_gradient_only_for_valid_tokens(self):
        current_log_probs = torch.zeros((1, 2, 3), requires_grad=True)
        old_log_probs = torch.zeros_like(current_log_probs)
        reference_log_probs = torch.zeros_like(current_log_probs)
        ratios = torch.exp(current_log_probs - old_log_probs)
        reference_log_ratio = reference_log_probs - current_log_probs
        approximate_kl = (
            torch.exp(reference_log_ratio) - reference_log_ratio - 1
        )
        comparison = LogProbabilityComparison(
            current_log_probs=current_log_probs,
            reference_log_probs=reference_log_probs,
            probability_ratios=ratios,
            approximate_kl=approximate_kl,
        )
        mask = torch.tensor([[[True, True, False], [True, False, False]]])
        rollout = make_rollout(torch.tensor([[1.0, -1.0]]), mask)

        output = compute_grpo_loss(comparison, rollout)
        output.total_loss.backward()

        self.assertTrue((current_log_probs.grad[mask] != 0).all())
        self.assertTrue((current_log_probs.grad[~mask] == 0).all())

    def test_kl_penalty_is_added_to_policy_loss(self):
        ratios = torch.ones((1, 2, 2))
        advantages = torch.tensor([[1.0, -1.0]])
        mask = torch.ones_like(ratios, dtype=torch.bool)
        approximate_kl = torch.tensor([[[0.2, 0.4], [0.6, 0.8]]])
        rollout = make_rollout(advantages, mask)

        output = compute_grpo_loss(
            make_comparison(ratios, approximate_kl),
            rollout,
            kl_beta=0.5,
        )

        self.assertAlmostEqual(output.policy_loss.item(), 0.0, places=6)
        self.assertAlmostEqual(output.mean_kl.item(), 0.5, places=6)
        self.assertAlmostEqual(output.total_loss.item(), 0.25, places=6)


if __name__ == "__main__":
    unittest.main()
