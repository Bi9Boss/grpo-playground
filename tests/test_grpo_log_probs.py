import copy
import unittest

import torch

from countdown import CountdownTask
from grpo_log_probs import (
    compare_policy_log_probs,
    compute_completion_log_probs,
)
from grpo_rollout import collect_rollouts
from tokenizer import build_shared_tokenizer
from transformer import Transformer, TransformerConfig


class GRPOLogProbabilityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = build_shared_tokenizer("Shakespeare\n")
        config = TransformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=128,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
        )
        self.policy = Transformer(config)
        self.policy.eval()
        self.reference = copy.deepcopy(self.policy)
        self.reference.eval()

        tasks = [
            CountdownTask(numbers=(2, 5, 7), target=19),
            CountdownTask(numbers=(3, 4, 6), target=18),
        ]
        self.rollout = collect_rollouts(
            self.policy,
            self.tokenizer,
            tasks,
            group_size=3,
            max_new_tokens=4,
            temperature=0.8,
        )

    def test_recomputed_log_probs_match_sampling_log_probs(self):
        recomputed = compute_completion_log_probs(self.policy, self.rollout)
        mask = self.rollout.completion_mask

        self.assertEqual(recomputed.shape, self.rollout.old_log_probs.shape)
        self.assertTrue(
            torch.allclose(
                recomputed[mask],
                self.rollout.old_log_probs[mask],
                atol=1e-6,
            )
        )

    def test_identical_models_start_with_ratio_one_and_kl_zero(self):
        comparison = compare_policy_log_probs(
            self.policy,
            self.reference,
            self.rollout,
        )
        mask = self.rollout.completion_mask

        self.assertTrue(
            torch.allclose(
                comparison.probability_ratios[mask],
                torch.ones_like(comparison.probability_ratios[mask]),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                comparison.approximate_kl[mask],
                torch.zeros_like(comparison.approximate_kl[mask]),
                atol=1e-6,
            )
        )

    def test_current_log_probs_keep_gradient_reference_does_not(self):
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

        comparison = compare_policy_log_probs(
            self.policy,
            self.reference,
            self.rollout,
        )

        self.assertTrue(comparison.current_log_probs.requires_grad)
        self.assertFalse(comparison.reference_log_probs.requires_grad)


if __name__ == "__main__":
    unittest.main()
