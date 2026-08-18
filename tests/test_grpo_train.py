import copy
import unittest

import torch

from countdown import CountdownTask
from grpo_log_probs import compute_completion_log_probs
from grpo_rollout import RolloutBatch
from grpo_train import optimize_rollout, summarize_rollout
from tokenizer import build_shared_tokenizer
from transformer import Transformer, TransformerConfig


class GRPOTrainTest(unittest.TestCase):
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
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def test_rollout_metrics(self):
        rollout = RolloutBatch(
            tasks=[],
            prompt_ids=[],
            completions=[],
            completion_ids=torch.zeros((2, 2, 3), dtype=torch.long),
            old_log_probs=torch.zeros((2, 2, 3)),
            completion_mask=torch.tensor(
                [
                    [[True, True, False], [True, False, False]],
                    [[True, True, True], [True, True, False]],
                ]
            ),
            rewards=torch.tensor([[1.0, 0.1], [0.3, 0.0]]),
            advantages=torch.zeros((2, 2)),
            temperature=1.0,
        )

        metrics = summarize_rollout(rollout)

        self.assertAlmostEqual(metrics.mean_reward, 0.35, places=6)
        self.assertAlmostEqual(metrics.correct_answer_fraction, 0.25, places=6)
        self.assertAlmostEqual(metrics.solved_task_fraction, 0.5, places=6)
        self.assertAlmostEqual(metrics.mean_completion_length, 2.0, places=6)

    def test_optimization_updates_policy_but_not_reference(self):
        task = CountdownTask(numbers=(2, 5, 7), target=19)
        completion_ids = torch.tensor(
            [
                [
                    self.tokenizer.encode("1\n"),
                    self.tokenizer.encode("2\n"),
                ]
            ],
            dtype=torch.long,
        )
        mask = torch.ones_like(completion_ids, dtype=torch.bool)
        rollout = RolloutBatch(
            tasks=[task],
            prompt_ids=[self.tokenizer.encode(task.prompt)],
            completions=[["1", "2"]],
            completion_ids=completion_ids,
            old_log_probs=torch.zeros_like(completion_ids, dtype=torch.float32),
            completion_mask=mask,
            rewards=torch.tensor([[1.0, 0.0]]),
            advantages=torch.tensor([[1.0, -1.0]]),
            temperature=1.0,
        )
        with torch.no_grad():
            rollout.old_log_probs = compute_completion_log_probs(
                self.policy,
                rollout,
            )

        original_policy = {
            name: parameter.detach().clone()
            for name, parameter in self.policy.named_parameters()
        }
        original_reference = {
            name: parameter.detach().clone()
            for name, parameter in self.reference.named_parameters()
        }
        optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=1e-3,
        )

        metrics = optimize_rollout(
            self.policy,
            self.reference,
            rollout,
            optimizer,
            updates=2,
        )

        policy_changed = any(
            not torch.equal(parameter, original_policy[name])
            for name, parameter in self.policy.named_parameters()
        )
        reference_unchanged = all(
            torch.equal(parameter, original_reference[name])
            for name, parameter in self.reference.named_parameters()
        )
        self.assertTrue(policy_changed)
        self.assertTrue(reference_unchanged)
        self.assertGreater(metrics[0].gradient_norm, 0.0)
        self.assertEqual(len(metrics), 2)


if __name__ == "__main__":
    unittest.main()
