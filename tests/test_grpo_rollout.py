from types import SimpleNamespace
import unittest

import torch
from torch import nn

from countdown import CountdownTask
from grpo_rollout import (
    collect_rollouts,
    compute_group_advantages,
    sample_completion_group,
)
from tokenizer import build_shared_tokenizer


class NewlinePolicy(nn.Module):
    """一个总是立即生成换行符的假策略，用来精确测试 EOS mask。"""

    def __init__(self, vocab_size: int, newline_id: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(max_seq_len=128, vocab_size=vocab_size)
        self.newline_id = newline_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.config.vocab_size),
            fill_value=float("-inf"),
            device=input_ids.device,
        )
        logits[:, :, self.newline_id] = 0.0
        return logits


class GRPORolloutTest(unittest.TestCase):
    def test_advantages_are_normalized_inside_each_group(self):
        rewards = torch.tensor(
            [
                [0.1, 0.3, 1.0, 0.3],
                [0.3, 0.3, 0.3, 0.3],
            ]
        )

        advantages = compute_group_advantages(rewards)

        self.assertAlmostEqual(advantages[0].mean().item(), 0.0, places=6)
        self.assertAlmostEqual(
            advantages[0].std(correction=0).item(), 1.0, places=6
        )
        self.assertTrue(torch.all(advantages[1] == 0))

    def test_eos_is_kept_but_tokens_after_eos_are_masked(self):
        tokenizer = build_shared_tokenizer("Shakespeare\n")
        model = NewlinePolicy(
            vocab_size=tokenizer.vocab_size,
            newline_id=tokenizer.char_to_id["\n"],
        )
        task = CountdownTask(numbers=(2, 5, 7), target=19)

        completions, completion_ids, old_log_probs, mask = (
            sample_completion_group(
                model,
                tokenizer,
                task,
                group_size=3,
                max_new_tokens=4,
            )
        )

        self.assertEqual(completions, ["", "", ""])
        self.assertEqual(completion_ids.shape, (3, 4))
        self.assertTrue(torch.all(mask[:, 0]))
        self.assertFalse(torch.any(mask[:, 1:]))
        self.assertTrue(torch.all(old_log_probs == 0))

    def test_collect_rollouts_returns_grouped_tensors(self):
        tokenizer = build_shared_tokenizer("Shakespeare\n")
        model = NewlinePolicy(
            vocab_size=tokenizer.vocab_size,
            newline_id=tokenizer.char_to_id["\n"],
        )
        tasks = [
            CountdownTask(numbers=(2, 5, 7), target=19),
            CountdownTask(numbers=(3, 4, 6), target=18),
        ]

        rollout = collect_rollouts(
            model,
            tokenizer,
            tasks,
            group_size=3,
            max_new_tokens=4,
        )

        self.assertEqual(rollout.completion_ids.shape, (2, 3, 4))
        self.assertEqual(rollout.old_log_probs.shape, (2, 3, 4))
        self.assertEqual(rollout.completion_mask.shape, (2, 3, 4))
        self.assertEqual(rollout.rewards.shape, (2, 3))
        self.assertEqual(rollout.advantages.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
