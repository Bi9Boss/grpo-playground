import unittest

import torch

from countdown import CountdownTask
from sft_countdown import SupervisedExample, make_batch
from tokenizer import build_shared_tokenizer


class CountdownSFTTest(unittest.TestCase):
    def test_batch_only_keeps_answer_targets(self):
        task = CountdownTask(numbers=(2, 5, 7), target=19)
        example = SupervisedExample(prompt=task.prompt, answer="7*2+5")
        tokenizer = build_shared_tokenizer("Shakespeare text\n")

        inputs, targets = make_batch(
            [example], tokenizer, batch_size=1, device="cpu"
        )

        answer_start = len(tokenizer.encode(example.prompt)) - 1
        expected_answer_targets = tokenizer.encode(example.answer + "\n")

        self.assertEqual(inputs.shape, targets.shape)
        self.assertTrue(torch.all(targets[0, :answer_start] == -100))
        self.assertEqual(
            targets[0, answer_start:].tolist(),
            expected_answer_targets,
        )


if __name__ == "__main__":
    unittest.main()
