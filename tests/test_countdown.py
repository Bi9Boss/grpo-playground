import unittest

from countdown import (
    CountdownTask,
    countdown_reward,
    evaluate_expression,
    extract_expression,
    solve_countdown,
)


class CountdownTest(unittest.TestCase):
    def test_prompt(self):
        task = CountdownTask(numbers=(2, 5, 7), target=19)
        self.assertEqual(task.prompt, "Numbers: 2 5 7\nTarget: 19\nEquation: ")

    def test_extract_expression_accepts_result_after_equal_sign(self):
        self.assertEqual(extract_expression("7 * 2 + 5 = 19"), "7 * 2 + 5")

    def test_evaluate_expression_uses_exact_fractions(self):
        value, numbers = evaluate_expression("1 / 3 * 3")
        self.assertEqual(value, 1)
        self.assertEqual(numbers, [1, 3, 3])

    def test_correct_expression_gets_full_reward(self):
        task = CountdownTask(numbers=(2, 5, 7), target=19)
        result = countdown_reward(task, "7 * 2 + 5")

        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.correct_result)

    def test_using_target_directly_cannot_cheat_reward(self):
        task = CountdownTask(numbers=(2, 5, 7), target=19)
        result = countdown_reward(task, "19")

        self.assertEqual(result.reward, 0.1)
        self.assertFalse(result.numbers_used_once)
        self.assertFalse(result.correct_result)

    def test_invalid_python_is_rejected(self):
        task = CountdownTask(numbers=(2, 5, 7), target=19)
        result = countdown_reward(task, "print(19)")

        self.assertEqual(result.reward, 0.0)
        self.assertFalse(result.valid_expression)

    def test_solver_finds_a_valid_solution(self):
        task = CountdownTask(numbers=(2, 5, 7), target=19)
        solution = solve_countdown(task)

        self.assertIsNotNone(solution)
        self.assertEqual(countdown_reward(task, solution).reward, 1.0)


if __name__ == "__main__":
    unittest.main()
