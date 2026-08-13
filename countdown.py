"""Countdown 数据读取、求解器和奖励函数。

Countdown 任务会给出几个数字和一个目标值。模型需要使用每个数字恰好一次，
通过 ``+``、``-``、``*``、``/`` 和括号组成一个结果等于目标值的表达式。

例如：

    数字：[2, 5, 7]
    目标：19
    答案：7 * 2 + 5

本文件没有直接使用 Python 的 eval() 执行模型输出，而是通过 ast 解析表达式，
只允许整数、四则运算和括号。这样奖励函数既容易理解，也不会执行任意代码。
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as parquet


DATA_PATH = Path("data/countdown/train.parquet")


@dataclass(frozen=True)
class CountdownTask:
    """一条 Countdown 任务。"""

    numbers: tuple[int, ...]
    target: int

    @property
    def prompt(self) -> str:
        """构造交给语言模型的文本提示。"""
        numbers_text = " ".join(str(number) for number in self.numbers)
        return f"Numbers: {numbers_text}\nTarget: {self.target}\nEquation: "


@dataclass(frozen=True)
class RewardResult:
    """奖励总分和各个子项，方便观察模型到底错在什么地方。"""

    reward: float
    valid_expression: bool
    numbers_used_once: bool
    correct_result: bool


def load_countdown_tasks(path: Path = DATA_PATH) -> list[CountdownTask]:
    """从 Parquet 文件读取全部 Countdown 任务。"""
    table = parquet.read_table(path, columns=["nums", "target"])
    rows = table.to_pylist()
    return [
        CountdownTask(numbers=tuple(row["nums"]), target=row["target"])
        for row in rows
    ]


def extract_expression(completion: str) -> str:
    """从模型回答中取出需要验证的算式。

    后续模型可能只输出 ``7*2+5``，也可能输出 ``7*2+5 = 19``。这里取第一行，
    并丢弃等号右侧内容，使两种格式都能交给同一个验证器。
    """
    first_line = completion.strip().splitlines()[0]
    return first_line.split("=", maxsplit=1)[0].strip()


def evaluate_expression(expression: str) -> tuple[Fraction, list[int]]:
    """安全地计算四则运算表达式，同时收集表达式使用的所有整数。

    Fraction 能精确表示分数，因此像 ``1 / 3 * 3`` 不会因为浮点误差而被判错。
    返回的整数列表用于检查每个给定数字是否恰好使用一次。
    """
    tree = ast.parse(expression, mode="eval")
    used_numbers: list[int] = []

    def visit(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return visit(node.body)

        # bool 是 int 的子类，因此这里显式排除 True 和 False。
        if isinstance(node, ast.Constant) and type(node.value) is int:
            used_numbers.append(node.value)
            return Fraction(node.value)

        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right

        raise ValueError("表达式只能包含整数、括号和 + - * / 运算")

    return visit(tree), used_numbers


def countdown_reward(task: CountdownTask, completion: str) -> RewardResult:
    """验证模型回答，并返回 0 到 1 之间的奖励。

    奖励分为三个能够逐步达到的部分：

    1. 0.1 分：回答是合法的四则运算表达式；
    2. 0.2 分：每个给定数字恰好使用一次，且没有使用额外数字；
    3. 0.7 分：表达式的计算结果等于目标值。

    “结果正确”必须建立在“数字使用正确”的基础上，否则模型可以完全忽略题目，
    直接输出目标数字来骗取正确性奖励。
    """
    try:
        expression = extract_expression(completion)
        value, used_numbers = evaluate_expression(expression)
    except (IndexError, SyntaxError, ValueError, ZeroDivisionError):
        return RewardResult(0.0, False, False, False)

    valid_expression = True
    numbers_used_once = Counter(used_numbers) == Counter(task.numbers)
    correct_result = numbers_used_once and value == task.target

    reward = 0.1
    if numbers_used_once:
        reward += 0.2
    if correct_result:
        reward += 0.7

    return RewardResult(
        reward=reward,
        valid_expression=valid_expression,
        numbers_used_once=numbers_used_once,
        correct_result=correct_result,
    )


def solve_countdown(task: CountdownTask) -> str | None:
    """通过递归搜索找到一个正确表达式，找不到时返回 None。

    每次从当前列表取出两个数，尝试四则运算后把结果放回列表。重复这个过程，
    直到只剩一个数。因为每一步都会消耗两个表达式并产生一个新表达式，所以
    原始数字天然只会使用一次。

    这个求解器主要用于生成少量 SFT 示例，不参与 GRPO reward 计算。
    """
    values = [(Fraction(number), str(number)) for number in task.numbers]

    def search(items: list[tuple[Fraction, str]]) -> str | None:
        if len(items) == 1:
            value, expression = items[0]
            return expression if value == task.target else None

        for left_index in range(len(items)):
            for right_index in range(left_index + 1, len(items)):
                left_value, left_expression = items[left_index]
                right_value, right_expression = items[right_index]
                remaining = [
                    item
                    for index, item in enumerate(items)
                    if index not in (left_index, right_index)
                ]

                candidates = [
                    (
                        left_value + right_value,
                        f"({left_expression}+{right_expression})",
                    ),
                    (
                        left_value * right_value,
                        f"({left_expression}*{right_expression})",
                    ),
                    (
                        left_value - right_value,
                        f"({left_expression}-{right_expression})",
                    ),
                    (
                        right_value - left_value,
                        f"({right_expression}-{left_expression})",
                    ),
                ]
                if right_value != 0:
                    candidates.append(
                        (
                            left_value / right_value,
                            f"({left_expression}/{right_expression})",
                        )
                    )
                if left_value != 0:
                    candidates.append(
                        (
                            right_value / left_value,
                            f"({right_expression}/{left_expression})",
                        )
                    )

                # 加法和乘法满足交换律，前面只保留一种顺序；set 用来跳过由于
                # 相同数字而产生的完全重复候选，减少无意义搜索。
                seen_candidates: set[tuple[Fraction, str]] = set()
                for candidate in candidates:
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)

                    solution = search(remaining + [candidate])
                    if solution is not None:
                        return solution

        return None

    return search(values)


if __name__ == "__main__":
    tasks = load_countdown_tasks()
    print(f"读取任务数量: {len(tasks):,}")

    for task in tasks[:5]:
        solution = solve_countdown(task)
        result = countdown_reward(task, solution or "")
        print(task.prompt, end="")
        print(f"{solution}  reward={result.reward:.1f}\n")
