"""预训练、SFT 和 GRPO 共用的字符级 tokenizer。"""


# 在预训练开始前就把下游 Countdown 会使用的字符加入词表。这样预训练模型和
# SFT 模型的 embedding、lm_head 尺寸及每个字符的 token ID 从一开始就一致。
# 这里只是在词表中预留字符，并没有把 Countdown 答案用于预训练。
COUNTDOWN_CHARACTERS = "Numbers Target Equation: 0123456789()+-*/\n"


class CharacterTokenizer:
    """在字符和整数 token ID 之间进行转换。

    这里不实现 BPE 等子词算法，因为 Countdown 的文本只包含少量英文字母、
    数字和运算符。字符级 tokenizer 足以表达任务，同时也更容易观察每个 token。
    """

    def __init__(self, characters: list[str]):
        self.characters = characters
        self.char_to_id = {
            character: token_id
            for token_id, character in enumerate(self.characters)
        }
        self.id_to_char = {
            token_id: character
            for token_id, character in enumerate(self.characters)
        }

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharacterTokenizer":
        """收集所有文本中出现的字符，并构造稳定的字符表。"""
        characters = sorted(set("".join(texts)))
        return cls(characters)

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        """把字符串转换成 token ID 列表。"""
        return [self.char_to_id[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """把 token ID 列表还原成字符串。"""
        return "".join(self.id_to_char[token_id] for token_id in token_ids)


def build_shared_tokenizer(pretraining_text: str) -> CharacterTokenizer:
    """创建贯穿预训练、SFT 和 GRPO 的统一 tokenizer。

    词表是 Tiny Shakespeare 字符与 Countdown 所需字符的并集。SFT 阶段不会
    重新创建词表，而是直接从预训练 checkpoint 恢复这里生成的字符列表。
    """
    return CharacterTokenizer.from_texts(
        [pretraining_text, COUNTDOWN_CHARACTERS]
    )
