import unittest

from tokenizer import COUNTDOWN_CHARACTERS, build_shared_tokenizer


class SharedTokenizerTest(unittest.TestCase):
    def test_shared_tokenizer_covers_both_training_stages(self):
        shakespeare_text = "To be, or not to be\n"
        tokenizer = build_shared_tokenizer(shakespeare_text)

        self.assertEqual(
            tokenizer.decode(tokenizer.encode(shakespeare_text)),
            shakespeare_text,
        )
        self.assertEqual(
            tokenizer.decode(tokenizer.encode(COUNTDOWN_CHARACTERS)),
            COUNTDOWN_CHARACTERS,
        )


if __name__ == "__main__":
    unittest.main()
