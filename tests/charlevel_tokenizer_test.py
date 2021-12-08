import unittest

from rapidnlp_datasets.charlevel_tokenizer import BertCharLevelTokenizer


class CharLevelTokenizerTest(unittest.TestCase):
    """Char level tokenizer test"""

    def test_bert_char_level_tokenizer(self):
        print()
        tokenizer = BertCharLevelTokenizer.from_file("testdata/vocab.txt", lowercase=True)
        encoding = tokenizer.encode("我喜欢自然语言处理NLP! It's amazing!")
        # fmt: off
        self.assertEqual(encoding.tokens, ['[CLS]', '我', '喜', '欢', '自', '然', '语', '言', '处', '理', 'n', 'l', 'p', '!', 'i', 't', "'", 's', 'a', 'm', 'a', 'z', 'i', 'n', 'g', '!', '[SEP]'])
        # fmt: on

        encoding = tokenizer.encode("我喜欢自然语言处理NLP! ", "It's amazing!")
        # fmt: off
        self.assertEqual(encoding.tokens, ['[CLS]', '我', '喜', '欢', '自', '然', '语', '言', '处', '理', 'n', 'l', 'p', '!', '[SEP]', 'i', 't', "'", 's', 'a', 'm', 'a', 'z', 'i', 'n', 'g', '!', '[SEP]'])
        # fmt: on

        encodings = tokenizer.encode_batch(["我喜欢自然语言处理NLP! It's amazing!"] * 4, add_special_tokens=False)
        for encoding in encodings:
            # fmt: off
            self.assertEqual(encoding.tokens, ['我', '喜', '欢', '自', '然', '语', '言', '处', '理', 'n', 'l', 'p', '!', 'i', 't', "'", 's', 'a', 'm', 'a', 'z', 'i', 'n', 'g', '!'])
            # fmt: on


if __name__ == "__main__":
    unittest.main()
