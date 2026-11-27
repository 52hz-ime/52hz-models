import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lexicon import Pairs


class LexiconTests(unittest.TestCase):
    def test_pairs_come_from_the_corpus_list_and_single_characters_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexicon.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (
                {"word": "明", "count": 900}, {"word": "天", "count": 800}, {"word": "明天", "keys": "mingtian", "count": 500},
                {"word": "天开", "keys": "tiankai", "count": 50}, {"word": "银航", "keys": "yinhang", "count": 9},
                {"word": "开会", "keys": "kaihui", "count": 700})) + "\n",
                encoding="utf-8")
            pairs = Pairs(path)
        self.assertTrue(pairs.loaded)
        self.assertEqual(pairs.novel("明天开会", "明天开会"), [])
        self.assertEqual(pairs.novel("明天开会", "明天"), [])            # 开会 is known, and so is the boundary 天开
        self.assertEqual(pairs.novel("明天天开会", "明天"), ["天天"])     # never seen
        self.assertEqual(pairs.novel("明天银航", "明天"), ["天银", "银航"])  # 银航 was seen, but too rarely
        self.assertEqual(pairs.novel("明天，开会。", "明天"), [])         # punctuation is not part of any pair

    def test_without_a_list_nothing_is_known(self):
        with patch.dict(os.environ, IME_LEXICON=''):
            self.assertFalse(Pairs().loaded)
        self.assertFalse(Pairs('/nonexistent/lexicon.jsonl').loaded)


if __name__ == "__main__":
    unittest.main()
