import json
import tempfile
import unittest
from pathlib import Path

from tests.benchmark_cases import character_errors, length_bucket, requests, score


class BenchmarkCaseTests(unittest.TestCase):
    def cases(self):
        return [{"id": "a:1~clean", "source_id": "a", "keys": "nihaoma", "text": "你好吗", "noise_ops": []},
                {"id": "a:1~drop", "source_id": "a", "keys": "nihoma", "text": "你好吗", "noise_ops": [{"type": "drop"}]},
                {"id": "b:2~clean", "source_id": "b", "keys": "mingtianqugongzuolangshangban", "context": "昨天",
                 "text": "明天去公司路上顺便买杯咖啡喝", "noise_ops": []},
                {"id": "b:2~swap", "source_id": "b", "keys": "mingtain", "text": "明天去公司路上顺便买杯咖啡喝", "noise_ops": [{"type": "swap"}]}]

    def replies(self):
        return [{"id": "a:1~clean", "action": "commit", "candidates": ["你好吗"], "search": {"pool": ["你好吗"]}},
                {"id": "a:1~drop", "action": "commit", "candidates": ["尼好吗"], "search": {"pool": ["你好吗", "尼好吗"]}},
                {"id": "b:2~clean", "action": "ask", "focus": "gongsi", "options": ["公司", "공사"], "search": {"pool": ["x"]}},
                {"id": "b:2~swap", "error": "裁判未返回合法候选编号", "search": {"pool": []}}]

    def test_requests_hide_the_answers(self):
        exported = list(requests(self.write(self.cases())))
        self.assertEqual(exported[1], {"id": "a:1~drop", "keys": "nihoma", "context": "", "memory": [], "selected": {}, "event": "space"})

    def test_score_separates_one_shot_success_wrong_commit_ask_and_error(self):
        report = score(self.write(self.cases()), self.write(self.replies()))
        self.assertEqual((report["total"], report["passed"], report["failed"]), (4, 1, 3))
        self.assertEqual((report["wrong_commits"], report["asks"], report["errors"]), (1, 1, 1))
        self.assertEqual(report["in_pool"], {"total": 4, "passed": 2, "accuracy": 0.5})
        # Two three-character commits, one wrong character in the second.
        self.assertAlmostEqual(report["char_error_rate"], 1 / 6)
        self.assertEqual(report["by_operation"]["drop"], {"total": 1, "passed": 0, "accuracy": 0.0, "wrong_commit": 1, "ask": 0, "error": 0})
        self.assertEqual(report["by_length"]["<=8"]["total"], 2)
        self.assertEqual(report["by_length"]["9-14"]["total"], 2)
        self.assertEqual(report["by_source"]["b"]["ask"], 1)

    def test_score_refuses_incomplete_or_duplicate_replies(self):
        cases = self.write(self.cases())
        with self.assertRaisesRegex(ValueError, "missing"):
            score(cases, self.write(self.replies()[:3]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score(cases, self.write(self.replies()[:1] + self.replies()[:1]))

    def test_helpers(self):
        self.assertEqual(character_errors("明天开会", "明天开会"), 0)
        self.assertEqual(character_errors("明天开会", "明天开门"), 1)
        self.assertEqual(character_errors("明天开会", "明天开会啦"), 1)
        self.assertEqual(character_errors("开会", "明天开会"), 2)
        self.assertEqual([length_bucket(size) for size in (4, 9, 15, 21, 44)], ["<=8", "9-14", "15-20", "21-30", ">30"])

    def write(self, rows):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)
