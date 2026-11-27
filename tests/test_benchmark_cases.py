import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lexicon import Pairs
from tests.benchmark_cases import character_errors, differing, errors_in, label, length_bucket, requests, score


class BenchmarkCaseTests(unittest.TestCase):
    def cases(self):
        return [{"id": "a:1~clean", "source_id": "a", "keys": "nihaoma", "text": "你好吗", "noise_ops": []},
                {"id": "a:1~drop", "source_id": "a", "keys": "nihoma", "text": "你好吗", "noise_ops": [{"type": "drop"}]},
                {"id": "a:1~x2", "source_id": "a", "keys": "nihaon", "text": "你好吗", "noise_ops": [{"type": "swap"}, {"type": "drop"}]},
                {"id": "b:2~clean", "source_id": "b", "keys": "mingtianqugongzuolangshangban", "context": "昨天",
                 "text": "明天去公司路上顺便买杯咖啡喝", "noise_ops": []}]

    def replies(self):
        return [{"id": "a:1~clean", "action": "commit", "candidates": ["你好吗"], "search": {"pool": ["你好吗"]}},
                {"id": "a:1~drop", "action": "commit", "candidates": ["尼好吗"], "search": {"pool": ["你好吗", "尼好吗"]}},
                {"id": "a:1~x2", "action": "ask", "focus": "nihaon", "options": ["你好嗯", "你好"], "search": {"pool": ["x"]}},
                {"id": "b:2~clean", "error": "首选与按键的读音相差太远", "search": {"pool": []}}]

    def test_requests_hide_the_answers(self):
        exported = list(requests(self.write(self.cases())))
        self.assertEqual(exported[1], {"id": "a:1~drop", "keys": "nihoma", "context": "", "memory": [], "selected": {}, "event": "space"})

    def test_labels_and_error_counts(self):
        self.assertEqual([label(row) for row in self.cases()], ["clean", "drop", "x2", "clean"])
        self.assertEqual([errors_in(row) for row in self.cases()], [0, 1, 2, 0])
        self.assertEqual([length_bucket(size) for size in (4, 9, 15, 21, 44)], ["<=8", "9-14", "15-20", "21-30", ">30"])

    def test_score_separates_one_shot_success_wrong_commit_ask_and_error(self):
        report = score(self.write(self.cases()), self.write(self.replies()))
        self.assertEqual((report["total"], report["passed"], report["failed"]), (4, 1, 3))
        self.assertEqual((report["wrong_commits"], report["asks"], report["errors"]), (1, 1, 1))
        self.assertIsNone(report["homophone_passing"])
        self.assertIsNone(report["tolerated"])
        self.assertEqual(report["in_pool"], {"total": 4, "passed": 2, "accuracy": 0.5})
        # Two three-character commits, one wrong character in the second.
        self.assertAlmostEqual(report["char_error_rate"], 1 / 6)
        self.assertEqual(report["by_operation"]["drop"], {"total": 1, "passed": 0, "accuracy": 0.0, "wrong_commit": 1, "ask": 0,
                                                         "error": 0, "homophone_passing": 0, "tolerated": None})
        self.assertEqual(report["by_errors"]["2"]["ask"], 1)
        self.assertEqual(report["by_errors"]["0"]["error"], 1)
        self.assertEqual(report["by_length"]["<=8"]["total"], 3)
        self.assertEqual(report["by_length"]["9-14"]["total"], 1)
        self.assertEqual(report["by_source"]["b"]["error"], 1)

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
        self.assertEqual(differing("在这种年终关键时候，", "在这种年中关键时候，"),
                         [("终", "中", "在这种年＿关键时候，")])

    def lexicon(self, root):
        path = root / "lexicon.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in (
            {"word": "年", "count": 900}, {"word": "终", "count": 300}, {"word": "中", "count": 800},
            {"word": "年中", "keys": "nianzhong", "count": 50}, {"word": "关键", "keys": "guanjian", "count": 500},
            {"word": "时候", "keys": "shihou", "count": 700}, {"word": "制片", "keys": "zhipian", "count": 90})) + "\n", encoding="utf-8")
        return path

    def homophones(self):
        return [{"id": "h:1", "source_id": "a", "keys": "zaizhezhongnianzhongguanjianshihou，", "context": "我记得，是在1988年。",
                 "text": "在这种年终关键时候，", "noise_ops": [{"type": "nasal"}]},
                {"id": "h:2", "source_id": "a", "keys": "yecebuzhidaozhipiandangchuxinshishidandanshuoyuanban，", "context": "",
                 "text": "也不知道制片当初信誓旦旦说原班，", "noise_ops": [{"type": "swap"}]},
                {"id": "h:3", "source_id": "a", "keys": "lomaobameixifangzaiqianyaodeweizhishang，", "context": "他只是踢得更聪明了。",
                 "text": "老马把梅西放在前腰的位置上，", "noise_ops": [{"type": "neighbor"}]}]

    def homophone_replies(self):
        return [{"id": "h:1", "action": "commit", "candidates": ["在这种年中关键时候，"]},
                {"id": "h:2", "action": "commit", "candidates": ["也不知道只片当初新事但但说原班，"]},
                {"id": "h:3", "action": "commit", "candidates": ["老马把梅西放在钱要的位置上，"]}]

    def test_a_homophone_counts_when_it_is_real_words_and_is_tolerated_when_the_context_cannot_decide(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, replies, pairs = self.write(self.homophones(), root), self.write(self.homophone_replies(), root), Pairs(self.lexicon(root))
            undecidable = dict.fromkeys(range(3), 0.5)
            with patch("tests.benchmark_cases.favour", side_effect=lambda *args: undecidable.pop(0)):
                report = score(cases, replies, pairs=pairs, judge="http://localhost")
            # 年中/年终 with a weak context is tolerated. 只片 and 新事但但 are not words, and 钱要 is not a known
            # pair either, so those two replies never reach the homophone count.
            self.assertEqual((report["passed"], report["homophone_passing"], report["tolerated"]), (0, 1, 1))
            self.assertEqual(report["by_operation"]["nasal"]["tolerated"], 1)
            self.assertEqual(report["by_operation"]["neighbor"]["homophone_passing"], 0)
            # A tolerated reply is still one case: group totals stay at the number of cases.
            self.assertEqual(report["by_operation"]["nasal"], {"total": 1, "passed": 0, "accuracy": 0.0, "wrong_commit": 1,
                                                              "ask": 0, "error": 0, "homophone_passing": 1, "tolerated": 1})
            self.assertEqual(sum(group["total"] for group in report["by_source"].values()), 3)
            decided = dict.fromkeys(range(3), 0.9)
            with patch("tests.benchmark_cases.favour", side_effect=lambda *args: decided.pop(0)):
                report = score(cases, replies, pairs=pairs, judge="http://localhost")
            self.assertEqual((report["homophone_passing"], report["tolerated"]), (1, 0))
            # Without a judge the words are counted as homophones but nothing is marked tolerated.
            report = score(cases, replies, pairs=pairs)
            self.assertEqual((report["homophone_passing"], report["tolerated"]), (1, None))

    def test_the_judgement_is_about_the_expected_spelling_whichever_way_round_it_is_asked(self):
        """Goes through the real question to the model: a fake server that always picks the spelling of one sentence."""
        def server(pick_from):
            def answer(request, timeout):
                asked = re.search(r"A「(.+?)」还是 B「(.+?)」", json.loads(request.data)["messages"][-1]["content"])
                strong, weak = (-0.01, -5.0) if asked.group(1) in pick_from else (-5.0, -0.01)
                return io.BytesIO(json.dumps({"choices": [{"logprobs": {"content": [{"top_logprobs": [
                    {"token": "A", "logprob": strong}, {"token": "B", "logprob": weak}]}]}}]}).encode())
            return answer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, replies, pairs = self.write(self.homophones(), root), self.write(self.homophone_replies(), root), Pairs(self.lexicon(root))
            # The model clearly prefers 年终, the expected spelling: the context did decide, so 年中 is not tolerated.
            with patch("preference.urlopen", side_effect=server(self.homophones()[0]["text"])):
                report = score(cases, replies, pairs=pairs, judge="http://localhost")
            self.assertEqual((report["homophone_passing"], report["tolerated"]), (1, 0))
            # The model clearly prefers 年中, the other spelling: the expected one cannot be held against the decoder.
            with patch("preference.urlopen", side_effect=server(self.homophone_replies()[0]["candidates"][0])):
                report = score(cases, replies, pairs=pairs, judge="http://localhost")
            self.assertEqual((report["homophone_passing"], report["tolerated"]), (1, 1))

    def test_differing_ignores_pieces_that_were_dropped_or_added(self):
        self.assertEqual(differing("只要他在今天拿到15分，", "只要他今天拿到15分，"), [])
        self.assertEqual(differing("我喜欢吃饭", "我喜欢吃面"), [("饭", "面", "我喜欢吃＿")])

    def write(self, rows, root=None):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False, dir=root)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
