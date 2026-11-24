import unittest

from ime_cases import cases, matches, noise_variants, score


class ImeRegressionTests(unittest.TestCase):
    def test_fixture_contract(self):
        seen = set()
        for case in cases():
            with self.subTest(case=case["id"]):
                self.assertNotIn(case["id"], seen)
                seen.add(case["id"])
                request = case["request"]
                expected = case["expected"]
                self.assertNotIn("expected", request)
                self.assertIn(request["event"], ("type", "more", "space", "select"))
                self.assertIn(expected["action"], ("preview", "commit", "ask"))
                if expected["action"] == "ask":
                    self.assertIn(expected["focus"], request["keys"])
                    self.assertGreaterEqual(len(set(expected["options"])), 2)
                else:
                    self.assertTrue(expected["text"])
                    for choice in request.get("selected", {}).values():
                        self.assertIn(choice, expected["text"])

    def test_noise_replay(self):
        variants = noise_variants(("wan", "fan", "chi", "shen", "me"), seed=52)
        self.assertEqual(list(variants), ["wanfanchishenme", "wanfnchishenme", "wwanfanchishenme",
                                          "wanfanchishenmr", "wanfanchishenem", "wanfanshenchime"])

    def test_first_candidate_and_action_both_matter(self):
        expected = {"action": "commit", "text": "明天开会"}
        self.assertTrue(matches(expected, {"action": "commit", "candidates": ["明天开会", "明天开户"]}))
        self.assertFalse(matches(expected, {"action": "commit", "candidates": ["明天开户", "明天开会"]}))
        self.assertFalse(matches(expected, {"action": "preview", "candidates": ["明天开会"]}))

    def test_one_focus_with_expected_choices(self):
        expected = {"action": "ask", "focus": "kaihu", "options": ["开会", "开户"]}
        self.assertTrue(matches(expected, {**expected, "options": ["开户", "开会"]}))
        self.assertFalse(matches(expected, {**expected, "focus": "lixin"}))
        self.assertFalse(matches(expected, {**expected, "options": ["开会"]}))

    def test_scoring_and_incomplete_batches(self):
        replies = []
        for case in cases():
            expected = case["expected"]
            fields = expected if expected["action"] == "ask" else {
                "action": expected["action"], "candidates": [expected["text"]]}
            replies.append({"id": case["id"], **fields})
        self.assertEqual(score(reversed(replies))["passed"], len(replies))
        text_rows = [row for row in replies if 'candidates' in row]
        for row in text_rows:
            row['search'] = {'pool': row['candidates'], 'winner': row['candidates'][0]}
        text_rows[0]['search'] = {'pool': []}
        text_rows[1]['search']['winner'] = '别的候选'
        self.assertEqual(score(replies)['sentence_search'], {'target_sentences': len(text_rows),
                         'in_pool': len(text_rows) - 1, 'winner_correct': len(text_rows) - 2})
        ambiguous = next(reply for reply in replies if reply["action"] == "ask")
        ambiguous.update(action="commit", candidates=["明天开户"])
        report = score(replies)
        self.assertEqual(report["failed_ids"], [ambiguous["id"]])
        self.assertEqual(report["wrong_commit_ids"], [ambiguous["id"]])
        for invalid in (replies[:-1], [*replies, replies[0]], [*replies, {"id": "unknown"}]):
            with self.assertRaises(ValueError):
                score(invalid)
