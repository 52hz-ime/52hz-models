import io
import json
import math
import unittest
from unittest.mock import patch

from preference import favour, preference


def answer(letter, share=0.99):
    """A chat completion in which the model answers `letter` with `share` of the probability between A and B."""
    if letter == "?":
        return packet([("?", 1.0)])
    weights = {"A": 1 - share, "B": share} if letter == "B" else {"A": share, "B": 1 - share}
    return packet([(" “A”", weights["A"]), ("B", weights["B"])])


def packet(top):
    return io.BytesIO(json.dumps({"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": token, "logprob": math.log(weight)} for token, weight in top]}]}}]}).encode())


def serving(reply):
    def server(request, timeout):
        return reply(json.loads(request.data))
    return server


class PreferenceTests(unittest.TestCase):
    def test_the_value_is_the_support_for_the_second_spelling(self):
        asked = []

        def server(request, timeout):
            asked.append(json.loads(request.data))
            return answer("A")
        with patch("preference.urlopen", side_effect=server):
            value = preference("http://localhost/", "上文", "kaihui", "开会", "开户", "明天去＿")
        self.assertAlmostEqual(value, 0.01, places=4)
        self.assertEqual(len(asked), 1)
        self.assertEqual(asked[0]["messages"][-1]["content"].splitlines(),
                         ["上文：上文", "句子：明天去＿", "用户在这一处打的拼音：kaihui", "填进去更通顺、更符合上文的是 A「开会」还是 B「开户」？"])
        self.assertEqual((asked[0]["temperature"], asked[0]["max_tokens"], asked[0]["logprobs"]), (0, 1, True))
        with patch("preference.urlopen", side_effect=serving(lambda payload: answer("B"))):
            self.assertAlmostEqual(preference("http://localhost", "", "kaihui", "开会", "开户"), 0.99, places=4)
        # An answer that names neither spelling says nothing.
        with patch("preference.urlopen", side_effect=serving(lambda payload: answer("?"))):
            self.assertEqual(preference("http://localhost", "", "kaihui", "开会", "开户"), 0.5)

    def test_favour_asks_both_ways_round(self):
        # A model that always answers A has no preference between the spellings, whichever is expected.
        with patch("preference.urlopen", side_effect=serving(lambda payload: answer("A"))):
            self.assertAlmostEqual(favour("http://localhost", "", "kaihui", "开会", "开户"), 0.5, places=4)
        # A model that prefers 开会 says so with either spelling in front.
        def prefers(request):
            content = request["messages"][-1]["content"]
            return answer("A" if "A「开会」" in content else "B")
        with patch("preference.urlopen", side_effect=serving(prefers)):
            self.assertAlmostEqual(favour("http://localhost", "", "kaihui", "开会", "开户"), 0.99, places=4)
            self.assertAlmostEqual(favour("http://localhost", "", "kaihui", "开户", "开会"), 0.01, places=4)


if __name__ == "__main__":
    unittest.main()
