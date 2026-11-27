import unittest

from pinyin import units_of
from syllables import alternatives
from verify import align, budget, distance, summary


class VerifyTests(unittest.TestCase):
    def flagged(self, keys, text):
        report = align(keys, text)
        return report["cost"], report["consistent"], [(unit["char"], unit["typed"]) for index, unit in enumerate(report["units"])
                                                       if index in report["flagged"]]

    def test_a_sentence_typed_exactly_costs_nothing(self):
        self.assertEqual(self.flagged("nihaoma", "你好吗"), (0, True, []))
        self.assertEqual(self.flagged("woyongiPhone15", "我用iPhone15"), (0, True, []))
        self.assertEqual(self.flagged("mingtian9dian，", "明天9点，"), (0, True, []))

    def test_each_keystroke_error_costs_one_and_names_its_character(self):
        self.assertEqual(self.flagged("kelashi", "柯老师"), (1, True, [("老", "la")]))          # a letter was dropped
        self.assertEqual(self.flagged("mingtina", "明天"), (1, True, [("天", "tina")]))         # transposed inside a syllable
        self.assertEqual(self.flagged("mingtiaqnu", "明天去"), (1, True, [("天", "tiaq")]))    # transposed across a boundary
        self.assertEqual(self.flagged("jintianfenhenda", "今天风很大"), (1, True, [("风", "fen")]))  # en/eng merged
        self.assertEqual(self.flagged("wosisicuanren", "我是四川人"), (2, True, [("是", "si"), ("川", "cuan")]))
        self.assertEqual(self.flagged("lauiyuantyu", "来源于"), (2, True, [("来", "laui"), ("于", "tyu")]))

    def test_a_less_common_reading_is_not_a_typo(self):
        self.assertEqual(self.flagged("kelashi", "柯拉什"), (0, True, []))

    def test_literals_are_checked_too(self):
        self.assertEqual(self.flagged("mingtian9dian，", "明天9点,"), (1, True, [(",", "，")]))
        self.assertEqual(align("nihaoma!", "你好吗")["cost"], 1)        # a typed letter nobody claims
        self.assertEqual(align("nihaoma", "你好吗!")["cost"], 1)        # a character that was never typed

    def test_the_budget_grows_with_the_sentence(self):
        self.assertEqual(budget(units_of("你好")), 2)
        self.assertEqual(budget(units_of("我们明天再讨论吧")), 2)
        self.assertEqual(budget(units_of("我明天要去上海看朋友")), 3)
        text = "我明天要去上海看朋友"
        three = "w" + "ming" + "tina" + "yao" + "qu" + "shagn" + "hai" + "kan" + "peng" + "you"
        four = three[:-3] + "yuo"
        self.assertEqual((align(three, text)["cost"], align(three, text)["consistent"]), (3, True))
        self.assertEqual((align(four, text)["cost"], align(four, text)["consistent"]), (4, False))

    def test_output_unrelated_to_the_keys_is_never_consistent(self):
        report = align("nikanjiushiwangji，", "你简直就是王八犊子，")
        self.assertFalse(report["consistent"])
        self.assertGreaterEqual(report["cost"], 3 * report["budget"] // 2 + 1)
        self.assertIn("简", [report["units"][index]["char"] for index in report["doubts"]])
        # A syllable of the keys that the text leaves out busts the budget on its own.
        report = align("woyaotianqushanghaikanpengyou", "我要去上海看朋友")
        self.assertFalse(report["consistent"])
        self.assertTrue(report["doubts"] or report["skipped"])

    def test_nothing_to_align(self):
        self.assertIsNone(align("nihaoma", ""))

    def test_distance_and_summary(self):
        self.assertEqual(distance("kaihu", "kaihui"), 1)
        self.assertEqual(distance("abc", "abc"), 0)
        self.assertEqual(summary(align("kelashi", "柯老师")), {"cost": 1, "budget": 2, "doubts": 0, "flagged": 1, "skipped": 0})

    def test_alternatives_are_real_syllables_near_the_typed_letters(self):
        self.assertIn("lao", alternatives("la", "la"))
        self.assertIn("feng", alternatives("fen", "fen"))
        self.assertIn("tian", alternatives("tina", ("ti", "tina")))
        self.assertNotIn("fen", alternatives("fen", ("feng", "fen")))
        self.assertEqual(alternatives("", "la"), [])
        self.assertEqual(alternatives("abcdefghijkl", "la"), [])


if __name__ == "__main__":
    unittest.main()
