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
        self.assertEqual(self.flagged("lauiyuantyu", "来源于"), (2, True, [("来", "laui")]))      # an extra u, and a stray t

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

    def test_letters_nobody_claims_keep_their_order_and_say_where_they_sit(self):
        # 未来 is left out: it shows as one stretch after the second character, and the neighbours stay clean.
        report = align("worenweiweinaibutieshibukebimiandehuati。", "我认为补贴是不可避免的话题。")
        self.assertEqual((report["cost"], report["skipped"], report["gaps"]),
                         (6, ["w", "e", "i", "n", "a", "i"], [{"before": 3, "letters": "weinai"}]))
        self.assertEqual([(unit["char"], unit["typed"]) for unit in report["units"][:4]], [("我", "wo"), ("认", "ren"), ("为", "wei"), ("补", "bu")])
        self.assertEqual(report["doubts"], [])
        # A stray letter is one keystroke too many, not a character to repair: it is not charged to 源 or 于.
        report = align("lauiyuantyu", "来源于")
        self.assertEqual([(gap["before"], gap["letters"]) for gap in report["gaps"]], [(2, "t")])
        self.assertEqual([unit["char"] for unit in report["units"] if unit["cost"]], ["来"])
        report = align("woxiangchifanhaone", "我想吃")
        self.assertEqual((report["cost"], report["gaps"]), (8, [{"before": 3, "letters": "fanhaone"}]))     # past the end
        report = align("woxiangyaochifan", "我想吃")                  # two words left out, on either side of 吃
        self.assertEqual([(gap["before"], gap["letters"]) for gap in report["gaps"]], [(2, "yao"), (3, "fan")])
        report = align("jintianwan", "今天")
        self.assertEqual([(gap["before"], gap["letters"]) for gap in report["gaps"]], [(2, "wan")])

    def test_the_character_costs_add_up_to_the_total(self):
        # One untyped Chinese character counts UNTYPED in the total, so it must read UNTYPED in its row too.
        for keys, text in [("jintianwanshangchifanhaone", "我想吃"), ("nihaoma", "你好吗!"), ("jintianwanshangchi", "今天晚上吃饭"),
                           ("woxiangchifan", "我想吃"), ("mingtian9dian，", "明天9点,")]:
            with self.subTest(keys=keys, text=text):
                report = align(keys, text)
                self.assertEqual(report["cost"], sum(unit["cost"] for unit in report["units"]) + len(report["skipped"]))
        self.assertEqual([unit["cost"] for unit in align("jintianwanshangchi", "今天晚上吃饭")["units"]], [0, 0, 0, 0, 0, 4])

    def test_a_character_is_far_from_the_keys_only_when_no_equally_cheap_alignment_reads_it_near(self):
        # men mistyped as meng and mai with its a dropped share one g. Charging the g to 买 makes 买 look two
        # keystrokes off; the alignment that spreads the cost over 们 and 买 is just as cheap and has no such character.
        report = align("womengmidebaodoushijishikjuaidehaoba", "我们买的包都是几十块的好吧")
        self.assertEqual((report["cost"], [report["units"][index]["char"] for index in report["doubts"]]), (3, []))
        # 用 for yi cannot be read near the keys whatever the alignment.
        report = align("jintianwanshangchiyidunhenhaochidefan", "今天晚上吃用顿很好吃的饭")
        self.assertEqual([report["units"][index]["char"] for index in report["doubts"]], ["用"])

    def test_a_less_common_reading_of_another_length_counts_as_that_reading(self):
        # 了得 is typed liao de. 了 read as liao is not a stray l before a syllable "iao".
        report = align("xiangdangliaode", "相当了得")
        self.assertEqual((report["cost"], [(unit["char"], unit["typed"]) for unit in report["units"][-2:]], report["gaps"]), (0, [("了", "liao"), ("得", "de")], []))

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
