import re
import unittest

from experiments.constrained_decoding import grammar
from experiments.draw_clean import bucket, draw
from experiments.pinyin_conversion import prompt, syllables
from pinyin import to_keys


class ExperimentTests(unittest.TestCase):
    def test_the_joined_form_is_the_keys_and_anything_not_chinese_stays_as_typed(self):
        # The first measurement on clean-1000 was void because punctuation, digits and letters were dropped here.
        for text in ("不可以预约摄影师。", "想去青岛生活做HR估计要45岁以后的计划了", "分别被判处有期徒刑4年至10年不等，", "我要买一个iPhone15", "你好，世界！"):
            with self.subTest(text=text):
                self.assertEqual(syllables(text, "joined"), to_keys(text))
        self.assertEqual(syllables("想去青岛生活做HR估计要45岁以后的计划了", "spaced"),
                         "xiang qu qing dao sheng huo zuo HR gu ji yao 45 sui yi hou de ji hua le")
        self.assertEqual(syllables("你好，世界！", "toned"), "nǐ hǎo ， shì jiè ！")
        self.assertEqual(syllables("你好", "numbered"), "ni3 hao3")

    def test_the_prompt_shows_the_form_the_variant_names_and_what_it_adds(self):
        row = {"keys": "nihaoma", "text": "你好吗", "context": "上次聊到吃饭。"}
        bare = prompt("spaced", row)
        self.assertEqual([message["role"] for message in bare], ["system", "user"])
        self.assertEqual(bare[-1]["content"], "ni hao ma")
        self.assertEqual(prompt("spaced +context", row)[-1]["content"], "上文：上次聊到吃饭。\nni hao ma")
        self.assertEqual(prompt("spaced +strict", row)[0]["content"], bare[0]["content"] + "输出的每个字都要对得上拼音的读音，不增字不删字，不改写、不概括。")
        self.assertEqual(len(prompt("spaced +examples", row)), 2 + 2 * 13)

    def test_the_grammar_admits_one_character_per_character_of_the_text(self):
        rules = grammar("200学完后方可毕业。").split("\n")
        self.assertEqual(rules[0], 'root ::= "2" "0" "0" r3 r4 r5 r6 r7 r8 r9 "。"')
        self.assertEqual([line.split(" ::= ")[0] for line in rules[1:]], [f"r{index}" for index in range(3, 10)])
        self.assertIn('"学"', rules[1])            # xue
        self.assertNotIn('"校"', rules[1])         # xiao

    def test_the_lattice_grammar_is_built_on_the_keys_and_never_admits_them_back_as_a_word(self):
        import experiments.lattice_decoding as lattice_decoding
        grammar = lattice_decoding.lattice('womenmingtian9diankaizongjiehui')
        nodes = [line for line in grammar.split('\n') if line.startswith('n')]
        # Every position leads on to the positions after what can be written there; the model picks the boundaries.
        self.assertTrue('n0 ::= rwo n2' in nodes and 'n2 ::= rm n3 | rme n4 | rmen n5' in nodes)   # the model picks the boundaries
        self.assertTrue('n13 ::= "9" n14' in nodes)                      # a digit run is passed on as typed
        literal = [piece for piece in grammar.split('"')[1::2]]
        self.assertFalse(any('dian' in piece or 'kai' in piece for piece in literal))    # pinyin letters are never a word
        self.assertTrue('rdian ::= "' in grammar and '"点"' in grammar.split('rdian ::= ')[1].split('\n')[0])   # dian is a rule, 点 one of its characters
        # A run of pinyin letters that is not pinyin at all has no walk; the request then goes without a grammar.
        self.assertIsNone(lattice_decoding.lattice('hellox'))

    def test_with_one_keystroke_a_dropped_letter_is_still_a_step(self):
        import experiments.lattice_decoding as lattice_decoding
        strict = lattice_decoding.lattice('kelashi')
        loose = lattice_decoding.lattice('kelashi', one_keystroke=True)
        self.assertNotIn('rlao', strict)
        self.assertIn('rlao n5', loose)                               # lao written over the letters la, then on to position 5
        self.assertRegex(next(line for line in loose.split('\n') if line.startswith('n5 ::=')), r'(?::=| \| )rshi(?= \||$)')   # shi over shi reaches the end
        self.assertIn('rke n2', next(line for line in loose.split('\n') if line.startswith('n0 ::=')))          # and ke over ke
        self.assertTrue(all('_' not in name for name in re.findall(r'^(\S+) ::=', loose, re.M)))     # llama.cpp takes no underscores in rule names

    def test_the_clean_set_is_drawn_the_same_way_twice_and_covers_every_cell(self):
        cases = [{"id": f"{source}:{number}", "source_id": source, "text": "字" * size, "noise_ops": []}
                 for source in "ab" for size in (5, 12, 18, 25) for number in range(10)]
        first, second = draw(cases, 24, __import__("random").Random(3)), draw(cases, 24, __import__("random").Random(3))
        self.assertEqual(first, second)
        self.assertEqual(sorted({(row["source_id"], bucket(len(row["text"]))) for row in first}),
                         sorted((source, size) for source in "ab" for size in ("4-8", "9-14", "15-20", "21-30")))


if __name__ == "__main__":
    unittest.main()
