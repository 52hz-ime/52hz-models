import unittest

from syllables import segment


class SyllableTests(unittest.TestCase):
    def test_clean_keys_are_cut_into_the_typed_syllables(self):
        self.assertEqual(segment("wozaihuochezhan"), "wo zai huo che zhan")
        self.assertEqual(segment("zhegewentiwomenmingtianzaitaolunba"), "zhe ge wen ti wo men ming tian zai tao lun ba")

    def test_literals_are_kept_and_only_lowercase_runs_are_cut(self):
        self.assertEqual(segment("mingtian9dian，"), "ming tian 9 dian ，")
        self.assertEqual(segment("woyongPDF"), "wo yong PDF")
        self.assertEqual(segment("PR-239"), "PR - 239")
        self.assertEqual(segment("woyaomaiyitaiMacBook，"), "wo yao mai yi tai Mac Book ，")
        self.assertEqual(segment("iPhone15"), "i? Phone 15")

    def test_one_keystroke_error_is_marked_and_corrected(self):
        self.assertEqual(segment("jintianwanshangchifann"), "jin tian wan shang chi fan?")
        self.assertEqual(segment("jintianwanshangcihfan"), "jin tian wan shang chi? fan")
        self.assertEqual(segment("jintianwanshangchifsn"), "jin tian wan shang chi fan?")
        self.assertEqual(segment("jintianwannshangchifan"), "jin tian wan? shang chi fan")
        self.assertEqual(segment("taikaoyanchushidegongl！"), "tai kao yan chu shi de gong? ！")

    def test_an_extra_letter_joins_the_neighbouring_syllable_instead_of_stranding_a_consonant(self):
        self.assertEqual(segment("zhidsaoma"), "zhi dao? ma")

    def test_a_doubled_letter_is_attached_to_the_word_it_belongs_to(self):
        self.assertEqual(segment("hengaanren，"), "hen gan? ren ，")
        self.assertEqual(segment("nicaiyyicaima"), "ni cai? yi cai ma")
        self.assertEqual(segment("xianzaainijiutingchoude"), "xian zai? ni jiu ting chou de")

    def test_runs_that_are_not_pinyin_stay_whole(self):
        self.assertEqual(segment("excel"), "excel?")
        self.assertEqual(segment("ok"), "ok?")
        self.assertEqual(segment(""), "")

    def test_capitalised_pinyin_is_cut_but_capitalised_words_stay_whole(self):
        self.assertEqual(segment("nihaoshijie"), "ni hao shi jie")
        self.assertEqual(segment("Nihaoshijie"), "ni hao shi jie")
        self.assertEqual(segment("woyaomaiAtlas2.1b"), "wo yao mai Atlas 2 . 1 b?")
