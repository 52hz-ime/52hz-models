import unittest

from fidelity import distance, reliable
from pinyin import to_keys


class FidelityTests(unittest.TestCase):
    def test_text_returns_to_the_keys_it_was_typed_with(self):
        self.assertEqual(to_keys('明天开户'), 'mingtiankaihu')
        self.assertEqual(to_keys('只要他在今天拿到15分，就可以超越卡尔-马龙(4761分)。'),
                         'zhiyaotazaijintiannadao15fen，jiukeyichaoyuekaer-malong(4761fen)。')
        self.assertEqual(to_keys('我用iPhone15'), 'woyongiPhone15')

    def test_distance_counts_each_edit_once(self):
        self.assertEqual(distance('kaihu', 'kaihui'), 1)
        self.assertEqual(distance('abc', 'abc'), 0)

    def test_related_readings_are_shown_and_only_unrelated_output_is_kept_back(self):
        self.assertTrue(reliable('kelashi', '柯老师'))                                 # a letter was dropped
        self.assertTrue(reliable('mingtiankaihui', '明天开会'))                         # typed exactly
        self.assertTrue(reliable('duibutongxuixiangqubieduidai。', '对不同需求区别对待。'))  # a wrong word, but a reading
        self.assertFalse(reliable('nikanjiushiwangji，', '你简直就是王八犊子，'))
        self.assertFalse(reliable('daoyanshikelinte·iysitewude。', '导演是克里斯托弗·诺兰。'))
        self.assertFalse(reliable('luosanshiiyoumi，', '落三石幽篁'))

    def test_the_limit_grows_with_the_length_of_the_keys(self):
        self.assertFalse(reliable('nihao', '你简直就是'))                               # 5 edits in 5 keys
        self.assertTrue(reliable('nihao' + 'ma' * 40, '你好' + '吗' * 39 + '嘛'))         # 3 edits in 85 keys


if __name__ == '__main__':
    unittest.main()
