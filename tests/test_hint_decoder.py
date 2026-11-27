import io
import json
import os
import unittest
from unittest.mock import patch

from infer import HINT_EXAMPLES, REPAIR_SYSTEM, predict, problems, tidied, unfinished
from verify import align


def envelope(text):
    return json.dumps({'choices': [{'message': {'content': text}}]}).encode()


class HintDecoderTests(unittest.TestCase):
    def requests(self, post):
        return [json.loads(call.args[0].data) for call in post.call_args_list]

    def user_text(self, payload):
        return payload['messages'][-1]['content']

    def test_typing_previews_with_one_call_and_carries_context_memory_and_selections(self):
        row = {'id': 'case', 'event': 'type', 'keys': 'jintianwanshangchifann', 'context': '上次聊到吃饭。',
               'memory': ['他不吃辣。'], 'selected': {'chifan': '吃饭'}}
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope('今天晚上吃饭。'))) as post:
            reply = predict('http://localhost', row)
        first, = self.requests(post)
        self.assertEqual(reply, {'action': 'preview', 'candidates': ['今天晚上吃饭。'], 'id': 'case',
                                 'verify': {'cost': 1, 'budget': 2, 'doubts': 0, 'flagged': 1, 'skipped': 0},
                                 'repair': {'rounds': 0, 'before': 1, 'after': 1, 'first': '今天晚上吃饭。'}})
        self.assertEqual(self.user_text(first),
                         '上文：上次聊到吃饭。\n记忆：他不吃辣。\n已选定：吃饭（按键 chifan，不可改动）\n'
                         '按键：jintianwanshangchifann\n切分：jin tian wan shang chi fan?')
        self.assertEqual(len(first['messages']), 1 + 2 * len(HINT_EXAMPLES) + 1)
        self.assertTrue(all(message['content'] for message in first['messages'][1:-1:2]))

    def test_every_example_passes_the_verifier(self):
        # An example shows what a good reply is; it must not teach a rewrite that the verifier would keep back.
        for example in HINT_EXAMPLES:
            with self.subTest(keys=example['keys']):
                report = align(example['keys'], example['text'])
                self.assertTrue(report['consistent'])
                self.assertLessEqual(report['cost'], 2)       # the errors the example was given, and no more

    def test_space_commits_a_reply_that_reads_like_the_keys_with_one_call(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'kelashi', 'context': '课还没开始。'}
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope('柯老师'))) as post:
            reply = predict('http://localhost', row)
        self.assertEqual(reply, {'action': 'commit', 'candidates': ['柯老师'], 'id': 'case',
                                 'verify': {'cost': 1, 'budget': 2, 'doubts': 0, 'flagged': 1, 'skipped': 0},
                                 'repair': {'rounds': 0, 'before': 1, 'after': 1, 'first': '柯老师'}})
        self.assertNotIn('response_format', self.requests(post)[0])

    def test_a_reply_that_leaves_the_budget_is_repaired_and_the_repair_is_kept_when_it_fits_better(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihuiquyinhang', 'context': ''}
        # 「银钱」 does not read like the keys; 「印航」 would, since it sounds the same as 银行.
        bad, good = '明天开会去银钱', '明天开会去银行'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(bad)), io.BytesIO(envelope(good))]) as post:
            reply = predict('http://localhost', row)
        requests = self.requests(post)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]['messages'][0]['content'], REPAIR_SYSTEM)
        self.assertEqual(self.user_text(requests[1]).splitlines()[:4],
                         ['按键：mingtiankaihuiquyinhang', '切分：ming tian kai hui qu yin hang', f'转写：{bad}', '对不上：'])
        self.assertTrue(self.user_text(requests[1]).splitlines()[4].startswith('第7个字「钱」读作 qian，按键这一处是 hang'))
        self.assertEqual(reply['candidates'], [good])
        self.assertEqual(reply['verify']['cost'], 0)

    def test_a_rewrite_that_invents_words_is_discarded_even_though_it_fits_the_keys(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihuiquyinhang', 'context': ''}
        bad, nonsense = '明天开会去银钱', '明天开会去银航'
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.PAIRS') as pairs, \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(bad)), io.BytesIO(envelope(nonsense))]):
            pairs.loaded = True
            pairs.novel.side_effect = lambda text, earlier: ['银航'] if text == nonsense else []
            reply = predict('http://localhost', row)
        # The rewrite fits the keys better but is not words, so nothing changed and the loop ends there.
        self.assertIn('error', reply)
        self.assertEqual(reply['rejected'], bad)
        self.assertEqual(reply['repair'], {'rounds': 1, 'before': 3, 'after': 3, 'first': bad})

    def test_when_no_position_can_be_named_the_model_is_told_the_cost_and_only_the_real_overrun(self):
        # Three stray letters, each on its own, name no position; only a real overrun is called one.
        self.assertEqual(problems(align('jintianbwanxshangchifanz', '今天晚上吃饭')),
                         ['整句要 3 个错误才能解释按键，超过了 2 的额度；请重新考虑各字的读法'])
        self.assertEqual(problems(align('jintianbwanxshangchifan', '今天晚上吃饭')), ['整句要 2 个错误才能解释按键；请重新考虑各字的读法'])

    def test_the_repair_is_told_where_the_reply_left_something_out(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'woxiangchifanhaone', 'context': ''}
        short, full = '我想吃', '我想吃饭好呢'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(short)), io.BytesIO(envelope(full))]) as post:
            reply = predict('http://localhost', row)
        self.assertEqual(self.user_text(self.requests(post)[1]).splitlines(),
                         ['按键：woxiangchifanhaone', '切分：wo xiang chi fan hao ne', f'转写：{short}', '对不上：',
                          '句末有 8 个字母没有对应上：fanhaone，可以读作 fan hao ne'])
        self.assertEqual((reply['candidates'], reply['verify']['cost']), ([full], 0))
        # Something left out at the very start is said to be at the start, not before a "zeroth" character.
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope('吃饭')), io.BytesIO(envelope('今天吃饭'))]) as post:
            reply = predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'jintianchifan', 'context': ''})
        self.assertEqual(self.user_text(self.requests(post)[1]).splitlines()[-1], '句首有 7 个字母没有对应上：jintian，可以读作 jin tian')
        self.assertEqual(reply['candidates'], ['今天吃饭'])

    def test_a_reply_within_its_budget_is_looked_at_again_when_a_character_is_far_from_the_keys(self):
        # Twelve characters allow four errors, so 用, three keystrokes from yi, does not bust the budget. It is
        # still a reading the verifier has evidence against, so the reply is looked at again all the same.
        row = {'id': 'case', 'event': 'space', 'keys': 'jintianwanshangchiyidunhenhaochidefan', 'context': ''}
        bad, good = '今天晚上吃用顿很好吃的饭', '今天晚上吃一顿很好吃的饭'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(bad)), io.BytesIO(envelope(good))]) as post:
            reply = predict('http://localhost', row)
        self.assertEqual((reply['verify'], len(self.requests(post)), reply['candidates'], reply['repair']['rounds']),
                         ({'cost': 0, 'budget': 4, 'doubts': 0, 'flagged': 0, 'skipped': 0}, 2, [good], 1))

    def test_readings_one_keystroke_away_and_a_stray_letter_are_left_alone(self):
        # Two keystrokes too many, within the budget, and no reading far from the keys: nothing points at a
        # mistake, so the reply is committed without a second call.
        row = {'id': 'case', 'event': 'space', 'keys': 'jinntianwanshangchifanb', 'context': ''}
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', return_value=io.BytesIO(envelope('今天晚上吃饭'))) as post:
            reply = predict('http://localhost', row)
        self.assertEqual((reply['action'], reply['repair']['rounds'], reply['verify']['cost'], reply['verify']['budget'], len(self.requests(post))),
                         ('commit', 0, 2, 2, 1))

    def test_an_over_budget_reply_is_looked_at_once_even_without_evidence(self):
        # Over the budget the loop runs whatever the verifier can point at; a round that changes nothing ends it.
        row = {'id': 'case', 'event': 'space', 'keys': 'jinntianwanshangchifanbb', 'context': ''}
        same = '今天晚上吃饭'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(same)), io.BytesIO(envelope(same))]) as post:
            reply = predict('http://localhost', row)
        self.assertEqual((reply['rejected'], reply['repair']['rounds'], len(self.requests(post))), (same, 1, 2))

    def test_a_rewrite_may_lower_its_letters_but_may_not_turn_characters_into_pinyin(self):
        # HTML was typed in lower case: writing it as typed makes the reply read like the keys.
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope('HTML功能')), io.BytesIO(envelope('html功能'))]) as post:
            reply = predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'htmlgongneng', 'context': ''})
        self.assertEqual((reply['candidates'], len(self.requests(post))), (['html功能'], 2))
        # 国 written as its keys instead of as a character matches them perfectly, and is not a conversion.
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope('中国国')), io.BytesIO(envelope('中国guo'))]) as post:
            reply = predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'zhongguo', 'context': ''})
        self.assertEqual((reply.get('rejected'), len(self.requests(post))), ('中国国', 2))

    def test_a_rewrite_that_brings_in_letters_is_discarded_even_though_it_matches_the_keys(self):
        # Echoing the keys matches them perfectly, since letters match themselves.
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihuiquyinhang', 'context': ''}
        bad = '明天开会去银钱'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(bad)), io.BytesIO(envelope(row['keys']))]) as post:
            reply = predict('http://localhost', row)
        self.assertEqual((reply['rejected'], reply['repair']['rounds'], len(self.requests(post))), (bad, 1, 2))

    def test_a_comma_nobody_typed_is_not_evidence_of_a_misreading(self):
        report = align('jintianwanshangchifan', '今天，晚上吃饭')
        self.assertEqual((report['cost'], report['doubts'], unfinished(report)), (1, [2], False))

    def test_typed_punctuation_is_kept_as_typed_and_no_other_is_added(self):
        keys = "jintian，wanshangchifan?"
        self.assertEqual(tidied({'keys': keys}, "今天晚上吃饭")[0], "今天，晚上吃饭?")          # the typed ， and ? come back as typed
        self.assertEqual(tidied({'keys': keys}, "今天，晚上吃饭?")[0], "今天，晚上吃饭?")
        self.assertEqual(tidied({'keys': keys}, "今天，晚上，吃饭？")[0], "今天，晚上吃饭?")    # an added comma goes, ？ becomes ?
        self.assertEqual(tidied({'keys': 'jintianwanshangchifan'}, "今天，晚上吃饭。")[0], "今天晚上吃饭")
        self.assertEqual(tidied({'keys': 'jintianwanshangchifan'}, "今天晚上吃范")[0], "今天晚上吃范")       # characters are not touched
        self.assertEqual(tidied({'keys': 'woxiangchifanhaone'}, "我想吃")[0], "我想吃")          # unclaimed pinyin is not pasted in

    def test_a_homophone_is_never_sent_for_repair_even_when_it_is_the_wrong_word(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihuiquyinhang', 'context': '去取钱。'}
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope('明天开会去印航'))) as post:
            reply = predict('http://localhost', row)
        self.assertEqual((reply['action'], len(self.requests(post))), ('commit', 1))

    def test_an_unexplained_reply_is_kept_back_after_the_repair_fails_to_help(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'nikanjiushiwangji，', 'context': ''}
        junk = '你简直就是王八犊子，'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(junk)), io.BytesIO(envelope(junk)),
                                                    io.BytesIO(envelope('你看看就是旺季，'))]) as post:
            reply = predict('http://localhost', row)
        # The rewrite repeats the reply; the same request would only get the same answer, so the loop ends there.
        self.assertIn('error', reply)
        self.assertNotIn('candidates', reply)
        self.assertEqual(len(self.requests(post)), 2)

    def test_a_repair_that_does_not_fit_better_is_discarded(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihuiquyinhang', 'context': ''}
        bad, worse = '明天开会去银钱', '明天开回去银钱'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(bad)), io.BytesIO(envelope(worse))]) as post:
            reply = predict('http://localhost', row)
        self.assertIn('error', reply)
        self.assertEqual(reply['rejected'], bad)
        self.assertEqual(len(self.requests(post)), 2)

    def test_a_dropped_selection_falls_back_to_the_json_contract(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihu', 'context': '银行已经预约好了。', 'selected': {'kaihu': '开会'}}
        answer = '{"action":"commit","candidates":["明天开会"]}'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope('明天开户')), io.BytesIO(envelope(answer))]) as post:
            reply = predict('http://localhost', row)
        self.assertEqual(reply['candidates'], ['明天开会'])
        fallback, = self.requests(post)[1:]
        self.assertEqual(fallback['response_format']['json_schema']['name'], 'ime_decision')

    def test_asking_on_demand_keeps_the_json_contract(self):
        row = {'id': 'case', 'event': 'more', 'keys': 'kaihu', 'focus': 'kaihu'}
        answer = '{"action":"ask","focus":"kaihu","options":["开会","开户"]}'
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope(answer))) as post:
            reply = predict('http://localhost', row)
        payload, = self.requests(post)
        self.assertEqual(reply['action'], 'ask')
        self.assertEqual(payload['response_format']['json_schema']['name'], 'ime_decision')

    def test_empty_reply_is_reported_instead_of_committed(self):
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope('  '))):
            reply = predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'nihaoma', 'context': ''})
        self.assertIn('error', reply)
        self.assertNotIn('candidates', reply)

    def test_hint_is_the_default_when_no_decoder_is_named(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'jintianwanshangchifan', 'context': ''}
        environment = {key: value for key, value in os.environ.items() if key != 'IME_DECODER'}
        with patch.dict(os.environ, environment, clear=True), patch('infer.urlopen', return_value=io.BytesIO(envelope('今天晚上吃饭。'))) as post:
            reply = predict('http://localhost', row)
        self.assertEqual((reply['action'], reply['candidates']), ('commit', ['今天晚上吃饭']))     # no period was typed
        self.assertNotIn('response_format', self.requests(post)[0])     # one plain chat call, not the two-stage search

    def test_unknown_decoder_is_rejected(self):
        with patch.dict(os.environ, IME_DECODER='other'):
            with self.assertRaisesRegex(ValueError, 'IME_DECODER'):
                predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'ni', 'context': ''})
