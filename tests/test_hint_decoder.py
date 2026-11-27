import io
import json
import os
import unittest
from unittest.mock import patch

from infer import HINT_EXAMPLES, REPAIR_SYSTEM, predict


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
        bad, nonsense, good = '明天开会去银钱', '明天开会去银航', '明天开会去银行'
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.PAIRS') as pairs, \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(bad)), io.BytesIO(envelope(nonsense)),
                                                    io.BytesIO(envelope(good))]):
            pairs.loaded = True
            pairs.novel.side_effect = lambda text, earlier: ['银航'] if text == nonsense else []
            reply = predict('http://localhost', row)
        self.assertEqual(reply['candidates'], [good])
        self.assertEqual(reply['repair'], {'rounds': 2, 'before': 3, 'after': 0, 'first': bad})

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
        # Two rounds are tried; the second rewrite no longer explains the keys either (it drops one).
        self.assertIn('error', reply)
        self.assertNotIn('candidates', reply)
        self.assertEqual(len(self.requests(post)), 3)

    def test_a_repair_that_does_not_fit_better_is_discarded(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihuiquyinhang', 'context': ''}
        bad, worse = '明天开会去银钱', '明天开回去银钱'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope(bad)), io.BytesIO(envelope(worse)),
                                                    io.BytesIO(envelope(worse))]) as post:
            reply = predict('http://localhost', row)
        self.assertIn('error', reply)
        self.assertEqual(reply['rejected'], bad)
        self.assertEqual(len(self.requests(post)), 3)

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

    def test_unknown_decoder_is_rejected(self):
        with patch.dict(os.environ, IME_DECODER='other'):
            with self.assertRaisesRegex(ValueError, 'IME_DECODER'):
                predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'ni', 'context': ''})
