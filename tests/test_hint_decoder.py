import io
import json
import os
import unittest
from unittest.mock import patch

from infer import HINT_EXAMPLES, predict


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
        self.assertEqual(reply, {'action': 'preview', 'candidates': ['今天晚上吃饭。'], 'id': 'case'})
        self.assertEqual(self.user_text(first),
                         '上文：上次聊到吃饭。\n记忆：他不吃辣。\n已选定：吃饭（按键 chifan，不可改动）\n'
                         '按键：jintianwanshangchifann\n切分：jin tian wan shang chi fan?')
        self.assertEqual(len(first['messages']), 1 + 2 * len(HINT_EXAMPLES) + 1)
        self.assertTrue(all(message['content'] for message in first['messages'][1:-1:2]))

    def test_space_commits_the_reply_with_one_call_and_never_asks(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'kelashi', 'context': '课还没开始。'}
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope('柯老师'))) as post:
            reply = predict('http://localhost', row)
        self.assertEqual(reply, {'action': 'commit', 'candidates': ['柯老师'], 'id': 'case'})
        self.assertNotIn('response_format', self.requests(post)[0])

    def test_a_reply_that_does_not_read_like_the_keys_is_kept_back(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'nikanjiushiwangji', 'context': ''}
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope('你简直就是王八犊子，'))) as post:
            reply = predict('http://localhost', row)
        self.assertIn('error', reply)
        self.assertNotIn('candidates', reply)
        self.assertEqual(len(self.requests(post)), 1)

    def test_a_dropped_selection_falls_back_to_the_json_contract(self):
        row = {'id': 'case', 'event': 'space', 'keys': 'mingtiankaihu', 'context': '银行已经预约好了。', 'selected': {'kaihu': '开会'}}
        answer = '{"action":"commit","candidates":["明天开会"]}'
        with patch.dict(os.environ, IME_DECODER='hint'), \
                patch('infer.urlopen', side_effect=[io.BytesIO(envelope('明天开户')), io.BytesIO(envelope(answer))]) as post:
            reply = predict('http://localhost', row)
        self.assertEqual(reply['candidates'], ['明天开会'])
        fallback, = self.requests(post)[1:]
        self.assertEqual(fallback['response_format']['json_schema']['name'], 'ime_decision')
        self.assertEqual(json.loads(self.user_text(fallback))['selected'], {'kaihu': '开会'})

    def test_asking_on_demand_keeps_the_json_contract(self):
        row = {'id': 'case', 'event': 'more', 'keys': 'kaihu', 'focus': 'kaihu'}
        answer = '{"action":"ask","focus":"kaihu","options":["开会","开户"]}'
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope(answer))) as post:
            reply = predict('http://localhost', row)
        self.assertEqual(reply['action'], 'ask')
        payload, = self.requests(post)
        self.assertEqual(payload['response_format']['json_schema']['name'], 'ime_decision')
        self.assertNotIn('切分', self.user_text(payload))

    def test_empty_reply_is_reported_instead_of_committed(self):
        with patch.dict(os.environ, IME_DECODER='hint'), patch('infer.urlopen', return_value=io.BytesIO(envelope('  '))):
            reply = predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'nihaoma', 'context': ''})
        self.assertIn('error', reply)
        self.assertNotIn('candidates', reply)

    def test_unknown_decoder_is_rejected(self):
        with patch.dict(os.environ, IME_DECODER='other'):
            with self.assertRaisesRegex(ValueError, 'IME_DECODER'):
                predict('http://localhost', {'id': 'case', 'event': 'space', 'keys': 'ni', 'context': ''})
