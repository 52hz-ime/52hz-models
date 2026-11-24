import json
import unittest
from unittest.mock import Mock, patch

from candidate_search import forks, generate, ranked


def token(piece, number, score, alternatives=()):
    return {'id': number, 'token': piece, 'bytes': list(piece.encode()), 'logprob': score, 'top_logprobs': alternatives}


class CandidateTests(unittest.TestCase):
    def test_repeated_token_can_fork_before_a_locally_unlikely_word(self):
        alternate = token('犹豫', 4, -3)
        primary = token('用', 2, -1, [alternate])
        rows = [token('我在', 1, -1, [token('我再', 3, -1.1)]), primary, primary]
        self.assertEqual(forks('我在用用哪个"', rows)[0], '我在犹豫')

    def test_separate_continuations_deduplicate_and_preserve_locks(self):
        post = Mock(side_effect=[{'prompt': 'raw'}, {'content': '找李鑫"', 'completion_probabilities': []},
                                 {'content': '李欣"'}, {'content': '注册"'}, {'content': '注册"'}])
        with patch('candidate_search.forks', return_value=['找', '找李鑫', '找李鑫']):
            pool = generate(post, [], ['李鑫'])
        self.assertEqual(pool, ['找李鑫', '找李鑫注册'])
        self.assertTrue(post.call_args_list[-1].args[1]['prompt'].endswith('找李鑫'))

    def test_judge_can_only_pick_from_complete_candidates(self):
        fields = {'keys': 'wozaixx', 'context': '上文', 'selected': {}}
        pool = ['我在想', '我在笑']
        chat = Mock(return_value={'action': 'commit', 'choice': 'B', 'candidates': ['无中生有']})
        with patch('candidate_search.generate', return_value=pool) as generate_pool:
            reply = ranked(None, chat, fields, '规则', ('commit', 'ask'), '提交或问一处')
            blind = json.loads(generate_pool.call_args.args[1][1]['content'])
            self.assertEqual(blind, fields)
            self.assertEqual(chat.call_args.args[1]['keys'], fields['keys'])
            self.assertEqual(reply['candidates'][0], sorted(pool)[1])
            self.assertEqual(reply['search']['pool'], pool)
            chat.return_value = {'action': 'commit', 'choice': 'Z'}
            self.assertIn('error', ranked(None, chat, fields, '规则', ('commit',), '提交'))

    def test_ask_options_are_local_and_backed_by_the_pool(self):
        fields = {'keys': 'lixin', 'focus': 'lixin'}
        with patch('candidate_search.generate', return_value=['李欣']):
            chat = Mock(side_effect=[{'sentences': ['李鑫', '李欣']},
                                     {'action': 'ask', 'choice': 'A', 'focus': 'lixin', 'options': ['李欣', '李鑫', '王飞']}])
            reply = ranked(None, chat, fields, '规则', ('ask',), '问此处')
        self.assertEqual(reply['options'], ['李欣', '李鑫'])
