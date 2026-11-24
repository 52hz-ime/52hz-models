import io
import json
import os
import unittest
from unittest.mock import patch

from infer import predict


class RawKeysTests(unittest.TestCase):
    def test_send_original_keystrokes_without_splitting_or_repair(self):
        envelope = json.dumps({'choices': [{'message': {'content': '{"action":"commit","candidates":["测试"]}'}}]}).encode()
        for keys in ('heAtlas2.1bwozaiyouyuyongnage', 'heAtlas2.1bwozaiyouyuyongnge', 'Atlas2.1 wo zai'):
            for event in ('type', 'space'):
                row = {'id': 'case', 'event': event, 'keys': keys, 'context': '备选模型有Atlas4.0。'}
                with self.subTest(keys=keys, event=event), patch.dict(os.environ, IME_DECODER='direct'), patch('infer.urlopen', return_value=io.BytesIO(envelope)) as post:
                    predict('http://localhost', row)
                    fields = json.loads(json.loads(post.call_args.args[0].data)['messages'][1]['content'])
                    self.assertEqual(fields['keys'], keys)
