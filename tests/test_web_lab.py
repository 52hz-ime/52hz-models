import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from unittest.mock import patch

from web_lab import LabHandler


class LabApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), LabHandler)
        cls.server.endpoint = 'http://127.0.0.1:8089'
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/api/infer'
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.worker.join()
        cls.server.server_close()

    def post(self, row, **headers):
        request = Request(self.url, data=json.dumps(row).encode(), headers={'Content-Type': 'application/json', **headers})
        return urlopen(request, timeout=3)

    def test_adapter_receives_same_selected_snapshot_as_eval(self):
        row = {'id': 'ui-3', 'keys': 'mingtiankaihu', 'event': 'select', 'origin': 'more', 'selected': {'kaihu': '开会'}}
        answer = {'id': row['id'], 'action': 'preview', 'candidates': ['明天开会']}
        with patch('web_lab.predict', return_value=answer) as predict, self.post(row) as response:
            self.assertEqual(json.load(response), answer)
            predict.assert_called_once_with(self.server.endpoint, row)

    def test_invalid_or_cross_origin_input_never_calls_model(self):
        row = {'id': 'ui-1', 'keys': 'ni', 'event': 'more', 'focus': 'other'}
        with patch('web_lab.predict') as predict:
            for headers, status in [({}, 400), ({'Origin': 'https://example.org'}, 403)]:
                with self.assertRaises(HTTPError) as error:
                    self.post(row, **headers)
                with error.exception as response:
                    self.assertEqual(response.code, status)
            predict.assert_not_called()

    def test_offline_model_is_reported(self):
        with patch('web_lab.predict', side_effect=URLError('offline')), self.assertRaises(HTTPError) as error:
            self.post({'id': 'ui-1', 'keys': 'ni', 'event': 'type'})
        with error.exception as response:
            self.assertEqual(response.code, 502)
            self.assertIn('make status', response.read().decode())
