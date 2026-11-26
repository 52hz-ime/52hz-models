import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import URLError
from urllib.request import Request, urlopen

from infer import decision_contract, output_schema, run


ROOT = Path(__file__).resolve().parents[1]


class EvalWorkflowTests(unittest.TestCase):
    def test_decision_schema_by_event(self):
        for request, allowed in [({'event': 'type'}, ('preview',)), ({'event': 'more'}, ('ask',)),
                                 ({'event': 'space'}, ('commit', 'ask')),
                                 ({'event': 'select', 'origin': 'more'}, ('preview',)),
                                 ({'event': 'select', 'origin': 'space'}, ('commit', 'ask'))]:
            with self.subTest(request=request):
                actions, _ = decision_contract(request)
                schema = output_schema(actions)
                branches = schema.get('oneOf', [schema])
                self.assertEqual([branch['properties']['action']['enum'][0] for branch in branches], list(allowed))
                for action, branch in zip(allowed, branches):
                    fields = {'action', 'focus', 'options'} if action == 'ask' else {'action', 'candidates'}
                    self.assertEqual(set(branch['required']), fields)

    def test_eval_writes_separate_complete_runs(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/apply-template':
                    inputs.append(json.loads(payload['messages'][1]['content']))
                    response = {'prompt': '待转换拼音'}
                elif self.path == '/completion':
                    response = {'content': '无效' if len(inputs) == 1 else '明天开户"', 'completion_probabilities': []}
                else:
                    name = payload['response_format']['json_schema']['name']
                    content = '{"choice":"A","action":"commit"}' if name == 'ime_judge' else '{}'
                    response = {'choices': [{'message': {'content': content}}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, *args):
                pass

        inputs = []
        with tempfile.TemporaryDirectory() as directory, HTTPServer(('127.0.0.1', 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for target in ('eval', 'eval-strict'):
                    process = subprocess.run(['make', target, 'DECODER=ranked', f'PORT={server.server_port}', f'RUNS_DIR={directory}'],
                                             cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(process.returncode == 0, target == 'eval', process.stderr)
            finally:
                server.shutdown()
                worker.join()
            batches = list(Path(directory).iterdir())
            self.assertEqual(len(batches), 2)
            for batch in batches:
                requests = [json.loads(line) for line in (batch / 'requests.jsonl').read_text().splitlines()]
                replies = [json.loads(line) for line in (batch / 'responses.jsonl').read_text().splitlines()]
                self.assertEqual([row['id'] for row in requests], [row['id'] for row in replies])
                report = json.loads((batch / 'score.json').read_text())
                self.assertEqual(report['total'], len(requests))
                self.assertTrue(report['failed_ids'])
                failures = [json.loads(line) for line in (batch / 'failures.jsonl').read_text().splitlines()]
                self.assertEqual(sorted(row['id'] for row in failures), report['failed_ids'])
                self.assertTrue(all('expected' in row and 'actual' in row for row in failures))
            self.assertEqual(len(inputs), 2 * len(requests))
            self.assertTrue(all('keys' in row and 'expected' not in row and 'id' not in row for row in inputs))

    def test_parallel_requests_overlap_and_replies_keep_request_order(self):
        active, peak, lock = [0], [0], threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.05)
                with lock:
                    active[0] -= 1
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')

            def log_message(self, *args):
                pass

        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                root = Path(directory)
                source = root / 'requests.jsonl'
                source.write_text(''.join(json.dumps({'id': f'{number:03d}', 'keys': 'ni'}) + '\n' for number in range(12)))
                target = root / 'responses.jsonl'

                def predict(endpoint, row):
                    with urlopen(Request(endpoint + '/completion', data=b'{}'), timeout=5) as response:
                        json.load(response)
                    return {'id': row['id'], 'action': 'commit', 'candidates': [row['keys']]}

                with patch('infer.predict', predict):
                    run(f'http://127.0.0.1:{server.server_port}', source, target, workers=4)
            finally:
                server.shutdown()
                worker.join()
            replies = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual([row['id'] for row in replies], [f'{number:03d}' for number in range(12)])
            self.assertGreaterEqual(peak[0], 2)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_parallel_failure_keeps_the_replies_that_precede_it(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'requests.jsonl'
            source.write_text(''.join(json.dumps({'id': f'{number:03d}', 'keys': 'ni'}) + '\n' for number in range(8)))
            target = Path(directory) / 'responses.jsonl'

            def flaky(endpoint, row):
                if row['id'] == '004':
                    raise URLError('offline')
                return {'id': row['id']}

            with patch('infer.predict', flaky):
                with self.assertRaisesRegex(ValueError, '第 5 条请求未完成'):
                    run('http://localhost', source, target, workers=3)
            self.assertFalse(target.exists())
            partial, = Path(directory).glob('*.partial')
            self.assertEqual([json.loads(line)['id'] for line in partial.read_text().splitlines()], ['000', '001', '002', '003'])

    def test_eval_still_fails_when_service_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory, socket.socket() as reserved:
            reserved.bind(('127.0.0.1', 0))
            port = reserved.getsockname()[1]
            process = subprocess.run(['make', 'eval', f'PORT={port}', f'RUNS_DIR={directory}'],
                                     cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('make status', process.stderr)
            self.assertFalse(list(Path(directory).glob('*/score.json')))

    def test_transport_failure_preserves_previous_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'requests.jsonl'
            source.write_text('{"id":"first","keys":"ni"}\n{"id":"second","keys":"hao"}\n')
            target = Path(directory) / 'responses.jsonl'
            target.write_text('previous run\n')
            with patch('infer.predict', side_effect=[{'id': 'first'}, URLError('offline')]):
                with self.assertRaisesRegex(ValueError, 'make status'):
                    run('http://localhost', source, target)
            self.assertEqual(target.read_text(), 'previous run\n')
            partial, = Path(directory).glob('*.partial')
            self.assertEqual(json.loads(partial.read_text()), {'id': 'first'})

    def test_prepare_checks_checksum_and_reuses_download(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / 'bin'
            tools.mkdir()
            docker = tools / 'docker'
            docker.write_text('#!/bin/sh\ncase "$1" in image) test -e "$STATE/image";; pull) touch "$STATE/image";; *) exit 99;; esac\n')
            curl = tools / 'curl'
            curl.write_text('#!/bin/sh\nwhile [ "$1" != -o ]; do shift; done\nprintf fixture > "$2"\necho download >> "$STATE/calls"\n')
            docker.chmod(0o755)
            curl.chmod(0o755)
            env = dict(os.environ, STATE=directory, PATH=str(tools) + os.pathsep + os.environ['PATH'])
            command = ['make', 'prepare', f'MODEL_DIR={root}/models', 'HF_FILE=test.gguf']
            def prepare(digest):
                return subprocess.run([*command, f'HF_SHA256={digest}'], cwd=ROOT, env=env, capture_output=True)
            digest = hashlib.sha256(b'fixture').hexdigest()
            self.assertNotEqual(prepare('0' * 64).returncode, 0)
            self.assertFalse((root / 'models/test.gguf').exists())
            self.assertEqual(prepare(digest).returncode, 0)
            self.assertEqual(prepare(digest).returncode, 0)
            self.assertEqual((root / 'models/test.gguf').read_bytes(), b'fixture')
            self.assertEqual((root / 'calls').read_text().splitlines(), ['download', 'download'])
